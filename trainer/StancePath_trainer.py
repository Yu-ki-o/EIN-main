"""Full-path teacher followed by a sparsely gated StancePath student."""

from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from model.StancePath import StancePath, compute_stance_path_loss
from utils.logger import get_log_dir, get_logger


def load_stance_path_checkpoint(path, device="cpu"):
    """Reconstruct the student, including its threshold, without a teacher."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != "stance_path_v1":
        raise ValueError("Expected a StancePath checkpoint with model configuration")
    config = checkpoint["model_config"]
    model = StancePath(config["in_feats"], config["hidden_dim"], config["num_classes"],
                       SimpleNamespace(**config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class StancePathTrainer:
    """Select each stage on validation only; use test data after both stages."""
    def __init__(self, datasets, model, args, device):
        if len(datasets) != 3:
            raise ValueError("Supply train, validation, and test datasets")
        self.args, self.device = args, torch.device(device)
        self.model, self.teacher = model.to(self.device), None
        self.eval_only = bool(getattr(args, "eval_only", False))
        self.selection_metric = getattr(args, "selection_metric", "val_loss")
        if self.selection_metric not in {"val_loss", "val_acc", "val_auc", "val_f1"}:
            raise ValueError("Unsupported validation checkpoint selection metric")
        self.grad_clip = float(getattr(args, "stance_path_grad_clip", 5.0))
        self.micro_batch = int(getattr(args, "stance_path_micro_batch_size", 1))
        if self.micro_batch < 1 or not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
            raise ValueError("Positive micro batch size and gradient clip are required")
        self.log_dir = Path(get_log_dir(args))
        if self.eval_only:
            cutoff = Path(str(getattr(args, "early_test_root", "test")).rstrip("/\\")).name
            self.log_dir = self.log_dir / "early_detection" / (cutoff or "test")
            self.model, _ = load_stance_path_checkpoint(args.checkpoint_path, self.device)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(str(self.log_dir))
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        self.logger = get_logger(str(self.log_dir), name=str(self.log_dir), debug=False)
        self.teacher_path = self.log_dir / "best_teacher.pth"
        self.best_path = self.log_dir / "best_model.pth"
        self.history = []
        generator = torch.Generator().manual_seed(int(getattr(args, "seed", 0)))
        self.loaders = {}
        for name, dataset in zip(("train", "val", "test"), datasets):
            if self.eval_only and name != "test":
                continue
            if dataset is None or not len(dataset):
                raise ValueError("The {} dataset is empty".format(name))
            # Validate/extract topology once on CPU; do not repeat Python tree
            # traversal or device-to-host synchronization in every forward pass.
            trees = []
            for index in range(len(dataset)):
                try:
                    tree = self.model.prepare(dataset[index])[0]
                    self._targets([tree])
                    if tree.x.size(1) != self.model.in_feats:
                        raise ValueError("Feature width does not match in_feats")
                    trees.append(tree)
                except ValueError as error:
                    raise ValueError("{} event {}: {}".format(name, index, error)) from error
            self.loaders[name] = DataLoader(
                trees, batch_size=int(args.batch_size), shuffle=name == "train",
                collate_fn=list, num_workers=int(getattr(args, "num_workers", 0)),
                generator=generator if name == "train" else None,
            )
            self.logger.info("Prepared %s: %d events, %d candidate paths", name, len(trees),
                             sum(len(tree.paths) for tree in trees))

    def _targets(self, trees):
        labels = []
        for tree in trees:
            if tree.y is None or tree.y.numel() != 1:
                raise ValueError("One event label is required for training/evaluation metrics")
            label = tree.y.item()
            if not math.isfinite(label) or label != int(label) or not 0 <= label < self.model.num_classes:
                raise ValueError("Event label must be an integer class index")
            labels.append(int(label))
        return torch.tensor(labels, device=self.device, dtype=torch.long)

    def _save(self, model, path, stage, epoch, value):
        config = {key: value for key, value in vars(self.args).items()
                  if key.startswith("stance_path_") or key == "dropout"}
        config.update(in_feats=model.in_feats, hidden_dim=model.hidden_dim,
                      num_classes=model.num_classes, stance_path_threshold=model.threshold)
        torch.save(dict(format="stance_path_v1", stage=stage, epoch=epoch,
                        selection_metric=self.selection_metric, selection_value=value,
                        model_config=config, model_state_dict=model.state_dict()), path)

    def _write_json(self, filename, value):
        (self.log_dir / filename).write_text(
            json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def evaluate(self, model, loader, *, full_paths=False, source_only=False):
        model.eval()
        predictions, targets = [], []
        total_loss, count, paths, selected_paths, expected_paths = 0.0, 0, 0, 0.0, 0.0
        nodes, selected_nodes, root_only = 0, 0, 0
        self._synchronize()
        started = time.perf_counter()
        for batch in loader:
            for offset in range(0, len(batch), self.micro_batch):
                trees = batch[offset:offset + self.micro_batch]
                target = self._targets(trees)
                logits, details = model(trees, full_paths=full_paths, source_only=source_only,
                                        return_details=True)
                total_loss += F.cross_entropy(logits, target, reduction="sum").item()
                count += len(trees)
                predictions.append(logits.cpu())
                targets.append(target.cpu())
                for detail in details:
                    paths += detail["num_paths"]
                    selected_paths += detail["path_mask"].sum().item()
                    expected_paths += detail["expected_paths"].item()
                    nodes += detail["num_nodes"]
                    kept = detail["selected_node_ids"].numel()
                    selected_nodes += kept
                    root_only += int(kept == 1)
        self._synchronize()
        seconds = time.perf_counter() - started
        logits, target = torch.cat(predictions), torch.cat(targets)
        prediction = logits.argmax(-1)
        # Match existing EIN AUC/F1 for comparisons, and additionally report
        # probability-based AUC explicitly so the two definitions are visible.
        auc, probability_auc = float("nan"), float("nan")
        if model.num_classes == 2 and target.unique().numel() == 2:
            auc = float(roc_auc_score(target.numpy(), prediction.numpy()))
            probability_auc = float(roc_auc_score(target.numpy(), logits.softmax(-1)[:, 1].numpy()))
        metrics = dict(
            loss=total_loss / count, acc=float(accuracy_score(target, prediction)), auc=auc,
            auc_probability=probability_auc,
            f1=float(f1_score(target, prediction, average="binary" if model.num_classes == 2 else "macro", zero_division=0)),
            macro_f1=float(f1_score(target, prediction, average="macro", zero_division=0)),
            retained_path_ratio=selected_paths / paths if paths else 0.0,
            retained_node_ratio=selected_nodes / nodes,
            source_only_ratio=root_only / count, expected_paths_per_event=expected_paths / count,
            selected_paths_per_event=selected_paths / count, inference_seconds=seconds,
            milliseconds_per_event=1000 * seconds / count,
        )
        if model.num_classes == 2:
            for label in (0, 1):
                metrics["f1_class{}".format(label)] = float(f1_score(
                    target, prediction, labels=[label], average=None, zero_division=0)[0])
        return metrics

    def _train_stage(self, model, stage, path, teacher=None):
        epochs = int(getattr(self.args, "stance_path_{}_epochs".format(stage), self.args.n_epochs))
        patience = int(getattr(self.args, "stance_path_{}_patience".format(stage), self.args.patience))
        lr = float(getattr(self.args, "stance_path_{}_lr".format(stage), self.args.lr))
        if min(epochs, patience) < 1 or not math.isfinite(lr) or lr <= 0:
            raise ValueError("Each training stage needs positive epochs, patience, and learning rate")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=getattr(self.args, "weight_decay", 0.0))
        best, stale = None, 0
        full_paths = stage == "teacher"
        for epoch in range(1, epochs + 1):
            model.train()
            sums = dict(total=0.0, classification=0.0, sufficiency=0.0, mask_kl=0.0)
            count = 0
            for batch in self.loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, len(batch), self.micro_batch):
                    trees = batch[offset:offset + self.micro_batch]
                    target = self._targets(trees)
                    logits, details = model(trees, full_paths=full_paths, return_details=True)
                    if full_paths:
                        loss = F.cross_entropy(logits, target)
                        terms = dict(classification=loss, sufficiency=loss.new_zeros(()), mask_kl=loss.new_zeros(()))
                    else:
                        with torch.no_grad():
                            teacher_logits = teacher(trees, full_paths=True)
                        loss, terms = compute_stance_path_loss(
                            logits, target, details, teacher_logits,
                            sufficiency_weight=model.sufficiency_weight,
                            sufficiency_margin=model.sufficiency_margin, kl_weight=model.kl_weight)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite {} loss at epoch {}".format(stage, epoch))
                    # Backpropagate each microbatch immediately, retaining only
                    # one event's attention graph at the default microbatch=1.
                    (loss * len(trees) / len(batch)).backward()
                    count += len(trees)
                    sums["total"] += loss.item() * len(trees)
                    for name, value in terms.items():
                        sums[name] += value.item() * len(trees)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
            validation = self.evaluate(model, self.loaders["val"], full_paths=full_paths)
            row = dict(stage=stage, epoch=epoch, train={key: value / count for key, value in sums.items()},
                       validation=validation)
            self.history.append(row)
            self._write_json("history.json", self.history)
            key = self.selection_metric.removeprefix("val_")
            value = validation[key]
            if not math.isfinite(value):
                raise ValueError("Non-finite selection metric {}".format(self.selection_metric))
            improved = best is None or (value < best if key == "loss" else value > best)
            if improved:
                best, stale = value, 0
                self._save(model, path, stage, epoch, value)
            else:
                stale += 1
            self.logger.info(
                "%s epoch=%d loss=%.5f CE=%.5f suff=%.5f KL=%.5f val_loss=%.5f paths=%.3f nodes=%.3f root_only=%.3f",
                stage, epoch, row["train"]["total"], row["train"]["classification"],
                row["train"]["sufficiency"], row["train"]["mask_kl"], validation["loss"],
                validation["retained_path_ratio"], validation["retained_node_ratio"], validation["source_only_ratio"])
            if stale >= patience:
                break
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    def train_process(self):
        if not self.eval_only:
            self.teacher = deepcopy(self.model)
            self._train_stage(self.teacher, "teacher", self.teacher_path)
            self.teacher.eval().requires_grad_(False)
            self.model.load_state_dict(self.teacher.state_dict())
            self._train_stage(self.model, "student", self.best_path, self.teacher)
        # Only now read the held-out loader for metrics. Inference needs no
        # teacher, ground-truth label, or selector set summary as classifier input.
        result = self.evaluate(self.model, self.loaders["test"])
        source_only = self.evaluate(self.model, self.loaders["test"], source_only=True)
        report = dict(student=result, source_only=source_only,
                      checkpoint=str(self.best_path if not self.eval_only else self.args.checkpoint_path),
                      auc_definition="auc uses hard predictions; auc_probability uses class probabilities")
        if self.teacher is not None:
            report["full_teacher"] = self.evaluate(self.teacher, self.loaders["test"], full_paths=True)
        self._write_json("test_results.json", report)
        self.logger.info("Student test: %s", result)
        return result
