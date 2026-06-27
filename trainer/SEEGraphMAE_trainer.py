import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

from utils.earlystopping import EarlyStopping
from utils.logger import get_log_dir, get_logger


SELECTION_METRIC_MODES = {
    "val_loss": "min",
    "val_acc": "max",
    "val_auc": "max",
    "val_f1": "max",
}


class SEEGraphMAETrainer(object):
    def __init__(self, datasets, model, optimizer, args, device):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.device = device

        train_dataset, val_dataset, test_dataset = datasets
        loader_kwargs = self._loader_kwargs()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_kwargs
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            **loader_kwargs
        )
        test_batch_size = int(
            getattr(
                args,
                "see_ttt_batch_size",
                args.batch_size,
            )
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            **loader_kwargs
        )
        self.train_per_epoch = len(self.train_loader)

        args.log_dir = get_log_dir(args)
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, "best_model.pth")

        self.logger.info("Experiment log path in: {}".format(args.log_dir))
        self.logger.info("Experiment configs are: {}".format(args))
        self.logger.info(
            "Runtime device: {} | pin_memory: {} | non_blocking transfer: {}".format(
                self.device,
                self.device.type == "cuda",
                self.device.type == "cuda",
            )
        )

    def _loader_kwargs(self):
        num_workers = max(0, int(getattr(self.args, "num_workers", 0)))
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(
                getattr(self.args, "persistent_workers", True)
            )
        return kwargs

    def _move_to_device(self, data):
        return data.to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )

    def get_selection_metric(self):
        metric = getattr(self.args, "selection_metric", "val_loss")
        if metric is None:
            metric = "val_loss"
        metric = str(metric).strip()
        if metric not in SELECTION_METRIC_MODES:
            raise ValueError(
                "selection_metric must be one of {}, got {}".format(
                    sorted(SELECTION_METRIC_MODES),
                    metric,
                )
            )
        return metric

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0

        for data in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            data = self._move_to_device(data)
            out = self.model(data)
            cls_loss = self.model.classification_loss(out, data.y)
            ssl_loss, _ = self.model.self_supervised_loss(data)
            loss = cls_loss + ssl_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite SEEGraphMAE loss detected at epoch {}.".format(
                        epoch
                    )
                )
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        train_epoch_loss = train_loss / self.train_per_epoch
        self.logger.info(
            "*******Traininig Epoch {}: averaged Loss : {:.6f}".format(
                epoch,
                train_epoch_loss,
            )
        )
        return train_epoch_loss

    def validate_epoch(self, epoch):
        val_losses = []
        y_true = []
        y_pred = []
        self.model.eval()

        with torch.no_grad():
            for data in self.val_loader:
                data = self._move_to_device(data)
                out = self.model(data)
                val_loss = self.model.classification_loss(out, data.y)
                val_losses.append(val_loss.item())
                y_true += data.y.view(-1).tolist()
                y_pred += out.max(1).indices.tolist()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(y_true, y_pred)
        try:
            val_auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            val_auc = np.nan
        val_f1 = f1_score(y_true, y_pred, zero_division=0)
        val_metrics = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_f1": val_f1,
        }
        self.logger.info(
            "*******Val Epoch {}: Loss {:.6f} | Acc {:.4f} | AUC {:.4f} | F1 {:.4f}".format(
                epoch,
                val_loss,
                val_acc,
                val_auc,
                val_f1,
            )
        )
        return val_metrics

    def _clone_state_dict(self):
        return {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

    def _frozen_reference_model(self):
        reference_model = copy.deepcopy(self.model).to(self.device)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad_(False)
        return reference_model

    def _set_requires_grad(self, model, requires_grad):
        states = [param.requires_grad for param in model.parameters()]
        for param in model.parameters():
            param.requires_grad_(requires_grad)
        return states

    def _restore_requires_grad(self, model, states):
        for param, requires_grad in zip(model.parameters(), states):
            param.requires_grad_(requires_grad)

    def _optimize_attention_mask(self, model, data):
        steps = int(getattr(self.args, "see_subgraph_mask_steps", 10))
        if steps <= 0:
            return data.x.new_ones(data.x.size(0))

        was_training = model.training
        model.eval()
        states = self._set_requires_grad(model, False)
        try:
            with torch.no_grad():
                target = F.log_softmax(
                    model.predict_logits(data),
                    dim=-1,
                )

            init_value = float(getattr(self.args, "see_subgraph_mask_init", 2.0))
            mask_logits = torch.full(
                (data.x.size(0),),
                init_value,
                dtype=data.x.dtype,
                device=data.x.device,
                requires_grad=True,
            )
            optimizer = torch.optim.Adam(
                [mask_logits],
                lr=float(getattr(self.args, "see_subgraph_mask_lr", 0.1)),
            )
            sparsity = float(getattr(self.args, "see_subgraph_sparsity", 0.1))

            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                mask = torch.sigmoid(mask_logits)
                masked_target = F.log_softmax(
                    model.predict_logits(data, node_mask=mask),
                    dim=-1,
                )
                preserve_loss = F.mse_loss(masked_target, target)
                sparse_loss = sparsity * mask.pow(2).mean()
                (preserve_loss + sparse_loss).backward()
                optimizer.step()

            return torch.sigmoid(mask_logits).detach()
        finally:
            self._restore_requires_grad(model, states)
            model.train(was_training)

    def _ttt_optimizer(self):
        params = list(self.model.self_supervised_parameters())
        return torch.optim.Adam(
            params,
            lr=float(getattr(self.args, "see_ttt_lr", self.args.lr)),
            weight_decay=float(getattr(self.args, "see_ttt_weight_decay", 0.0)),
        )

    def _adapt_test_batch(self, data, trained_state, reference_model):
        if bool(getattr(self.args, "see_ttt_reset_each_batch", True)):
            self.model.load_state_dict(trained_state)

        epochs = int(getattr(self.args, "see_ttt_epochs", 3))
        if epochs <= 0:
            return

        alpha_sub = float(getattr(self.args, "see_alpha_sub", 0.5))
        use_subgraph = bool(getattr(self.args, "see_use_subgraph_regularizer", True))
        reference_mask = None
        reference_masked_output = None

        if use_subgraph and alpha_sub > 0.0:
            reference_mask = self._optimize_attention_mask(reference_model, data)
            with torch.no_grad():
                reference_masked_output = F.log_softmax(
                    reference_model.predict_logits(
                        data,
                        node_mask=reference_mask,
                    ),
                    dim=-1,
                )

        optimizer = self._ttt_optimizer()
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            ssl_loss, _ = self.model.self_supervised_loss(data)
            loss = ssl_loss

            if reference_mask is not None:
                was_training = self.model.training
                self.model.eval()
                current_masked_output = F.log_softmax(
                    self.model.predict_logits(
                        data,
                        node_mask=reference_mask,
                    ),
                    dim=-1,
                )
                self.model.train(was_training)
                subgraph_loss = F.mse_loss(
                    current_masked_output,
                    reference_masked_output,
                )
                loss = loss + alpha_sub * subgraph_loss

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SEEGraphMAE TTT loss.")
            loss.backward()
            optimizer.step()

    def _evaluate_predictions(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = np.nan
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {"acc": acc, "auc": auc, "f1": f1}

    def _predict_loader_without_ttt(self):
        y_true = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                data = self._move_to_device(data)
                out = self.model(data)
                y_true += data.y.view(-1).tolist()
                y_pred += out.max(1).indices.tolist()
        return self._evaluate_predictions(y_true, y_pred)

    def _log_test_metrics(self, prefix, metrics):
        self.logger.info(
            "{} Acc: {:.4f} | AUC: {:.4f} | F1 {:.4f}".format(
                prefix,
                metrics["acc"],
                metrics["auc"],
                metrics["f1"],
            )
        )

    def test(self):
        y_true = []
        y_pred = []
        ttt_enabled = bool(getattr(self.args, "see_ttt_enabled", True))
        plain_metrics = None

        if ttt_enabled:
            trained_state = self._clone_state_dict()
            reference_model = self._frozen_reference_model()
            if bool(getattr(self.args, "see_log_plain_test_before_ttt", True)):
                plain_metrics = self._predict_loader_without_ttt()
                self._log_test_metrics("Test without TTT", plain_metrics)
                self.model.load_state_dict(trained_state)
            for data in self.test_loader:
                data = self._move_to_device(data)
                self._adapt_test_batch(data, trained_state, reference_model)
                self.model.eval()
                with torch.no_grad():
                    out = self.model(data)
                y_true += data.y.view(-1).tolist()
                y_pred += out.max(1).indices.tolist()
            self.model.load_state_dict(trained_state)
        else:
            self.model.eval()
            with torch.no_grad():
                for data in self.test_loader:
                    data = self._move_to_device(data)
                    out = self.model(data)
                    y_true += data.y.view(-1).tolist()
                    y_pred += out.max(1).indices.tolist()

        metrics = self._evaluate_predictions(y_true, y_pred)
        self._log_test_metrics("Test", metrics)
        if plain_metrics is not None:
            metrics.update(
                {
                    "without_ttt_acc": plain_metrics["acc"],
                    "without_ttt_auc": plain_metrics["auc"],
                    "without_ttt_f1": plain_metrics["f1"],
                }
            )
        return metrics

    def train_process(self):
        start_time = time.time()
        selection_metric = self.get_selection_metric()
        selection_mode = SELECTION_METRIC_MODES[selection_metric]
        self.logger.info(
            "Checkpoint selection metric: {} ({})".format(
                selection_metric,
                selection_mode,
            )
        )
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            mode=selection_mode,
            metric_name=selection_metric,
        )

        for epoch in range(self.args.n_epochs):
            train_epoch_loss = self.train_epoch(epoch)
            if train_epoch_loss > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break

            val_metrics = self.validate_epoch(epoch)
            early_stopping(
                val_metrics[selection_metric],
                self.model,
                epoch,
                self.best_path,
            )
            if early_stopping.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. Training stops.".format(
                        self.args.patience
                    )
                )
                break

        training_time = time.time() - start_time
        self.logger.info(
            "== Training finished.\n"
            "Total training time: {:.2f} min\t"
            "best {}: {:.4f}\t"
            "best epoch: {}\t".format(
                training_time / 60,
                selection_metric,
                early_stopping.best_value,
                early_stopping.best_epoch,
            )
        )

        best_model_path = self.best_path + ".m"
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            self.logger.info(
                "Loaded best checkpoint for testing: {}".format(best_model_path)
            )

        return self.test()
