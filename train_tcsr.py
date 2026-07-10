"""Minimal training script for TCSR.

The script accepts either explicit train/val/test dataset paths or a single
data path that will be split repeatedly for multi-seed experiments.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader

from model.model_tcsr import TCSRModel, compute_tcsr_loss
from utils_metrics import classification_metrics, mean_std


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(data, device):
    try:
        return data.to(device, non_blocking=device.type == "cuda")
    except TypeError:
        return data.to(device)


def load_graph_dataset(path):
    path = _resolve_dataset_file(path)
    obj = _torch_load(path)
    return _coerce_to_data_list(obj)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    stance_loss_weight=1.0,
    grad_clip=0.0,
):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for data in loader:
        data = move_to_device(data, device)
        optimizer.zero_grad(set_to_none=True)
        logits, aux_outputs = model(data)
        loss, _ = compute_tcsr_loss(
            logits,
            data,
            aux_outputs,
            stance_loss_weight=stance_loss_weight,
        )
        loss.backward()
        if grad_clip and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = int(data.y.view(-1).size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, stance_loss_weight=1.0):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits = []
    all_targets = []
    for data in loader:
        data = move_to_device(data, device)
        logits, aux_outputs = model(data)
        loss, _ = compute_tcsr_loss(
            logits,
            data,
            aux_outputs,
            stance_loss_weight=stance_loss_weight,
        )
        target = data.y.view(-1).long()
        batch_size = int(target.size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach())
        all_targets.append(target.detach())

    if all_logits:
        logits = torch.cat(all_logits, dim=0)
        target = torch.cat(all_targets, dim=0)
        metrics = classification_metrics(
            logits,
            target,
            num_classes=num_classes,
        )
    else:
        metrics = {"accuracy": 0.0, "macro_f1": 0.0}
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics


def run_seed(seed, args, datasets, device):
    set_seed(seed)
    train_dataset, val_dataset, test_dataset = datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    input_dim = int(train_dataset[0].x.size(-1))
    model = TCSRModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        conv_type=args.conv_type,
        window_k=args.window_k,
        min_future_nodes=args.min_future_nodes,
        use_anomaly=args.use_anomaly,
        use_isolation=args.use_isolation,
        use_dominance=args.use_dominance,
        use_threshold=args.use_threshold,
        use_external_stance=args.use_external_stance,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"tcsr_seed{seed}.pt")
    best_val = -float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            stance_loss_weight=args.stance_loss_weight,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.num_classes,
            stance_loss_weight=args.stance_loss_weight,
        )
        score = val_metrics[args.selection_metric]
        if score > best_val:
            best_val = score
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "seed": seed,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            patience_counter += 1

        print(
            "Seed {seed} Epoch {epoch:03d} | train loss {train_loss:.4f} "
            "| val loss {loss:.4f} | val acc {accuracy:.4f} "
            "| val macro-F1 {macro_f1:.4f}".format(
                seed=seed,
                epoch=epoch,
                train_loss=train_loss,
                **val_metrics,
            )
        )
        if patience_counter >= args.patience:
            break

    checkpoint = _torch_load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        args.num_classes,
        stance_loss_weight=args.stance_loss_weight,
    )
    print(
        "Seed {seed} best epoch {epoch} | test acc {accuracy:.4f} "
        "| test macro-F1 {macro_f1:.4f}".format(
            seed=seed,
            epoch=best_epoch,
            **test_metrics,
        )
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_loss": test_metrics["loss"],
        "checkpoint": best_path,
    }


def build_datasets(args, seed):
    if args.train_path and args.val_path and args.test_path:
        return (
            load_graph_dataset(args.train_path),
            load_graph_dataset(args.val_path),
            load_graph_dataset(args.test_path),
        )
    if args.dataset_dir:
        train_path, val_path, test_path = _paths_from_dataset_dir(args.dataset_dir)
        return (
            load_graph_dataset(train_path),
            load_graph_dataset(val_path),
            load_graph_dataset(test_path),
        )
    if not args.data_path:
        raise ValueError(
            "Provide either --train_path/--val_path/--test_path, "
            "--dataset_dir, or --data_path."
        )

    dataset = load_graph_dataset(args.data_path)
    return split_dataset(dataset, args.train_ratio, args.val_ratio, seed)


def split_dataset(dataset, train_ratio, val_ratio, seed):
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_end = int(len(indices) * float(train_ratio))
    val_end = train_end + int(len(indices) * float(val_ratio))
    train_ids = indices[:train_end]
    val_ids = indices[train_end:val_end]
    test_ids = indices[val_end:]
    return (
        [dataset[index] for index in train_ids],
        [dataset[index] for index in val_ids],
        [dataset[index] for index in test_ids],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train TCSR rumor detector")
    parser.add_argument("--data_path", default=None, help="Single .pt dataset")
    parser.add_argument("--dataset_dir", default=None, help="Directory with train/val/test processed data")
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--val_path", default=None)
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints/tcsr")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--conv_type", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--window_k", type=int, default=2)
    parser.add_argument("--min_future_nodes", type=int, default=1)
    parser.add_argument("--stance_loss_weight", type=float, default=1.0)
    parser.add_argument("--selection_metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--use_anomaly", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_isolation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_dominance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_external_stance", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    results = []
    for seed in seeds:
        datasets = build_datasets(args, seed)
        if len(datasets[0]) == 0 or len(datasets[1]) == 0 or len(datasets[2]) == 0:
            raise ValueError("Train/val/test splits must all be non-empty.")
        results.append(run_seed(seed, args, datasets, device))

    acc_mean, acc_std = mean_std([item["test_accuracy"] for item in results])
    f1_mean, f1_std = mean_std([item["test_macro_f1"] for item in results])
    print("TCSR {} seed summary".format(len(results)))
    print("Accuracy: {:.4f} +/- {:.4f}".format(acc_mean, acc_std))
    print("Macro-F1: {:.4f} +/- {:.4f}".format(f1_mean, f1_std))


def _paths_from_dataset_dir(dataset_dir):
    dataset_dir = Path(dataset_dir)
    paths = []
    for split in ("train", "val", "test"):
        candidates = [
            dataset_dir / split / "processed" / "data.pt",
            dataset_dir / f"{split}.pt",
            dataset_dir / split / "data.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                paths.append(candidate)
                break
        else:
            raise FileNotFoundError(
                "Could not find {} split under {}".format(split, dataset_dir)
            )
    return tuple(paths)


def _resolve_dataset_file(path):
    path = Path(path)
    if path.is_dir():
        candidates = [path / "processed" / "data.pt", path / "data.pt"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _coerce_to_data_list(obj):
    if isinstance(obj, Data):
        return [obj]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], dict):
        data, slices = obj
        length = _infer_slices_length(slices)
        return [
            separate(
                cls=data.__class__,
                batch=data,
                idx=index,
                slice_dict=slices,
                decrement=False,
            )
            for index in range(length)
        ]
    if isinstance(obj, dict):
        for key in ("data_list", "dataset", "graphs"):
            if key in obj:
                return _coerce_to_data_list(obj[key])
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        return [obj[index] for index in range(len(obj))]
    raise TypeError("Unsupported dataset object type: {}".format(type(obj)))


def _infer_slices_length(slices):
    for value in slices.values():
        if torch.is_tensor(value):
            return max(0, int(value.numel()) - 1)
        if isinstance(value, dict):
            nested = _infer_slices_length(value)
            if nested > 0:
                return nested
    raise ValueError("Could not infer dataset length from slices.")


if __name__ == "__main__":
    main()
