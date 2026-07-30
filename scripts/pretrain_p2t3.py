#!/usr/bin/env python3
"""EIN-native P2T3 mutual-information pre-training."""

import argparse
import json
import math
import os
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.P2T3 import P2T3  # noqa: E402
from utils.p2t3_pretrain import (  # noqa: E402
    P2T3PretrainChunkDataset,
    P2T3TextCorpus,
    prepare_p2t3_pretrain_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pre-train EIN P2T3 on UWeibo/UTwitter JSON files and save the "
            "checkpoint configured by p2t3_pretrained_path."
        )
    )
    parser.add_argument(
        "--config_filename",
        default="configs/EIN/DRWeibo_P2T3_word2vec.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--rebuild_word2vec", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolve_device(requested):
    requested = str(requested).strip().lower()
    if requested.isdigit():
        requested = "cuda:{}".format(requested)
    elif requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment cannot access "
                "a CUDA GPU. Install/use a CUDA-enabled PyTorch environment."
            )
        device = torch.device(requested)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                "Requested {}, but only {} GPU(s) are visible.".format(
                    requested,
                    torch.cuda.device_count(),
                )
            )
        torch.cuda.set_device(device)
        return device
    if requested != "cpu":
        raise ValueError("Unsupported device: {}".format(requested))
    return torch.device("cpu")


def initialize_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_word2vec(args, raw_dir, rebuild=False, limit_files=None):
    if args.word_embedding != "word2vec":
        raise ValueError(
            "The native P2T3 pre-training command currently requires "
            "word_embedding: word2vec."
        )

    from gensim.models import Word2Vec
    from utils.word2vec import Embedding

    configured_path = getattr(args, "word2vec_model_path", None)
    if configured_path is None or not str(configured_path).strip():
        raise ValueError(
            "Set word2vec_model_path in the P2T3 YAML so pre-training and "
            "fine-tuning share exactly the same encoder."
        )
    model_path = resolve_path(configured_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if rebuild or not model_path.is_file():
        source_datasets = getattr(
            args,
            "p2t3_word2vec_source_datasets",
            [args.dataset],
        )
        if isinstance(source_datasets, str):
            source_datasets = [
                item.strip()
                for item in source_datasets.split(",")
                if item.strip()
            ]
        corpus_dirs = [raw_dir]
        for dataset in source_datasets:
            source_dir = REPO_ROOT / "dataset" / dataset / "source"
            if not source_dir.is_dir():
                raise FileNotFoundError(
                    "Word2Vec source directory does not exist: {}".format(
                        source_dir
                    )
                )
            corpus_dirs.append(source_dir)

        limit_map = {}
        if limit_files is not None:
            limit_map[str(raw_dir)] = limit_files
        corpus = P2T3TextCorpus(
            corpus_dirs,
            args.language,
            args.tokenize_mode,
            limit_by_directory=limit_map,
        )
        print(
            "Training shared P2T3 Word2Vec -> {}".format(model_path),
            flush=True,
        )
        word2vec = Word2Vec(
            sentences=corpus,
            vector_size=int(args.vector_size),
            window=int(getattr(args, "p2t3_word2vec_window", 5)),
            min_count=int(getattr(args, "p2t3_word2vec_min_count", 5)),
            workers=max(1, int(getattr(args, "p2t3_word2vec_workers", 8))),
            epochs=max(1, int(getattr(args, "p2t3_word2vec_epochs", 30))),
            sg=1,
            seed=int(args.seed),
        )
        word2vec.save(str(model_path))

    encoder = Embedding(
        str(model_path),
        args.language,
        args.tokenize_mode,
    )
    if int(args.in_feats) != int(encoder.embedding_dim):
        raise ValueError(
            "Config in_feats={} does not match Word2Vec dimension {}.".format(
                args.in_feats,
                encoder.embedding_dim,
            )
        )
    return encoder, model_path


def linear_warmup_decay(optimizer, warmup_steps, total_steps):
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def scale(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = total_steps - step
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(remaining) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_resume_checkpoint(path, model, optimizer, scheduler, scaler, device):
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "Cannot resume: checkpoint is not an EIN P2T3 pre-training file."
        )
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint.get("epoch", 0)), int(
        checkpoint.get("optimizer_step", 0)
    )


def main():
    cli = parse_args()
    config_path = resolve_path(cli.config_filename)
    with config_path.open("r", encoding="utf-8") as file_obj:
        config = yaml.load(file_obj, Loader=yaml.FullLoader)
    args = Namespace(**config)
    args.config_filename = str(config_path)
    if cli.device is not None:
        args.device = cli.device
    if cli.epochs is not None:
        args.p2t3_pretrain_epochs = cli.epochs

    initialize_seed(int(args.seed))
    device = resolve_device(args.device)

    pretrain_dataset = str(
        getattr(args, "p2t3_pretrain_dataset", "UWeibo")
    )
    configured_raw_dir = getattr(
        args,
        "p2t3_pretrain_raw_dir",
        "dataset/{}/dataset/raw".format(pretrain_dataset),
    )
    raw_dir = resolve_path(cli.raw_dir or configured_raw_dir)
    checkpoint_setting = (
        cli.checkpoint_path
        or getattr(args, "p2t3_pretrained_path", None)
    )
    if checkpoint_setting is None or not str(checkpoint_setting).strip():
        raise ValueError(
            "Set p2t3_pretrained_path in the YAML or pass --checkpoint_path."
        )
    checkpoint_path = resolve_path(checkpoint_setting)

    print("P2T3 pre-training dataset: {}".format(raw_dir), flush=True)
    print("P2T3 runtime device: {}".format(device), flush=True)
    encoder, word2vec_path = build_word2vec(
        args,
        raw_dir,
        rebuild=cli.rebuild_word2vec,
        limit_files=cli.limit_files,
    )

    processed_root = resolve_path(
        getattr(
            args,
            "p2t3_pretrain_processed_dir",
            "dataset/{}/dataset/processed".format(pretrain_dataset),
        )
    )
    cache_dir = prepare_p2t3_pretrain_cache(
        raw_dir,
        encoder,
        args,
        processed_root=processed_root,
        chunk_size=int(getattr(args, "p2t3_pretrain_cache_chunk_size", 512)),
        force_rebuild=cli.rebuild_cache,
        limit_files=cli.limit_files,
    )
    dataset = P2T3PretrainChunkDataset(
        cache_dir,
        seed=int(args.seed),
        shuffle=True,
    )
    batch_size = max(2, int(getattr(args, "p2t3_pretrain_batch_size", 8)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=max(0, int(getattr(args, "p2t3_pretrain_num_workers", 0))),
        pin_memory=device.type == "cuda",
    )
    if len(dataset) < batch_size:
        raise ValueError(
            "Pre-training needs at least {} samples, found {}.".format(
                batch_size,
                len(dataset),
            )
        )

    model = P2T3(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(args, "p2t3_pretrain_lr", 5e-5)),
        weight_decay=float(
            getattr(args, "p2t3_pretrain_weight_decay", 1e-3)
        ),
    )

    effective_batch_size = max(
        batch_size,
        int(getattr(args, "p2t3_pretrain_effective_batch_size", 32)),
    )
    if effective_batch_size % batch_size != 0:
        raise ValueError(
            "p2t3_pretrain_effective_batch_size must be divisible by "
            "p2t3_pretrain_batch_size."
        )
    accumulation_steps = effective_batch_size // batch_size
    epochs = max(1, int(getattr(args, "p2t3_pretrain_epochs", 30)))
    batches_per_epoch = len(dataset) // batch_size
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / accumulation_steps
    )
    total_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(
        float(getattr(args, "p2t3_pretrain_warmup_ratio", 0.1))
        * total_steps
    )
    scheduler = linear_warmup_decay(optimizer, warmup_steps, total_steps)

    use_amp = (
        device.type == "cuda"
        and bool(getattr(args, "p2t3_pretrain_amp", True))
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 0
    optimizer_step = 0
    resume = (
        bool(getattr(args, "p2t3_pretrain_resume", True))
        and not cli.no_resume
        and checkpoint_path.is_file()
    )
    if resume:
        start_epoch, optimizer_step = load_resume_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        print(
            "Resuming {} from completed epoch {}.".format(
                checkpoint_path,
                start_epoch,
            ),
            flush=True,
        )
    if start_epoch >= epochs:
        print(
            "Checkpoint already contains {} completed epoch(s); nothing to do.".format(
                start_epoch
            ),
            flush=True,
        )
        return

    gradient_clip = float(
        getattr(args, "p2t3_pretrain_gradient_clip", 1.0)
    )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        contributing_batches = 0
        pending_gradients = 0

        for batch_index, data in enumerate(loader, start=1):
            data = data.to(
                device,
                non_blocking=device.type == "cuda",
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                loss = model.pretraining_loss(data)

            if not loss.requires_grad:
                continue
            scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            pending_gradients += 1
            running_loss += float(loss.detach())
            contributing_batches += 1

            should_step = pending_gradients >= accumulation_steps
            is_last_batch = batch_index >= batches_per_epoch
            if should_step or (is_last_batch and pending_gradients > 0):
                scaler.unscale_(optimizer)
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
                pending_gradients = 0

        if pending_gradients > 0:
            scaler.unscale_(optimizer)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_step += 1

        mean_loss = running_loss / max(1, contributing_batches)
        payload = {
            "format": "ein-p2t3-pretrain-v1",
            "epoch": epoch + 1,
            "optimizer_step": optimizer_step,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": vars(args),
            "raw_dir": str(raw_dir),
            "cache_dir": str(cache_dir),
            "word2vec_model_path": str(word2vec_path),
            "mean_loss": mean_loss,
        }
        atomic_torch_save(payload, checkpoint_path)
        print(
            "Epoch {}/{} | MI loss {:.6f} | optimizer steps {} | saved {}".format(
                epoch + 1,
                epochs,
                mean_loss,
                optimizer_step,
                checkpoint_path,
            ),
            flush=True,
        )

    metadata_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".json"
    )
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "dataset": pretrain_dataset,
                "raw_dir": str(raw_dir),
                "word2vec_model_path": str(word2vec_path),
                "epochs": epochs,
            },
            file_obj,
            indent=2,
            ensure_ascii=False,
        )
    print("P2T3 pre-training complete: {}".format(checkpoint_path), flush=True)


if __name__ == "__main__":
    main()
