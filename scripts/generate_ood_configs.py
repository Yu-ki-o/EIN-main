"""Convert an existing model YAML to a source-only OOD experiment config.

Run from the repository root. Manifest paths are resolved at training time using
the same working directory as main.py. This command never loads embeddings.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

import yaml


DATASETS = ("DRWeibo", "Weibo", "Pheme")
LANGUAGES = {"DRWeibo": "ch", "Weibo": "ch", "Pheme": "en"}
# These fields can silently reuse an ID split, target-trained model, or output.
DROP_FIELDS = (
    "dataset_cache_name", "legacy_dataset_cache_name", "reuse_dataset_cache",
    "rebuild_dataset_cache", "checkpoint_path", "checkpoint_dir",
    "p2t3_pretrained_path", "kpg_checkpoint", "kpg_test_only", "eval_only", "early_test_root",
    "result_dir", "result_path", "result_group", "output_dir", "summary_name", "split_manifest",
    "split_manifest_path", "word2vec_model_path", "split", "k",
)


def read_manifest(pattern: str) -> dict | None:
    """Read the seed-zero manifest when present; allow generating E5 templates first."""
    try:
        path = Path(pattern.format(seed=0))
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError("--manifest supports only the {seed} placeholder") from exc
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("target_dataset") not in DATASETS:
        raise ValueError(f"Invalid OOD manifest target_dataset: {path}")
    return manifest


def make_ood_config(
    base: dict,
    manifest_pattern: str,
    *,
    dataset: str | None = None,
    embedding: str = "e5",
    result_name: str | None = None,
    manifest: dict | None = None,
) -> dict:
    if not isinstance(base, dict) or not base.get("base_model"):
        raise ValueError("The base YAML must contain a base_model")
    manifest_target = manifest.get("target_dataset") if manifest else None
    if dataset and manifest_target and dataset != manifest_target:
        raise ValueError("--dataset disagrees with the manifest target_dataset")
    dataset = dataset or manifest_target
    if dataset not in DATASETS:
        raise ValueError("Prepare the manifest first, or pass --dataset DRWeibo/Weibo/Pheme")
    if embedding not in ("e5", "word2vec"):
        raise ValueError(f"Unsupported embedding: {embedding}")
    sources = list(manifest.get("source_datasets", [])) if manifest else []
    if embedding == "word2vec":
        if not sources:
            raise ValueError("Prepare the OOD manifest before generating a Word2Vec config")
        if any(LANGUAGES.get(source) != LANGUAGES[dataset] for source in sources):
            raise ValueError("Cross-language OOD requires the shared multilingual E5 encoder")

    config = copy.deepcopy(base)
    for key in DROP_FIELDS:
        config.pop(key, None)
    config.update(
        dataset=dataset,
        language=LANGUAGES[dataset],
        experiment_mode="ood",
        ood_manifest=manifest_pattern,
        ood_source_datasets=sources,
        ood_val_domain="source",
        selection_metric="val_loss",
        tokenize_mode="jieba" if LANGUAGES[dataset] == "ch" else "nltk",
        max_hop=max(72, int(base.get("max_hop", 72))),
    )
    # An ID template may contain target class weights or target-time adaptation.
    if "classification_class_weights" in config:
        config["classification_class_weights"] = [1.0] * int(config.get("num_classes", 2))
    if "see_ttt_enabled" in config or config.get("base_model") == "SEEGraphMAE":
        config["see_ttt_enabled"] = False
    if "p2t3_pretrain_resume" in config:
        config["p2t3_pretrain_resume"] = False
    if embedding == "e5":
        config.update(
            word_embedding="multilingual-e5-base",
            e5_model_name="intfloat/multilingual-e5-base",
            e5_local_files_only=True,
            in_feats=768,
        )
        config.setdefault("e5_max_length", 128)
        config.setdefault("e5_batch_size", 64)
    else:
        vector_size = int(config.get("vector_size", 200))
        config.update(word_embedding="word2vec", in_feats=vector_size, vector_size=vector_size)

    if not result_name:
        identity = {key: value for key, value in config.items() if key != "result_name"}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:8]
        protocol = Path(manifest_pattern).parent.name
        result_name = f"ood_{protocol}_{config['base_model']}_{embedding}_{digest}"
    result_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", result_name).strip("._-")
    if not result_name:
        raise ValueError("result_name must contain a letter or digit")
    config["result_name"] = result_name
    if config.get("base_model") == "TCSR" or "checkpoint_dir" in base:
        config["checkpoint_dir"] = f"checkpoints/ood/{dataset}/{result_name}"
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, help="JSON path, optionally containing {seed}")
    parser.add_argument("--dataset", choices=DATASETS, help="Target dataset; inferred if manifest exists")
    parser.add_argument("--embedding", choices=("e5", "word2vec"), default="e5")
    parser.add_argument("--result-name")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Replace an existing output YAML")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.force:
        parser.error(f"Output exists: {args.output}; use --force to replace it")
    try:
        with args.base_config.open(encoding="utf-8") as stream:
            base = yaml.safe_load(stream)
        config = make_ood_config(
            base, args.manifest, dataset=args.dataset, embedding=args.embedding,
            result_name=args.result_name, manifest=read_manifest(args.manifest),
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        stream.write("# Source-only OOD; prepare ood_manifest before running main.py.\n")
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    print(f"Wrote {args.output} ({config['base_model']}, {config['word_embedding']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
