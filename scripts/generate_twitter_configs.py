"""Generate Twitter EIN config files from existing English/Pheme templates."""

from __future__ import annotations

from pathlib import Path

import yaml


MAX_HOP = 147


CONFIG_SPECS = [
    (
        "configs/EIN/Pheme.yaml",
        "configs/EIN/Twitter_ResGCN_word2vec.yaml",
        {"base_model": "ResGCN"},
    ),
    (
        "configs/EIN/DRWeibo_BiGCN_word2vec.yaml",
        "configs/EIN/Twitter_BiGCN_word2vec.yaml",
        {"base_model": "BiGCN"},
    ),
    (
        "configs/EIN/Pheme_BiGCN_Uncertainty.yaml",
        "configs/EIN/Twitter_BiGCN_Uncertainty_word2vec.yaml",
        {"base_model": "BiGCN_Uncertainty"},
    ),
    (
        "configs/EIN/DRWeibo_ResGCN_Uncertainty_word2vec.yaml",
        "configs/EIN/Twitter_ResGCN_Uncertainty_word2vec.yaml",
        {"base_model": "ResGCN_Uncertainty"},
    ),
    (
        "configs/EIN/Pheme_BiGCN_StateAuxSameDiff.yaml",
        "configs/EIN/Twitter_BiGCN_StateAuxSameDiff_word2vec.yaml",
        {"base_model": "BiGCN_StateAuxSameDiff"},
    ),
    (
        "configs/EIN/Pheme_ResGCN_StateAuxSameDiff.yaml",
        "configs/EIN/Twitter_ResGCN_StateAuxSameDiff_word2vec.yaml",
        {"base_model": "ResGCN_StateAuxSameDiff"},
    ),
    (
        "configs/EIN/Pheme_BiGCN_RevisionAwareSemanticChange.yaml",
        "configs/EIN/Twitter_BiGCN_RevisionAwareSemanticChange_word2vec.yaml",
        {"base_model": "BiGCN_RevisionAwareSemanticChange"},
    ),
    (
        "configs/EIN/Pheme_BiGCN_RevisionAwareSemanticChange.yaml",
        "configs/EIN/Twitter_ResGCN_RevisionAwareSemanticChange_word2vec.yaml",
        {"base_model": "ResGCN_RevisionAwareSemanticChange"},
    ),
    (
        "configs/EIN/Pheme_BiGCN_UncertaintySemanticChange.yaml",
        "configs/EIN/Twitter_BiGCN_UncertaintySemanticChange_word2vec.yaml",
        {"base_model": "BiGCN_UncertaintySemanticChange"},
    ),
    (
        "configs/EIN/Pheme_BiGCN_UncertaintySemanticChange.yaml",
        "configs/EIN/Twitter_ResGCN_UncertaintySemanticChange_word2vec.yaml",
        {"base_model": "ResGCN_UncertaintySemanticChange"},
    ),
    (
        "configs/EIN/Pheme_GCN_UncertaintySemanticChange.yaml",
        "configs/EIN/Twitter_GCN_UncertaintySemanticChange_word2vec.yaml",
        {"base_model": "GCN_UncertaintySemanticChange"},
    ),
    (
        "configs/EIN/Pheme_GIN_UncertaintySemanticChange.yaml",
        "configs/EIN/Twitter_GIN_UncertaintySemanticChange_word2vec.yaml",
        {"base_model": "GIN_UncertaintySemanticChange"},
    ),
    (
        "configs/EIN/Pheme_KAGNN.yaml",
        "configs/EIN/Twitter_KAGNN_word2vec.yaml",
        {"base_model": "KAGNN"},
    ),
    (
        "configs/EIN/Pheme_KAGNN_UncertaintySemanticChange.yaml",
        "configs/EIN/Twitter_KAGNN_UncertaintySemanticChange_word2vec.yaml",
        {"base_model": "KAGNN_UncertaintySemanticChange"},
    ),
    (
        "configs/EIN/Pheme_RAGCL_BiGCN_word2vec.yaml",
        "configs/EIN/Twitter_RAGCL_BiGCN_word2vec.yaml",
        {"base_model": "RAGCL_BiGCN", "use_unsup_loss": True},
    ),
    (
        "configs/EIN/Pheme_RAGCL_ResGCN_word2vec.yaml",
        "configs/EIN/Twitter_RAGCL_ResGCN_word2vec.yaml",
        {"base_model": "RAGCL_ResGCN", "use_unsup_loss": True},
    ),
    (
        "configs/EIN/Pheme_NEGT_word2vec.yaml",
        "configs/EIN/Twitter_NEGT_word2vec.yaml",
        {"base_model": "NEGT"},
    ),
    (
        "configs/EIN/Pheme_SEEGraphMAE.yaml",
        "configs/EIN/Twitter_SEEGraphMAE_word2vec.yaml",
        {"base_model": "SEEGraphMAE"},
    ),
    (
        "configs/EIN/Pheme_TCSR_word2vec.yaml",
        "configs/EIN/Twitter_TCSR_word2vec.yaml",
        {"base_model": "TCSR", "model_name": "TCSR"},
    ),
    (
        "configs/EIN/Pheme_LIRS.yaml",
        "configs/EIN/Twitter_LIRS.yaml",
        {
            "base_model": "LIRS",
            "word_embedding": "multilingual-e5-base",
            "in_feats": 768,
        },
    ),
]


def compact_name(output_path: Path, base_model: str, embedding: str) -> str:
    stem = output_path.stem
    if stem.startswith("Twitter_"):
        stem = stem[len("Twitter_") :]
    return "Twitter_{}_{}".format(base_model, embedding).replace("/", "_")


def normalize_common(config: dict, output_path: Path, overrides: dict) -> dict:
    config = dict(config)
    config.update(overrides)

    embedding = config.get("word_embedding", "word2vec")
    config["dataset"] = "Twitter"
    config["language"] = "en"
    config["experiment_mode"] = "id"
    config["ood_source_datasets"] = []
    config["ood_val_domain"] = "source"
    config["selection_metric"] = config.get("selection_metric", "val_loss")
    config["result_name"] = compact_name(
        output_path,
        str(config.get("base_model", "Model")),
        str(embedding),
    )

    if embedding == "word2vec":
        config["tokenize_mode"] = "naive"
        config["in_feats"] = 200
        config["vector_size"] = 200
    else:
        config["tokenize_mode"] = "naive"
        config["e5_local_files_only"] = False

    config["max_hop"] = MAX_HOP
    if "vertical_path_attention_max_distance" in config:
        config["vertical_path_attention_max_distance"] = MAX_HOP
    if "semantic_tree_transformer_max_depth" in config:
        config["semantic_tree_transformer_max_depth"] = MAX_HOP

    if config.get("hidden_dim") == 128 and embedding == "word2vec":
        config["hidden_dim"] = 64
    for key in (
        "relation_hidden_dim",
        "semantic_change_hidden_dim",
        "uncertainty_trend_hidden_dim",
        "global_ds_hidden_dim",
        "semantic_tree_depth_dim",
    ):
        if config.get(key) == 128:
            config[key] = 64
    if config.get("classification_fusion_hidden_dim") == 256:
        config["classification_fusion_hidden_dim"] = 128
    if config.get("semantic_tree_transformer_ffn_dim") == 256:
        config["semantic_tree_transformer_ffn_dim"] = 128

    if "classification_class_weights" in config:
        config["classification_class_weights"] = [1.0, 1.0]
    if config.get("base_model") == "TCSR":
        config["checkpoint_dir"] = "checkpoints/tcsr/Twitter"

    return config


def main() -> None:
    written = []
    for template, output, overrides in CONFIG_SPECS:
        template_path = Path(template)
        output_path = Path(output)
        if not template_path.exists():
            raise FileNotFoundError(template_path)
        config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        config = normalize_common(config, output_path, overrides)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append(output_path)

    for path in written:
        print(path)
    print("Wrote {} Twitter config files.".format(len(written)))


if __name__ == "__main__":
    main()
