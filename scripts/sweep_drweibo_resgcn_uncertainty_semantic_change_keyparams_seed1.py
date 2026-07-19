#!/usr/bin/env python3
"""Seed-1 compact key-parameter sweep for DRWeibo ResGCN USC.

This entrypoint keeps the DRWeibo base config intact except for the requested
fixed overrides and the compact key-parameter grid.
"""

from sweep_uncertainty_semantic_change_common import REPO_ROOT, run_cli


DEFAULTS = {
    "config": REPO_ROOT
    / "configs/EIN/DRWeibo_ResGCN_UncertaintySemanticChange_word2vec.yaml",
    "config_out_dir": REPO_ROOT
    / "configs/sweeps/drweibo_resgcn_uncertainty_semantic_change_keyparams_seed1",
    "output_dir": REPO_ROOT
    / "experiments/EIN/DRWeibo/sweep_drweibo_resgcn_uncertainty_semantic_change_keyparams_seed1",
    "result_group": "sweep_drweibo_resgcn_uncertainty_semantic_change_keyparams_seed1",
    "sweep_name_prefix": "sweep_drweibo_resgcn_uncertainty_semantic_change_keyparams",
    "expected_dataset": "DRWeibo",
    "expected_base_model": "ResGCN_UncertaintySemanticChange",
    "fixed_fusion_mode": "change_semantic_tree",
    "fixed_selection_metric": "val_loss",
    "default_seed": 1,
    "default_preset": "core",
    "fixed_config_overrides": {
        "edge_relation_distribution": "softmax",
        "use_uncertainty_sampling": False,
        "semantic_change_encoder": "mlp",
        "use_semantic_tree_transformer": True,
        "use_semantic_tree_change_uncertainty_bias": True,
    },
    "extra_preset_grids": {
        "core": {
            "lambda_semantic_tree_change_mi_aux": [0.0, 0.01],
            "semantic_tree_input_mode": ["support_deny", "support_deny_original"],
            "semantic_tree_uncertainty_bias_scale": [0.5, 1.0],
            "classification_class_weights": [[1.0, 1.0], [1.0, 1.25]],
            "lambda_edge_relation_aux": [0.0, 0.05],
            "lambda_edge_relation_warmup": [0.0, 0.5],
            "lambda_view_mi_aux": [0.0, 0.01],
        },
    },
}


if __name__ == "__main__":
    run_cli(DEFAULTS)
