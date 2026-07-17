#!/usr/bin/env python3
"""Seed-0 key-parameter sweep for Weibo ResGCN UncertaintySemanticChange.

Checkpoint selection is fixed to val_loss in this entrypoint.
"""

from sweep_uncertainty_semantic_change_common import REPO_ROOT, run_cli


DEFAULTS = {
    "config": REPO_ROOT
    / "configs/EIN/Weibo_ResGCN_UncertaintySemanticChange_word2vec.yaml",
    "config_out_dir": REPO_ROOT
    / "configs/sweeps/weibo_resgcn_uncertainty_semantic_change_keyparams_seed0",
    "output_dir": REPO_ROOT
    / "experiments/EIN/Weibo/sweep_weibo_resgcn_uncertainty_semantic_change_keyparams_seed0",
    "result_group": "sweep_weibo_resgcn_uncertainty_semantic_change_keyparams_seed0",
    "sweep_name_prefix": "sweep_weibo_resgcn_uncertainty_semantic_change_keyparams",
    "expected_dataset": "Weibo",
    "expected_base_model": "ResGCN_UncertaintySemanticChange",
    "fixed_fusion_mode": "change_semantic_tree",
    "fixed_selection_metric": "val_loss",
    "default_seed": 0,
    "default_preset": "core",
    "extra_preset_grids": {
        "core": {
            "classification_class_weights": [
                [1.0, 1.0],
                [1.0, 1.1],
                [1.0, 1.25],
                [1.0, 1.5],
            ],
            "ds_unknown_prior": [1.0, 2.0, 4.0],
            "lambda_view_mi_aux": [0.0, 0.005, 0.01],
        },
        "loss_weights": {
            "classification_class_weights": [
                [1.0, 1.0],
                [1.0, 1.1],
                [1.0, 1.25],
                [1.0, 1.5],
            ],
        },
        "ds_routing": {
            "ds_unknown_prior": [0.5, 1.0, 2.0, 4.0],
            "lambda_ds_unknown_edge_aux": [0.0, 0.001, 0.005, 0.01],
        },
        "ds_ablation": {
            "use_ds_mass_routing": [False, True],
        },
        "edge_aux": {
            "lambda_edge_relation_aux": [0.05, 0.1, 0.2],
            "lambda_edge_relation_warmup": [0.2, 0.5, 1.0],
            "lambda_view_mi_aux": [0.0, 0.005, 0.01, 0.02],
        },
        "semantic_tree_core": {
            "semantic_tree_transformer_layers": [1, 2, 3],
            "semantic_tree_transformer_dropout": [0.2, 0.3, 0.4],
            "semantic_tree_transformer_max_depth": [40, 72],
        },
        "semantic_tree_capacity": {
            "semantic_tree_num_topics": [4, 8],
            "semantic_tree_transformer_ffn_dim": [256, 512],
            "semantic_tree_depth_dim": [64, 128],
            "semantic_tree_transformer_pool": ["mean", "root"],
        },
        "global_ds": {
            "use_global_ds_fusion": [True],
            "global_ds_unknown_prior": [0.5, 1.0, 2.0],
            "global_ds_fusion_rule": ["dempster", "yager"],
        },
        "capacity": {
            "classification_fusion_hidden_dim": [128, 256, 384],
            "dropout": [0.2, 0.3, 0.4],
        },
        "optimizer": {
            "lr": [0.0003, 0.0005, 0.001],
            "weight_decay": [0.000005, 0.00001, 0.00002],
        },
        "early_stop": {
            "patience": [10, 15, 20, 30],
        },
    },
}


if __name__ == "__main__":
    run_cli(DEFAULTS)
