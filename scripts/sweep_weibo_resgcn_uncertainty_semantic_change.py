#!/usr/bin/env python3
"""Seed-0 sweep for Weibo ResGCN change+semantic-tree model."""

from sweep_uncertainty_semantic_change_common import REPO_ROOT, run_cli


DEFAULTS = {
    "config": REPO_ROOT / "configs/EIN/Weibo_ResGCN_UncertaintySemanticChange_word2vec.yaml",
    "config_out_dir": REPO_ROOT / "configs/sweeps/weibo_resgcn_change_semantic_tree",
    "output_dir": REPO_ROOT / "experiments/EIN/Weibo/sweep_weibo_resgcn_change_semantic_tree",
    "result_group": "sweep_weibo_resgcn_change_semantic_tree",
    "sweep_name_prefix": "sweep_weibo_resgcn_change_semantic_tree",
    "expected_dataset": "Weibo",
    "expected_base_model": "ResGCN_UncertaintySemanticChange",
    "fixed_fusion_mode": "change_semantic_tree",
    "default_seed": 0,
    "default_preset": "semantic_tree_core",
    "extra_preset_grids": {
        "semantic_tree_core": {
            "semantic_tree_transformer_layers": [1, 2],
            "semantic_tree_transformer_dropout": [0.2, 0.3, 0.4],
            "semantic_tree_transformer_max_depth": [40, 72],
            "lambda_view_mi_aux": [0.0, 0.01, 0.02],
        },
        "semantic_tree_capacity": {
            "semantic_tree_num_topics": [4, 8],
            "semantic_tree_transformer_ffn_dim": [256, 512],
            "semantic_tree_depth_dim": [64, 128],
            "semantic_tree_transformer_pool": ["mean", "root"],
        },
        "semantic_tree_route": {
            "stance_route_temperature": [0.3, 0.5, 0.8],
            "stance_route_hard": [True, False],
            "use_uncertainty_sampling": [False, True],
        },
        "capacity": {
            "classification_fusion_hidden_dim": [128, 256, 384],
            "dropout": [0.2, 0.3, 0.4],
        },
        "optimizer": {
            "lr": [0.0003, 0.0005, 0.001],
            "weight_decay": [0.000005, 0.00001, 0.00002],
        },
    },
}


if __name__ == "__main__":
    run_cli(DEFAULTS)
