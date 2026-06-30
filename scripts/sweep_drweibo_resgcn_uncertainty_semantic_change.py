#!/usr/bin/env python3
"""Seed-4 sweep for DRWeibo ResGCN UncertaintySemanticChange."""

from sweep_uncertainty_semantic_change_common import REPO_ROOT, run_cli


DEFAULTS = {
    "config": REPO_ROOT / "configs/EIN/DRWeibo_ResGCN_UncertaintySemanticChange_word2vec.yaml",
    "config_out_dir": REPO_ROOT / "configs/sweeps/drweibo_resgcn_uncertainty_semantic_change",
    "output_dir": REPO_ROOT / "experiments/EIN/DRWeibo/sweep_drweibo_resgcn_uncertainty_semantic_change",
    "result_group": "sweep_drweibo_resgcn_uncertainty_semantic_change",
    "sweep_name_prefix": "sweep_drweibo_resgcn_uncertainty_semantic_change",
    "expected_dataset": "DRWeibo",
    "expected_base_model": "ResGCN_UncertaintySemanticChange",
    "extra_preset_grids": {
        "capacity": {
            "classification_fusion_hidden_dim": [128, 256, 384],
            "dropout": [0.2, 0.3, 0.4],
        },
        "optimizer": {
            "lr": [0.0003, 0.0005, 0.001],
            "weight_decay": [0.00005, 0.0001, 0.0002],
        },
    },
}


if __name__ == "__main__":
    run_cli(DEFAULTS)
