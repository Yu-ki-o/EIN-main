#!/usr/bin/env python3
"""Seed-4 sweep for Pheme BiGCN UncertaintySemanticChange."""

from sweep_uncertainty_semantic_change_common import REPO_ROOT, run_cli


DEFAULTS = {
    "config": REPO_ROOT / "configs/EIN/Pheme_BiGCN_UncertaintySemanticChange.yaml",
    "config_out_dir": REPO_ROOT / "configs/sweeps/pheme_bigcn_uncertainty_semantic_change",
    "output_dir": REPO_ROOT / "experiments/EIN/Pheme/sweep_pheme_bigcn_uncertainty_semantic_change",
    "result_group": "sweep_pheme_bigcn_uncertainty_semantic_change",
    "sweep_name_prefix": "sweep_pheme_bigcn_uncertainty_semantic_change",
    "expected_dataset": "Pheme",
    "expected_base_model": "BiGCN_UncertaintySemanticChange",
    "extra_preset_grids": {
        "capacity": {
            "classification_fusion_hidden_dim": [64, 128, 256],
            "dropout": [0.05, 0.1, 0.2],
        },
        "optimizer": {
            "lr": [0.0003, 0.0005, 0.001],
            "weight_decay": [0.00005, 0.0001, 0.0002],
        },
    },
}


if __name__ == "__main__":
    run_cli(DEFAULTS)
