#!/usr/bin/env python3
"""Seed-0 key-parameter sweep for Pheme BiGCN USC with MLP change encoder.

This entrypoint reuses the existing Pheme key-parameter grid, but it always
forces semantic_change_encoder to mlp when generating and running trial configs.
Checkpoint selection is fixed to val_loss.
"""

from copy import deepcopy

from sweep_pheme_bigcn_uncertainty_semantic_change_keyparams_seed0 import (
    DEFAULTS as BASE_DEFAULTS,
)
from sweep_uncertainty_semantic_change_common import REPO_ROOT, run_cli


DEFAULTS = deepcopy(BASE_DEFAULTS)
DEFAULTS.update(
    {
        "config": REPO_ROOT / "configs/EIN/Pheme_BiGCN_UncertaintySemanticChange.yaml",
        "config_out_dir": REPO_ROOT
        / "configs/sweeps/pheme_bigcn_uncertainty_semantic_change_keyparams_mlp_seed0",
        "output_dir": REPO_ROOT
        / "experiments/EIN/Pheme/sweep_pheme_bigcn_uncertainty_semantic_change_keyparams_mlp_seed0",
        "result_group": "sweep_pheme_bigcn_uncertainty_semantic_change_keyparams_mlp_seed0",
        "sweep_name_prefix": "sweep_pheme_bigcn_uncertainty_semantic_change_keyparams_mlp",
        "fixed_fusion_mode": "change_semantic_tree",
        "fixed_selection_metric": "val_loss",
        "default_seed": 0,
        "default_preset": "core",
        "fixed_config_overrides": {
            "semantic_change_encoder": "mlp",
        },
    }
)


if __name__ == "__main__":
    run_cli(DEFAULTS)
