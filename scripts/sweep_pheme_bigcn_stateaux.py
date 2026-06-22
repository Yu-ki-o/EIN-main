#!/usr/bin/env python3
"""One-seed hyperparameter sweep for Pheme BiGCN StateAuxSameDiff.

This wrapper reuses the DRWeibo/ResGCN sweep implementation while changing only
the dataset, backbone, config, and output locations. The sweep presets still
touch only StateAuxSameDiff parameters, leaving lr, weight_decay, dropout, and
other base hyperparameters as defined in the source config.
"""

from pathlib import Path

import sweep_drweibo_resgcn_stateaux as sweep


REPO_ROOT = Path(__file__).resolve().parents[1]

sweep.DEFAULT_CONFIG = REPO_ROOT / "configs/EIN/Pheme_BiGCN_StateAuxSameDiff.yaml"
sweep.DEFAULT_CONFIG_OUT_DIR = REPO_ROOT / "configs/sweeps/pheme_bigcn_stateaux"
sweep.DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/EIN/Pheme/sweep_pheme_bigcn_stateaux"
sweep.DEFAULT_SWEEP_NAME_PREFIX = "sweep_pheme_bigcn_stateaux"
sweep.EXPECTED_DATASET = "Pheme"
sweep.EXPECTED_BASE_MODEL = "BiGCN_StateAuxSameDiff"


if __name__ == "__main__":
    sweep.main()
