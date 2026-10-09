#!/usr/bin/env python3
"""Tune DRWeibo DIGNN with a single minimum-validation-loss checkpoint."""

from tune_dignn_pheme import main


if __name__ == '__main__':
    main(default_config='configs/EIN/DRWeibo_DIGNN_word2vec.yaml', strict_val_loss=True)
