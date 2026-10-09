#!/usr/bin/env python3
"""Tune Weibo DIGNN on CUDA using a single minimum-val_loss checkpoint."""

from tune_dignn_pheme import main


if __name__ == '__main__':
    main(default_config='configs/EIN/Weibo_DIGNN_word2vec.yaml',
         strict_val_loss=True, default_device='cuda:0', require_cuda=True)
