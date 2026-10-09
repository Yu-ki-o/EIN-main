"""Unique-epoch weighting and native trainer integration for checkpoint averages."""

import ast
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from utils.checkpoint_average import average_checkpoints, AVERAGE_SELECTORS, DEFAULT_TARGETS


ROOT = Path(__file__).resolve().parents[1]


def trainer_class():
    tree = ast.parse((ROOT / 'trainer/EIN_trainer.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EINTrainer')
    names = ('_average_selectors', '_record_average_candidates', '_load_averaged_candidates')
    nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(torch=torch, np=np, os=os, json=json, average_checkpoints=average_checkpoints,
                     AVERAGE_SELECTORS=AVERAGE_SELECTORS, DEFAULT_TARGETS=DEFAULT_TARGETS)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'trainer/EIN_trainer.py', 'exec'), namespace)
    return type('NativeAverageTrainer', (), {name: namespace[name] for name in names})


class CheckpointAverageTest(unittest.TestCase):
    def test_native_trainer_gives_each_unique_epoch_equal_weight(self):
        with TemporaryDirectory() as directory:
            trainer = trainer_class()()
            trainer.args = SimpleNamespace(log_dir=directory,
                                           checkpoint_average_metrics=list(AVERAGE_SELECTORS))
            trainer.device = torch.device('cpu')
            trainer.model = torch.nn.Linear(2, 2)
            trainer.logger = SimpleNamespace(info=lambda *args: None)
            for parameter in trainer.model.parameters():
                parameter.data.fill_(1.)
            trainer._record_average_candidates(
                dict(val_loss=.4, val_acc=.8, val_auc=.81, val_f1=.76), 0)
            for parameter in trainer.model.parameters():
                parameter.data.fill_(3.)
            trainer._record_average_candidates(
                dict(val_loss=.5, val_acc=.83, val_auc=.85, val_f1=.8), 1)
            before_shapes = {key: value.shape for key, value in trainer.model.state_dict().items()}
            trainer._load_averaged_candidates()
            # Four selectors share epoch 1; it must not receive four times the weight.
            for key, value in trainer.model.state_dict().items():
                self.assertEqual(value.shape, before_shapes[key])
                torch.testing.assert_close(value, torch.full_like(value, 2.), rtol=0, atol=0)
            self.assertTrue((Path(directory) / 'best_averaged_model.pth.m').exists())

    def test_default_trainer_does_not_record_or_replace_parameters(self):
        trainer = trainer_class()()
        trainer.args = SimpleNamespace()
        trainer._record_average_candidates({}, 0)
        trainer._load_averaged_candidates()
        self.assertFalse(hasattr(trainer, '_checkpoint_average_best'))

    def test_differing_integer_buffers_are_rejected(self):
        with TemporaryDirectory() as directory:
            for selector, count in (('loss', 1), ('acc', 2)):
                torch.save(dict(weight=torch.ones(2), count=torch.tensor(count)),
                           Path(directory) / f'best_val_{selector}.pth')
            metadata = dict(best=dict(loss=dict(epoch=0), acc=dict(epoch=1)))
            with self.assertRaisesRegex(ValueError, 'Nonfloating buffer differs'):
                average_checkpoints(directory, metadata, ['loss', 'acc'], 'cpu')


if __name__ == '__main__':
    unittest.main()
