"""Ensure the standard trainer measures score ranking, including OOD collapse."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class OODMetricTests(unittest.TestCase):
    def test_auc_uses_probabilities_when_all_hard_predictions_are_the_same(self):
        path = Path(__file__).resolve().parents[1] / 'trainer' / 'EIN_trainer.py'
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EINTrainer')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'test')
        namespace = dict(np=np, torch=torch, accuracy_score=accuracy_score,
                         f1_score=f1_score, roc_auc_score=roc_auc_score)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        probabilities = torch.tensor([[.9, .1], [.6, .4], [.8, .2], [.7, .3]])
        model = Mock(return_value=(probabilities.log(), None, None, None))
        trainer = SimpleNamespace(model=model,
                                  test_loader=[SimpleNamespace(y=torch.tensor([0, 1, 0, 1]))],
                                  _move_to_device=lambda x: x, logger=Mock(), _write_test_tensorboard=Mock())
        result = namespace['test'](trainer)
        self.assertEqual(result['auc'], 1.0)
        self.assertEqual(result['acc'], .5)
        self.assertEqual(result['f1'], 0.)
        self.assertAlmostEqual(result['macro_f1'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
