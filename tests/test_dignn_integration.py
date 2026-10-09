"""Config, cache routing and actual EINTrainer update checks.

AST-load entry functions to avoid requiring unrelated text encoders and
pretrained embeddings. Only dataset/logging setup is mocked; forward,
classification, auxiliary losses, backward and optimizer updates are real.
"""

import ast
from collections import defaultdict
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
import yaml

from model.DIGNN import DIGNN


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def load_config(dataset, method="DIGNN"):
    path = ROOT / "configs" / "EIN" / f"{dataset}_{method}_word2vec.yaml"
    return SimpleNamespace(**yaml.safe_load(path.read_text()))


class DIGNNIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_main_dispatch_and_result_names(self):
        tree = ast.parse((ROOT / "main.py").read_text())
        self.assertTrue(any(isinstance(node, ast.ImportFrom)
                            and node.module == "supervisor"
                            and any(alias.name == "EIN_DIGNN_supervisor" for alias in node.names)
                            for node in tree.body))
        namespace = load_functions(ROOT / "main.py", ["_safe_filename_part", "_selection_metric_part",
                                   "_summary_model_parts", "build_summary_filename"], {"re": re})
        for dataset in ("Pheme", "Weibo", "DRWeibo"):
            self.assertEqual(namespace["build_summary_filename"](load_config(dataset)),
                             "summary_DIGNN_undirected_val_loss_word2vec.txt")

    def test_same_features_splits_and_processed_cache_as_gcn(self):
        factory = Mock()
        namespace = load_functions(ROOT / "supervisor.py",
                                   ["_as_bool", "_safe_cache_part", "_graph_dataset_cache_part",
                                    "_requires_ragcl_centrality", "get_dataset_cache_name", "load_graph_dataset"],
                                   {"re": re, "ResGCNTreeDataset": factory})
        for dataset in ("Pheme", "Weibo", "DRWeibo"):
            with self.subTest(dataset=dataset):
                args, baseline = load_config(dataset), load_config(dataset, "GCN")
                self.assertEqual(args.base_model, "DIGNN")
                self.assertEqual(args.dataset, dataset)
                self.assertNotEqual(args.result_name, baseline.result_name)
                # Tuned optimization/model settings need not match the GCN.
                # Only data settings determine compatibility of cached graphs.
                for key in ("in_feats", "vector_size", "split", "max_hop", "tokenize_mode",
                            "language", "word_embedding", "undirected", "k"):
                    self.assertEqual(getattr(args, key), getattr(baseline, key), key)
                self.assertEqual(namespace["get_dataset_cache_name"](args),
                                 namespace["get_dataset_cache_name"](baseline))
                encoder = object()
                factory.reset_mock()
                namespace["load_graph_dataset"](args, "mock/train", encoder)
                factory.assert_called_once_with("mock/train", args.word_embedding, encoder,
                                                args.undirected, args=args)

    def test_supervisor_and_real_native_training_update_for_each_config(self):
        tree = ast.parse((ROOT / "trainer" / "EIN_trainer.py").read_text())
        trainer_class = next(node for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "EINTrainer")
        epoch = next(node for node in trainer_class.body
                     if isinstance(node, ast.FunctionDef) and node.name == "train_epoch")
        epoch_namespace = {"defaultdict": defaultdict, "F": F}
        exec(compile(ast.Module(body=[epoch], type_ignores=[]), "EIN_trainer.py", "exec"), epoch_namespace)
        for dataset in ("Pheme", "Weibo", "DRWeibo"):
            with self.subTest(dataset=dataset):
                args = load_config(dataset)
                args.device = "cpu"
                graphs = []
                for label, n in ((0, 4), (1, 3), (0, 1)):
                    edges = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
                    graphs.append(Data(x=torch.randn(n, args.in_feats), edge_index=edges,
                                       directed_edge_index=edges.clone(), y=torch.tensor([label]),
                                       user_state=torch.zeros(1, args.max_hop, 3)))
                datasets = (graphs, graphs, graphs)
                factory = Mock()
                factory.return_value.train_process.return_value = {"acc": 0.5}
                namespace = load_functions(
                    ROOT / "supervisor.py", ["EIN_DIGNN_supervisor"],
                    {"init_seed": Mock(), "resolve_device": lambda _: torch.device("cpu"),
                     "dataset_paths": lambda a, d: (f"dataset/{d}/source", "unused"),
                     "build_text_encoder": Mock(), "build_experiment_datasets": Mock(return_value=datasets),
                     "DIGNN": DIGNN, "EINTrainer": factory})
                self.assertEqual(namespace["EIN_DIGNN_supervisor"](args), {"acc": 0.5})
                factory.return_value.train_process.assert_called_once()
                _, model, optimizer, received_args, device = factory.call_args.args
                self.assertIsInstance(model, DIGNN)
                self.assertIs(received_args, args)
                trainer = SimpleNamespace(
                    model=model, optimizer=optimizer, train_loader=[Batch.from_data_list(graphs)],
                    train_per_epoch=1, _move_to_device=lambda data: data.to(device),
                    _accumulate_diagnostics=Mock(), _average_diagnostics=Mock(return_value={}),
                    _write_train_tensorboard=Mock(), logger=Mock())
                before = model.classifier.weight.detach().clone()
                loss = epoch_namespace["train_epoch"](trainer, 1)
                self.assertTrue(bool(torch.isfinite(torch.tensor(loss))))
                train_metrics = trainer._write_train_tensorboard.call_args.args[1]
                self.assertGreater(train_metrics["train_auxiliary_loss"], 0)
                self.assertFalse(torch.equal(before, model.classifier.weight))


if __name__ == "__main__":
    unittest.main()
