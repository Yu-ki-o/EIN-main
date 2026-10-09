"""Config/dispatch smoke tests with real NodeIGM and EINTrainer epoch code.

Load the relevant functions via AST so these tests do not require unrelated
text-encoder packages, pretrained embeddings, or real datasets. Dataset and
logging setup is mocked; the model, optimizer, and epoch update are real.
"""

import ast
from collections import defaultdict
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
import yaml

from model.NodeIGM import NodeIGM


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def load_config(dataset, method="NodeIGM"):
    path = ROOT / "configs" / "EIN" / f"{dataset}_{method}_word2vec.yaml"
    return SimpleNamespace(**yaml.safe_load(path.read_text()))


class NodeIGMConfigTest(unittest.TestCase):
    def test_main_import_and_result_naming(self):
        tree = ast.parse((ROOT / "main.py").read_text())
        self.assertTrue(any(isinstance(node, ast.ImportFrom)
                            and node.module == "supervisor"
                            and any(alias.name == "EIN_NodeIGM_supervisor" for alias in node.names)
                            for node in tree.body))
        names = ["_safe_filename_part", "_selection_metric_part",
                 "_summary_model_parts", "build_summary_filename"]
        namespace = load_functions(ROOT / "main.py", names, {"re": re})
        for dataset in ("Pheme", "Weibo", "DRWeibo"):
            args = load_config(dataset)
            self.assertEqual(namespace["build_summary_filename"](args),
                             "summary_NodeIGM_GCN_undirected_val_loss_word2vec.txt")

    def test_configs_match_baseline_features_splits_and_cache(self):
        names = ["_as_bool", "_safe_cache_part", "_graph_dataset_cache_part",
                 "_requires_ragcl_centrality", "get_dataset_cache_name", "load_graph_dataset"]
        dataset_factory = Mock()
        namespace = load_functions(ROOT / "supervisor.py", names,
                                   {"re": re, "ResGCNTreeDataset": dataset_factory})
        for dataset in ("Pheme", "Weibo", "DRWeibo"):
            with self.subTest(dataset=dataset):
                args = load_config(dataset)
                baseline = load_config(dataset, "GCN")
                self.assertEqual(args.base_model, "NodeIGM")
                self.assertEqual(args.dataset, dataset)
                self.assertNotEqual(args.result_name, baseline.result_name)
                for key in ("in_feats", "vector_size", "hidden_dim", "n_layers_conv",
                            "dropout", "lr", "weight_decay", "split", "max_hop",
                            "tokenize_mode", "language", "selection_metric", "undirected"):
                    self.assertEqual(getattr(args, key), getattr(baseline, key), key)
                self.assertEqual(namespace["get_dataset_cache_name"](args),
                                 namespace["get_dataset_cache_name"](baseline))
                self.assertFalse(namespace["_requires_ragcl_centrality"](args))
                dataset_factory.reset_mock()
                encoder = object()
                namespace["load_graph_dataset"](args, "synthetic/train", encoder)
                dataset_factory.assert_called_once_with(
                    "synthetic/train", args.word_embedding, encoder, args.undirected, args=args)

    def test_supervisor_and_native_trainer_update_for_each_config(self):
        trainer_tree = ast.parse((ROOT / "trainer" / "EIN_trainer.py").read_text())
        trainer_class = next(node for node in trainer_tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "EINTrainer")
        epoch = next(node for node in trainer_class.body
                     if isinstance(node, ast.FunctionDef) and node.name == "train_epoch")
        epoch_namespace = {"defaultdict": defaultdict, "F": F}
        exec(compile(ast.Module(body=[epoch], type_ignores=[]), "EIN_trainer.py", "exec"),
             epoch_namespace)
        for dataset in ("Pheme", "Weibo", "DRWeibo"):
            with self.subTest(dataset=dataset):
                args = load_config(dataset)
                args.device = "cpu"
                graphs = []
                for label, num_nodes in ((0, 4), (1, 3), (0, 1)):
                    edges = torch.stack((torch.arange(num_nodes - 1),
                                         torch.arange(1, num_nodes)))
                    graphs.append(Data(x=torch.randn(num_nodes, args.in_feats),
                                       edge_index=edges, directed_edge_index=edges.clone(),
                                       y=torch.tensor([label]),
                                       user_state=torch.zeros(1, args.max_hop, 3)))
                datasets = (graphs, graphs, graphs)
                factory = Mock()
                factory.return_value.train_process.return_value = {"acc": 0.5}
                namespace = load_functions(
                    ROOT / "supervisor.py", ["EIN_NodeIGM_supervisor"],
                    {"init_seed": Mock(), "resolve_device": lambda _: torch.device("cpu"),
                     "dataset_paths": lambda a, d: (f"dataset/{d}/source", "unused"),
                     "build_text_encoder": Mock(),
                     "build_experiment_datasets": Mock(return_value=datasets),
                     "NodeIGM": NodeIGM, "EINTrainer": factory})
                self.assertEqual(namespace["EIN_NodeIGM_supervisor"](args), {"acc": 0.5})
                factory.return_value.train_process.assert_called_once()
                _, model, optimizer, received_args, device = factory.call_args.args
                self.assertIsInstance(model, NodeIGM)
                self.assertIs(received_args, args)
                batch = Batch.from_data_list(graphs)
                trainer = SimpleNamespace(
                    model=model, optimizer=optimizer, train_loader=[batch], train_per_epoch=1,
                    _move_to_device=lambda data: data.to(device),
                    _accumulate_diagnostics=Mock(), _average_diagnostics=Mock(return_value={}),
                    _write_train_tensorboard=Mock(), logger=Mock())
                before = model.classifier.weight.detach().clone()
                with patch.object(model, "auxiliary_loss", wraps=model.auxiliary_loss) as auxiliary:
                    loss = epoch_namespace["train_epoch"](trainer, 1)
                    auxiliary.assert_called_once()
                self.assertTrue(torch.isfinite(torch.tensor(loss)))
                self.assertFalse(torch.equal(before, model.classifier.weight))
                model.eval()
                with torch.no_grad():
                    self.assertEqual(model(batch)[0].shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
