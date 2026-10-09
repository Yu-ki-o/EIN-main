"""GroupGain configs, dispatch, and the real three-stage training lifecycle.

Extract the entry-point functions with AST so unrelated text encoders and
backbones do not become dependencies of this synthetic integration test. The
GroupGain model, grouping, optimizers, losses, metrics, and checkpoints are real.
"""

import ast
import argparse
from contextlib import chdir, redirect_stdout
from copy import deepcopy
from datetime import datetime
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import numpy as np
from torch_geometric.data import Data
import yaml

from model.GroupGain import GroupGain


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Pheme", "Weibo", "DRWeibo")


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def load_config(dataset, method="GroupGain"):
    path = ROOT / "configs" / "EIN" / f"{dataset}_{method}_word2vec.yaml"
    return SimpleNamespace(**yaml.safe_load(path.read_text()))


def tiny_args(dataset="DRWeibo", **updates):
    args = load_config(dataset)
    values = dict(
        device="cpu", hidden_dim=8, in_feats=4, n_layers_conv=1,
        dropout=0.0, batch_size=2, num_workers=0, debug=False,
        group_gain_teacher_max_epochs=1, group_gain_teacher_patience=1,
        group_gain_gain_head_epochs=1, group_gain_student_max_epochs=1,
        group_gain_student_patience=1, group_gain_correct_only=False,
        group_gain_full_context_probability=1.0,
        use_tensorboard=False, eval_only=False,
    )
    values.update(updates)
    for name, value in values.items():
        setattr(args, name, value)
    return args


def tiny_datasets():
    """Different event IDs in each split, with both labels in every split."""
    datasets = []
    for split in ("train", "val", "test"):
        graphs = []
        for label in (0, 1):
            graphs.append(Data(
                x=torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0],
                                [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
                + label * 0.1,
                directed_edge_index=torch.tensor([[0, 0, 2], [1, 2, 3]]),
                root_index=torch.tensor([0]), y=torch.tensor([label]),
                graph_id=f"{split}-{label}",
            ))
        datasets.append(graphs)
    return tuple(datasets)


class GroupGainConfigTest(unittest.TestCase):
    def test_main_import_and_result_naming(self):
        tree = ast.parse((ROOT / "main.py").read_text())
        self.assertTrue(any(
            isinstance(node, ast.ImportFrom) and node.module == "supervisor"
            and any(alias.name == "EIN_GroupGain_supervisor" for alias in node.names)
            for node in tree.body
        ))
        names = ["_safe_filename_part", "_selection_metric_part",
                 "_summary_model_parts", "build_summary_filename"]
        namespace = load_functions(ROOT / "main.py", names, {"re": re})
        for dataset in DATASETS:
            args = load_config(dataset)
            self.assertEqual(namespace["build_summary_filename"](args),
                             "summary_GroupGain_RelationGNN_undirected_val_loss_word2vec.txt")
            self.assertEqual(namespace["_summary_model_parts"](args),
                             ("GroupGain", "RelationGNN"))

    def test_three_configs_reuse_original_features_splits_and_cache(self):
        names = ["_as_bool", "_safe_cache_part", "_graph_dataset_cache_part",
                 "_requires_ragcl_centrality", "get_dataset_cache_name", "load_graph_dataset"]
        dataset_factory = Mock()
        namespace = load_functions(
            ROOT / "supervisor.py", names,
            {"re": re, "ResGCNTreeDataset": dataset_factory},
        )
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                args = load_config(dataset)
                baseline = load_config(dataset, "GCN")
                self.assertEqual(args.base_model, "GroupGain")
                self.assertEqual(args.dataset, dataset)
                self.assertNotEqual(args.result_name, baseline.result_name)
                for key in ("in_feats", "vector_size", "word_embedding", "split",
                            "max_hop", "tokenize_mode", "language", "undirected"):
                    self.assertEqual(getattr(args, key), getattr(baseline, key), key)
                for key, expected in (
                    ("group_gain_variant", "full_model"),
                    ("group_gain_group_pooling", "max"),
                    ("group_gain_stance_target", "generic"),
                    ("group_gain_gate_mode", "both"),
                    ("selection_metric", "val_loss"),
                    ("group_gain_edge_direction", None),
                ):
                    self.assertEqual(getattr(args, key), expected, key)
                for stage in ("teacher_max_epochs", "gain_head_epochs", "student_max_epochs"):
                    self.assertGreater(getattr(args, "group_gain_" + stage), 0)
                self.assertEqual(namespace["get_dataset_cache_name"](args),
                                 namespace["get_dataset_cache_name"](baseline))
                self.assertFalse(namespace["_requires_ragcl_centrality"](args))
                dataset_factory.reset_mock()
                encoder = object()
                namespace["load_graph_dataset"](args, "synthetic/train", encoder)
                dataset_factory.assert_called_once_with(
                    "synthetic/train", args.word_embedding, encoder, args.undirected, args=args,
                )

    def test_supervisor_passes_native_model_to_dedicated_trainer(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                args = tiny_args(dataset)
                datasets = tiny_datasets()
                trainer_factory = Mock()
                trainer_factory.return_value.train_process.return_value = {"acc": 0.5}
                encoder = Mock()
                namespace = load_functions(
                    ROOT / "supervisor.py", ["EIN_GroupGain_supervisor"],
                    {"init_seed": Mock(), "resolve_device": lambda _: torch.device("cpu"),
                     "dataset_paths": lambda a, d: (f"dataset/{d}/source", "unused"),
                     "build_text_encoder": encoder,
                     "build_experiment_datasets": Mock(return_value=datasets),
                     "GroupGain": GroupGain, "GroupGainTrainer": trainer_factory},
                )
                self.assertEqual(namespace["EIN_GroupGain_supervisor"](args), {"acc": 0.5})
                trainer_factory.return_value.train_process.assert_called_once()
                received_datasets, model, received_args, device = trainer_factory.call_args.args
                self.assertIs(received_datasets, datasets)
                self.assertIsInstance(model, GroupGain)
                self.assertIs(received_args, args)
                self.assertEqual(device, torch.device("cpu"))
                self.assertEqual(model(tiny_datasets()[2]).shape, (2, 2))


class CountingLoader:
    """Observe whether the held-out loader is used before final evaluation."""

    def __init__(self, loader):
        self.loader = loader
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class GroupGainSplitOverlapTest(unittest.TestCase):
    def make_trainer(self, datasets, directory):
        import trainer.GroupGain_trainer as implementation

        args = tiny_args()
        model = GroupGain(args.in_feats, args.hidden_dim, args.num_classes, args)
        with patch.object(implementation, "get_log_dir", return_value=directory), \
                patch.object(implementation, "get_logger", return_value=Mock(
                    handlers=[Mock(spec=logging.FileHandler)])):
            return implementation.GroupGainTrainer(datasets, model, args, "cpu")

    def test_identical_training_copies_are_excluded_for_explicit_and_filename_ids(self):
        class FilenameDataset(list):
            def __init__(self, graphs):
                super().__init__(graphs)
                self.raw_file_names = [graph.graph_id + ".json" for graph in graphs]
                for graph in graphs:
                    del graph.graph_id

        for use_filenames in (False, True):
            with self.subTest(use_filenames=use_filenames), tempfile.TemporaryDirectory() as directory:
                datasets = tiny_datasets()
                datasets[0].extend([deepcopy(datasets[1][0]), deepcopy(datasets[2][1])])
                if use_filenames:
                    datasets = tuple(FilenameDataset(graphs) for graphs in datasets)
                trainer = self.make_trainer(datasets, directory)
                suffix = ".json" if use_filenames else ""
                self.assertEqual([graph.graph_id for graph in trainer.grouped_datasets["train"]],
                                 ["train-0" + suffix, "train-1" + suffix])
                for split in ("val", "test"):
                    self.assertEqual([graph.graph_id for graph in trainer.grouped_datasets[split]],
                                     [split + "-0" + suffix, split + "-1" + suffix])
                self.assertEqual(len(trainer.train_loader.dataset), 2)
                summary = json.loads(trainer.grouping_path.read_text())["train"]
                self.assertEqual((summary["input_events"], summary["events"]), (4, 2))
                self.assertEqual([(row["graph_id"], row["dataset_index"], row["retained_split"])
                                  for row in summary["excluded_overlap_events"]],
                                 [("val-0" + suffix, 2, "val"), ("test-1" + suffix, 3, "test")])
                self.assertEqual([row["dataset_index"] for row in summary["event_diagnostics"]],
                                 [0, 1])
                self.assertEqual(trainer.logger.warning.call_count, 2)

    def test_matching_ids_with_conflicting_content_or_labels_still_fail(self):
        for field in ("x", "directed_edge_index", "y"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                datasets = tiny_datasets()
                duplicate = deepcopy(datasets[2][1])
                if field == "x":
                    duplicate.x[0, 0] += 0.25
                elif field == "directed_edge_index":
                    duplicate.directed_edge_index[0, -1] = 0
                else:
                    duplicate.y = 1 - duplicate.y
                datasets[0].append(duplicate)
                with self.assertRaisesRegex(ValueError, "different graph content or labels"):
                    self.make_trainer(datasets, directory)

    def test_validation_test_overlap_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            datasets = tiny_datasets()
            datasets[2].append(deepcopy(datasets[1][0]))
            with self.assertRaisesRegex(ValueError, "occurs in both val and test"):
                self.make_trainer(datasets, directory)

    def test_duplicates_within_each_split_still_fail(self):
        for index, split in enumerate(("train", "val", "test")):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                datasets = tiny_datasets()
                datasets[index].append(deepcopy(datasets[index][0]))
                with self.assertRaisesRegex(ValueError, "Duplicate event identifier.*" + split):
                    self.make_trainer(datasets, directory)

    def test_excluding_all_training_events_reports_empty_split(self):
        with tempfile.TemporaryDirectory() as directory:
            datasets = list(tiny_datasets())
            datasets[0] = deepcopy(datasets[2])
            with self.assertRaisesRegex(ValueError, "train split is empty after excluding"):
                self.make_trainer(datasets, directory)


class GroupGainTrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_isolated_main_cli_runs_one_seed_and_writes_matching_summary(self):
        """Execute the real CLI body, supervisor, and trainer on tiny graphs.

        This isolates optional imports and substitutes text/data preparation;
        it does not claim the full main module or real dataset training ran.
        """
        import trainer.GroupGain_trainer as implementation

        with tempfile.TemporaryDirectory() as directory:
            configuration = tiny_args(device="cuda", seed=99)
            config_path = Path(directory) / "tiny_group_gain.yaml"
            config_path.write_text(yaml.safe_dump(vars(configuration)))
            logger_namespace = load_functions(
                ROOT / "utils" / "logger.py",
                ["_safe_path_part", "get_result_name", "get_result_group", "get_log_dir"],
                {"os": os, "re": re, "datetime": datetime,
                 "__file__": str(Path(directory) / "utils" / "logger.py")},
            )
            prepared = Mock(return_value=tiny_datasets())
            text_encoder = Mock()
            constructed_trainers = []

            def create_trainer(*parameters):
                trainer = implementation.GroupGainTrainer(*parameters)
                constructed_trainers.append(trainer)
                return trainer

            namespace = load_functions(
                ROOT / "main.py",
                ["_safe_filename_part", "_selection_metric_part", "_summary_model_parts",
                 "build_summary_filename", "normalize_device_arg", "summarize_results"],
                {"__name__": "__main__", "argparse": argparse, "yaml": yaml,
                 "np": np, "os": os, "re": re,
                 "get_result_name": logger_namespace["get_result_name"],
                 "torch": torch, "init_seed": Mock(),
                 "build_text_encoder": text_encoder,
                 "build_experiment_datasets": prepared,
                 "GroupGain": GroupGain, "GroupGainTrainer": create_trainer},
            )
            load_functions(
                ROOT / "supervisor.py",
                ["_as_bool", "_safe_cache_part", "_graph_dataset_cache_part",
                 "_requires_ragcl_centrality", "get_dataset_cache_name", "dataset_paths",
                 "resolve_device", "EIN_GroupGain_supervisor"], namespace,
            )
            tree = ast.parse((ROOT / "main.py").read_text())
            main_body = next(node for node in tree.body
                             if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                             and isinstance(node.test.left, ast.Name)
                             and node.test.left.id == "__name__")
            command = ["main.py", "--config_filename", str(config_path),
                       "--seed", "0", "--device", "cpu"]
            with chdir(directory), patch.object(sys, "argv", command), \
                    patch.object(implementation, "get_log_dir", logger_namespace["get_log_dir"]), \
                    patch.object(implementation, "get_logger", return_value=Mock(
                        handlers=[Mock(spec=logging.FileHandler)])), \
                    redirect_stdout(io.StringIO()):
                exec(compile(ast.Module(body=[main_body], type_ignores=[]),
                             str(ROOT / "main.py"), "exec"), namespace)

            self.assertEqual(len(constructed_trainers), 1)
            prepared.assert_called_once()
            text_encoder.assert_called_once()
            self.assertEqual(namespace["args"].seed, 0)
            self.assertEqual(namespace["args"].device, "cpu")
            self.assertEqual(len(namespace["results"]), 1)
            self.assertEqual(namespace["results"][0]["seed"], 0)
            trainer = constructed_trainers[0]
            self.assertEqual([row["stage"] for row in trainer.history],
                             ["teacher", "gain_head", "student", "test"])
            summary = (Path(directory) / "experiments" / "EIN" / "DRWeibo"
                       / configuration.result_name / "summary_val_loss.txt")
            self.assertEqual(summary.parent, trainer.log_dir.parent)
            self.assertTrue(Path(trainer.best_path).is_file())
            self.assertTrue(Path(trainer.teacher_path).is_file())
            text = summary.read_text()
            self.assertIn("Seed 0:", text)
            self.assertIn("Average results over 1 runs:", text)
            self.assertIn("Checkpoint selection metric: val_loss", text)

    def test_three_stage_training_uses_frozen_teacher_and_training_events(self):
        import trainer.GroupGain_trainer as implementation

        for dataset in DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(29)
                args = tiny_args(dataset)
                model = GroupGain(args.in_feats, args.hidden_dim, args.num_classes, args)
                component_names = ("classifier", "gain_head", "context_encoder")
                gradient_totals = {stage: {name: 0.0 for name in component_names}
                                   for stage in ("gain_head", "student")}
                current_stage = [None]
                hooks = []
                for name in component_names:
                    for parameter in getattr(model, name).parameters():
                        def record(gradient, name=name):
                            self.assertTrue(torch.isfinite(gradient).all())
                            gradient_totals[current_stage[0]][name] += gradient.abs().sum().item()
                        hooks.append(parameter.register_hook(record))
                seen_gain_ids = []
                original_gain_targets = implementation.compute_gain_targets

                def gain_targets(teacher, graphs, *positional, **options):
                    self.assertFalse(teacher.training)
                    self.assertTrue(all(not p.requires_grad and p.grad is None
                                        for p in teacher.parameters()))
                    self.assertEqual(options.get("split", "train"), "train")
                    seen_gain_ids.extend(graph.graph_id for graph in graphs)
                    self.assertTrue(all(graph.diagnostics.get("split") == "train"
                                        for graph in graphs))
                    return original_gain_targets(teacher, graphs, *positional, **options)

                with patch.object(implementation, "get_log_dir", return_value=directory), \
                        patch.object(implementation, "get_logger", return_value=Mock(
                            handlers=[Mock(spec=logging.FileHandler)])), \
                        patch.object(implementation, "compute_gain_targets", side_effect=gain_targets):
                    trainer = implementation.GroupGainTrainer(
                        tiny_datasets(), model, args, torch.device("cpu"),
                    )
                    held_out = CountingLoader(trainer.test_loader)
                    trainer.test_loader = held_out
                    original_epoch = trainer.train_epoch

                    def train_epoch(stage, *parameters):
                        current_stage[0] = stage
                        return original_epoch(stage, *parameters)

                    with patch.object(trainer, "train_epoch", side_effect=train_epoch):
                        result = trainer.train_process()

                self.assertEqual(held_out.iterations, 1)
                self.assertEqual([row["stage"] for row in trainer.history],
                                 ["teacher", "gain_head", "student", "test"])
                self.assertTrue(seen_gain_ids)
                self.assertEqual(set(seen_gain_ids), {"train-0", "train-1"})
                self.assertFalse(trainer.teacher.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None
                                    for p in trainer.teacher.parameters()))
                for name, total in gradient_totals["student"].items():
                    self.assertGreater(total, 0, name)
                self.assertEqual(gradient_totals["gain_head"]["classifier"], 0)
                for name in ("gain_head", "context_encoder"):
                    self.assertGreater(gradient_totals["gain_head"][name], 0, name)
                for hook in hooks:
                    hook.remove()
                self.assertTrue(Path(trainer.teacher_path).is_file())
                self.assertTrue(Path(trainer.best_path).is_file())
                for stage in ("teacher", "gain_head", "student"):
                    self.assertEqual(sum(row["stage"] == stage for row in trainer.history), 1, stage)
                self.assertEqual(sum(row["stage"] == "test" for row in trainer.history), 1)
                for split, graphs in trainer.grouped_datasets.items():
                    self.assertTrue(all(graph.x.device.type == "cpu" for graph in graphs))
                    self.assertTrue(all(graph.diagnostics["split"] == split for graph in graphs))
                for metric in ("acc", "auc", "f1"):
                    self.assertIn(metric, result)

                restored = GroupGain.load_checkpoint(trainer.best_path).eval()
                self.assertEqual(restored.checkpoint_metadata["selection_metric"], "val_loss")
                unlabeled = deepcopy(tiny_datasets()[2])
                with torch.no_grad():
                    expected = restored(unlabeled)
                    for graph in unlabeled:
                        graph.y = 1 - graph.y
                    torch.testing.assert_close(restored(unlabeled), expected)
                    for graph in unlabeled:
                        del graph.y
                    torch.testing.assert_close(restored(unlabeled), expected)

    def test_eval_only_loads_standalone_student_without_training_or_reference(self):
        import trainer.GroupGain_trainer as implementation

        with tempfile.TemporaryDirectory() as directory:
            args = tiny_args()
            saved = GroupGain(args.in_feats, args.hidden_dim, args.num_classes, args).eval()
            checkpoint = str(Path(directory) / "student.pth")
            saved.save_checkpoint(checkpoint)
            args.eval_only = True
            args.checkpoint_path = checkpoint
            # Inference grouping settings must follow the saved model.
            args.group_gain_group_cos_threshold = 0.5
            with patch.object(implementation, "get_log_dir", return_value=directory), \
                    patch.object(implementation, "get_logger", return_value=Mock(
                        handlers=[Mock(spec=logging.FileHandler)])), \
                    patch.object(GroupGain, "teacher_training_loss", side_effect=AssertionError(
                        "eval_only must never train a teacher")), \
                    patch.object(implementation, "compute_gain_targets", side_effect=AssertionError(
                        "eval_only must never generate labeled gain targets")):
                fresh = GroupGain(args.in_feats, args.hidden_dim, args.num_classes, args)
                trainer = implementation.GroupGainTrainer(
                    ([], [], tiny_datasets()[2]), fresh, args, torch.device("cpu"),
                )
                result = trainer.train_process()
            self.assertEqual(trainer.model.config["group_cos_threshold"],
                             saved.config["group_cos_threshold"])
            self.assertIsNone(trainer.teacher)
            self.assertEqual(trainer.grouped_datasets["train"], [])
            self.assertEqual(trainer.grouped_datasets["val"], [])
            self.assertEqual([row["stage"] for row in trainer.history], ["test"])
            self.assertTrue(all(name in result for name in ("acc", "auc", "f1")))
            self.assertFalse((Path(directory) / "best_teacher.pth").exists())


if __name__ == "__main__":
    unittest.main()
