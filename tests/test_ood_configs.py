"""Regression checks for translating existing model settings to source-only OOD."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.generate_ood_configs import make_ood_config, read_manifest


ROOT = Path(__file__).resolve().parents[1]


class OODConfigTest(unittest.TestCase):
    def test_existing_model_kept_without_inheriting_target_training_artifacts(self):
        base = yaml.safe_load((ROOT / "configs/EIN/Pheme_P2T3_word2vec.yaml").read_text())
        base.update(
            kpg_checkpoint="old.pt", kpg_test_only=True, eval_only=True,
            dataset_cache_name="id-cache", legacy_dataset_cache_name="id-legacy",
            word2vec_model_path="target.model", see_ttt_enabled=True,
        )
        original = copy.deepcopy(base)
        config = make_ood_config(
            base, "dataset/ood_splits/weibo_to_pheme/seed_{seed}.json",
            manifest={"source_datasets": ["Weibo"], "target_dataset": "Pheme"},
        )
        self.assertEqual(base, original)
        self.assertEqual(config["base_model"], "P2T3")
        for key in ("hidden_dim", "p2t3_num_heads", "dropout"):
            if key in base:
                self.assertEqual(config[key], base[key])
        self.assertEqual(config["in_feats"], 768)
        for key in ("p2t3_pretrained_path", "kpg_checkpoint", "kpg_test_only",
                    "eval_only", "dataset_cache_name", "legacy_dataset_cache_name",
                    "word2vec_model_path"):
            self.assertNotIn(key, config)
        self.assertFalse(config["see_ttt_enabled"])
        self.assertEqual(config["classification_class_weights"], [1.0, 1.0])

    def test_manifest_controls_target_and_cross_language_embedding(self):
        base = {"base_model": "BiGCN", "dataset": "DRWeibo", "vector_size": 200}
        manifest = {"source_datasets": ["DRWeibo"], "target_dataset": "Pheme"}
        with self.assertRaisesRegex(ValueError, "disagrees"):
            make_ood_config(base, "seed_{seed}.json", dataset="Weibo", manifest=manifest)
        with self.assertRaisesRegex(ValueError, "Cross-language"):
            make_ood_config(base, "seed_{seed}.json", embedding="word2vec", manifest=manifest)
        manifest["target_dataset"] = "Weibo"
        config = make_ood_config(base, "seed_{seed}.json", embedding="word2vec", manifest=manifest)
        self.assertEqual(config["dataset"], "Weibo")
        self.assertEqual(config["language"], "ch")
        self.assertEqual(config["in_feats"], 200)
        with self.assertRaisesRegex(ValueError, "Prepare the OOD manifest"):
            make_ood_config(base, "absent.json", dataset="Weibo", embedding="word2vec")

    def test_manifest_reader_and_pre_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            pattern = str(Path(folder) / "seed_{seed}.json")
            self.assertIsNone(read_manifest(pattern))
            manifest = {"source_datasets": ["Weibo"], "target_dataset": "DRWeibo"}
            (Path(folder) / "seed_0.json").write_text(json.dumps(manifest))
            self.assertEqual(read_manifest(pattern), manifest)
            with self.assertRaisesRegex(ValueError, "placeholder"):
                read_manifest(str(Path(folder) / "seed_{unknown}.json"))

    def test_six_pairs_use_same_source_hyperparameters(self):
        configs = [yaml.safe_load(path.read_text())
                   for path in sorted((ROOT / "configs/ood").glob("*_BiGCN_e5.yaml"))]
        pairs = {(config["ood_source_datasets"][0], config["dataset"]) for config in configs}
        datasets = {"DRWeibo", "Weibo", "Pheme"}
        self.assertEqual(pairs, {(source, target) for source in datasets
                                for target in datasets if source != target})
        for source in datasets:
            source_configs = [config for config in configs
                              if config["ood_source_datasets"] == [source]]
            ignored = {"dataset", "language", "tokenize_mode", "result_name", "ood_manifest"}
            comparable = [{key: value for key, value in config.items() if key not in ignored}
                          for config in source_configs]
            self.assertEqual(comparable[0], comparable[1])
            self.assertTrue(all(config["experiment_mode"] == "ood" for config in source_configs))


if __name__ == "__main__":
    unittest.main()

