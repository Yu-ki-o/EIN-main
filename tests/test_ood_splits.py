"""Protocol invariants with tiny post corpora; no graph/GPU dependency."""
import copy
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from utils.ood_splits import (
    build_ood_manifest, build_ood_manifests, load_ood_manifest, manifest_digest,
    materialize_ood_posts, read_pheme_events, write_ood_manifest,
)


class OODSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for dataset in ("DRWeibo", "Weibo", "Pheme"):
            for i in range(12):
                self.post(dataset, str(i), "{} item {}".format(dataset, i), i % 2,
                          root_id=dataset + str(i), theme="社会生活")

    def post(self, dataset, filename, text, label, root_id=None, theme=None):
        directory = self.root / dataset / "source"
        directory.mkdir(parents=True, exist_ok=True)
        data = {"source": {"tweet id": root_id or filename, "content": text,
                           "label": label, "theme": theme, "time": 123},
                "comment": [{"comment id": 0, "parent": -1, "content": "reply", "state": 2}],
                "state": [1, 2], "custom_graph_field": {"keep": True}}
        path = directory / (filename + ".json")
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def manifest(self, target="Pheme", **kwargs):
        return build_ood_manifest(self.root, sources=["DRWeibo"], target=target, workers=2, **kwargs)

    def save(self, manifest):
        path = self.root / "manifest.json"
        write_ood_manifest(manifest, path, overwrite=True)
        return path

    def test_seed_reproducibility_and_fingerprint(self):
        one, two = self.manifest(), self.manifest()
        self.assertEqual(one, two)
        different = self.manifest(seed=1)
        self.assertNotEqual(one["splits"]["train"], different["splits"]["train"])
        self.assertEqual(one["splits"]["test"], different["splits"]["test"])
        self.assertNotEqual(manifest_digest(one), manifest_digest(different))
        self.assertEqual(load_ood_manifest(self.save(one)), one)

    def test_target_labels_do_not_control_source_selection(self):
        before = self.manifest()
        for path in (self.root / "Pheme" / "source").glob("*.json"):
            post = json.loads(path.read_text())
            post["source"]["label"] = 1
            path.write_text(json.dumps(post), encoding="utf-8")
        after = self.manifest()
        self.assertEqual(before["splits"]["train"], after["splits"]["train"])
        self.assertEqual(before["splits"]["val"], after["splits"]["val"])
        self.assertTrue(after["diagnostics"]["single_class_test"])
        self.assertNotEqual(manifest_digest(before), manifest_digest(after))

    def test_same_source_membership_across_targets(self):
        # Even an overlapping target does not change the source pool.
        self.post("Weibo", "duplicate", "DRWeibo item 0", 1)
        first, second = self.manifest("Pheme"), self.manifest("Weibo")
        self.assertEqual(first["splits"]["train"], second["splits"]["train"])
        self.assertEqual(first["splits"]["val"], second["splits"]["val"])
        self.assertEqual(second["diagnostics"]["deduplication"]["target_removed_source_overlap"], 1)

    def test_source_duplicate_components_excluded_before_split(self):
        # Transitive component A -- same text -- B -- same ID -- C.
        self.post("DRWeibo", "dup_a", "same body", 0, root_id="1000")
        self.post("DRWeibo", "dup_b", "SAME  body", 0, root_id="1001")
        self.post("DRWeibo", "dup_c", "different body", 1, root_id="1001")
        manifest = self.manifest()
        used = manifest["splits"]["train"] + manifest["splits"]["val"]
        self.assertFalse(any(r["file"].startswith("dup_") for r in used))
        self.assertEqual(manifest["diagnostics"]["deduplication"]["source_removed_conflicting_labels"], 3)
        load_ood_manifest(self.save(manifest))

    def test_source_same_label_duplicates_have_one_representative(self):
        self.post("DRWeibo", "z_duplicate", " DRWeibo \nITEM 0 ", 0)
        manifest = self.manifest()
        self.assertEqual(sum(len(manifest["splits"][s]) for s in ("train", "val")), 12)
        self.assertEqual(manifest["diagnostics"]["deduplication"]["source_removed_duplicates"], 1)
        load_ood_manifest(self.save(manifest))

    def test_target_overlap_removed_for_both_id_and_text(self):
        self.post("Weibo", "same_id", "unrelated", 0, root_id="DRWeibo0")
        self.post("Weibo", "same_text", "drweibo item 1", 1)
        manifest = self.manifest("Weibo")
        self.assertEqual(len(manifest["splits"]["test"]), 12)
        self.assertEqual(manifest["diagnostics"]["deduplication"]["target_removed_source_overlap"], 2)

    def test_id_namespaces_separate_platforms_and_empty_text_is_not_duplicate(self):
        self.post("Pheme", "same_id", "", 0, root_id="DRWeibo0")
        self.post("Pheme", "other_empty", "", 1)
        manifest = self.manifest()
        self.assertEqual(len(manifest["splits"]["test"]), 14)
        load_ood_manifest(self.save(manifest))

    def test_content_hash_detects_mutation_and_cache_identity_changes(self):
        before = self.manifest()
        path = self.save(before)
        self.post("DRWeibo", "0", "changed graph source", 0, root_id="DRWeibo0")
        with self.assertRaisesRegex(ValueError, "Source file changed"):
            load_ood_manifest(path)
        after = self.manifest()
        self.assertNotEqual(manifest_digest(before), manifest_digest(after))

    def test_manifest_tampering_and_duplicate_split_rejected(self):
        manifest = self.manifest()
        manifest["seed"] = 19
        path = self.root / "bad.json"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_ood_manifest(path)
        manifest = self.manifest()
        manifest["splits"]["val"].append(copy.deepcopy(manifest["splits"]["train"][0]))
        manifest["fingerprint"] = manifest_digest(manifest)
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Repeated file"):
            load_ood_manifest(path)

    def test_validation_detects_text_leakage_even_with_updated_hash(self):
        manifest = self.manifest()
        source_record = manifest["splits"]["train"][0]
        source_post = json.loads((self.root / "DRWeibo" / "source" / source_record["file"]).read_text())
        target_record = manifest["splits"]["test"][0]
        target_path = self.root / "Pheme" / "source" / target_record["file"]
        post = json.loads(target_path.read_text())
        post["source"]["content"] = source_post["source"]["content"]
        target_path.write_text(json.dumps(post))
        import hashlib
        target_record["sha256"] = hashlib.sha256(target_path.read_bytes()).hexdigest()
        manifest["fingerprint"] = manifest_digest(manifest)
        with self.assertRaisesRegex(ValueError, "Duplicate root ID/text"):
            load_ood_manifest(self.save(manifest))

    def test_materialization_preserves_graph_and_root_id(self):
        manifest = self.manifest()
        record = manifest["splits"]["train"][0]
        original = json.loads((self.root / "DRWeibo" / "source" / record["file"]).read_text())
        identifier, post = materialize_ood_posts(manifest, "train")[0]
        self.assertTrue(identifier.startswith("DRWeibo__"))
        self.assertEqual(post["source"].pop("domain_id"), 0)
        self.assertEqual(original, post)
        self.assertEqual(materialize_ood_posts(manifest, "test")[0][1]["source"]["domain_id"], -1)

    def test_unknown_themes_excluded_in_theme_protocol(self):
        self.post("DRWeibo", "health0", "health zero", 0, theme="医药健康")
        self.post("DRWeibo", "health1", "health one", 1, theme="医药健康")
        for i in range(5):
            self.post("DRWeibo", "unknown" + str(i), "unknown " + str(i), 1, theme=None)
        manifest = build_ood_manifest(self.root, protocol="drweibo_theme", holdouts=["医药健康"])
        self.assertEqual(manifest["diagnostics"]["input"]["excluded_missing_domain"], 5)
        self.assertEqual(len(manifest["splits"]["test"]), 2)
        self.assertTrue(all(r["domain"] == "社会生活" for r in manifest["splits"]["train"]))
        self.assertTrue(all(r["domain"] == "医药健康" for r in manifest["splits"]["test"]))
        load_ood_manifest(self.save(manifest))

    def test_pheme_event_archive_auto_detects_compression(self):
        archive = self.root / "PHEME_veracity.tar.bz2"
        with tarfile.open(archive, "w:gz") as handle:
            for i in range(14):
                event = "charliehebdo" if i >= 12 else "ferguson"
                name = "all-rnr-annotated-threads/{}-all-rnr-threads/rumours/{}/source-tweets/{}.json".format(event, 1000 + i, 1000 + i)
                content = b"{}"
                info = tarfile.TarInfo(name)
                info.size = len(content)
                handle.addfile(info, io.BytesIO(content))
                self.post("Pheme", str(i), "Pheme event {}".format(i), i % 2, root_id=str(1000 + i))
        self.assertEqual(read_pheme_events(archive)["1012"], "charliehebdo")
        manifest = build_ood_manifest(self.root, protocol="pheme_event", holdouts=["charliehebdo"], pheme_archive=archive)
        self.assertEqual(len(manifest["splits"]["test"]), 2)
        self.assertEqual(manifest["source_domains"], {"ferguson": 0})
        load_ood_manifest(self.save(manifest))

    def test_existing_different_manifest_not_overwritten(self):
        manifest = self.manifest()
        path = self.save(manifest)
        write_ood_manifest(manifest, path)
        with self.assertRaises(FileExistsError):
            write_ood_manifest(self.manifest(seed=1), path)
        self.assertEqual(json.loads(path.read_text()), manifest)

    def test_seed_placeholder_and_invalid_config(self):
        manifest = self.manifest()
        write_ood_manifest(manifest, self.root / "seed_0.json")
        self.assertEqual(load_ood_manifest(self.root / "seed_{seed}.json", seed=0), manifest)
        with self.assertRaisesRegex(ValueError, "seed"):
            load_ood_manifest(self.root / "seed_0.json", seed=4)
        with self.assertRaises(ValueError):
            self.manifest(val_ratio=1)
        with self.assertRaises(ValueError):
            self.manifest(target="DRWeibo")
        with self.assertRaises(ValueError):
            build_ood_manifest(self.root, protocol="drweibo_theme", holdouts=["missing"])

    def test_all_pairs_cli_produces_six_manifests_with_matching_sources(self):
        output = self.root / "ood"
        script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_ood_splits.py"
        result = subprocess.run([sys.executable, str(script), "--all-pairs", "--dataset-root", str(self.root),
                                 "--output-dir", str(output), "--seeds", "0"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        files = sorted(output.glob("*/seed_0.json"))
        self.assertEqual(len(files), 6)
        groups = {}
        for path in files:
            manifest = load_ood_manifest(path)
            source = manifest["source_datasets"][0]
            if source in groups:
                self.assertEqual(groups[source]["train"], manifest["splits"]["train"])
                self.assertEqual(groups[source]["val"], manifest["splits"]["val"])
            else:
                groups[source] = manifest["splits"]


if __name__ == "__main__":
    unittest.main()
