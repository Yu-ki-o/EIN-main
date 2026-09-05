import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.enrich_pheme_timestamps import EnrichmentError, enrich_directory


def _raw_tweet(tweet_id, user_id, text, created_at, parent=None):
    return {
        "id": int(tweet_id),
        "id_str": tweet_id,
        "user": {"id": int(user_id), "id_str": user_id},
        "text": text,
        "created_at": created_at,
        "in_reply_to_status_id": None if parent is None else int(parent),
        "in_reply_to_status_id_str": parent,
    }


def _add_json(archive, name, value):
    payload = json.dumps(value).encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


class PhemeTimestampEnrichmentTests(unittest.TestCase):
    def _fixture(self, root):
        input_dir = root / "input"
        input_dir.mkdir()
        processed = {
            "source": {"tweet id": "100", "content": "source", "label": 0},
            "comment": [
                {
                    "comment id": 0,
                    "parent": -1,
                    "content": "duplicate",
                    "user id": 2,
                    "stance_label": 0,
                    "state": 0,
                    "hop": 1,
                },
                {
                    "comment id": 1,
                    "parent": -1,
                    "content": "duplicate",
                    "user id": 2,
                    "stance_label": 1,
                    "state": 1,
                    "hop": 1,
                },
                {
                    "comment id": 2,
                    "parent": 1,
                    "content": "child",
                    "user id": 3,
                    "stance_label": 0,
                    "state": 1,
                    "hop": 2,
                },
            ],
            "state": {
                "1-hop": {"state_0": 1, "state_1": 1},
                "2-hop": {"state_0": 0, "state_1": 1},
            },
        }
        with (input_dir / "100.json").open("w", encoding="utf-8") as file_obj:
            json.dump(processed, file_obj)

        archive_path = root / "PHEME_veracity.tar.bz2"
        prefix = "all/event/non-rumours/100"
        with tarfile.open(archive_path, "w:gz") as archive:
            _add_json(
                archive,
                prefix + "/source-tweets/100.json",
                _raw_tweet("100", "1", "source", "Wed Jan 01 00:00:00 +0000 2020"),
            )
            # Archive order is deliberately the reverse of structure order.
            _add_json(
                archive,
                prefix + "/reactions/102.json",
                _raw_tweet(
                    "102", "2", "duplicate", "Wed Jan 01 02:00:00 +0000 2020", "100"
                ),
            )
            _add_json(
                archive,
                prefix + "/reactions/101.json",
                _raw_tweet(
                    "101", "2", "duplicate", "Wed Jan 01 01:00:00 +0000 2020", "100"
                ),
            )
            _add_json(
                archive,
                prefix + "/reactions/103.json",
                _raw_tweet(
                    "103", "3", "child", "Wed Jan 01 03:00:00 +0000 2020", "102"
                ),
            )
            _add_json(
                archive,
                prefix + "/structure.json",
                {"100": {"101": {}, "102": {"103": {}}}},
            )
        return input_dir, archive_path

    def test_enriches_and_uses_structure_order_for_duplicate_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir, archive_path = self._fixture(root)
            output_dir = root / "output"
            report = enrich_directory(input_dir, archive_path, output_dir)

            with (output_dir / "100.json").open("r", encoding="utf-8") as file_obj:
                result = json.load(file_obj)
            self.assertEqual(
                result["source"]["created_at"],
                "Wed Jan 01 00:00:00 +0000 2020",
            )
            self.assertEqual(
                [comment["tweet id"] for comment in result["comment"]],
                ["101", "102", "103"],
            )
            self.assertEqual(result["comment"][2]["parent tweet id"], "102")
            self.assertEqual(result["comment"][2]["stance_label"], 0)
            self.assertEqual(report["events"], 1)
            self.assertEqual(report["comments"], 3)
            self.assertEqual(report["duplicate_candidate_resolutions"], 1)
            self.assertFalse((output_dir / "timestamp_enrichment_summary.json").exists())
            self.assertTrue(
                (root / "output_timestamp_enrichment_summary.json").is_file()
            )

    def test_refuses_in_place_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir, archive_path = self._fixture(root)
            with self.assertRaises(EnrichmentError):
                enrich_directory(input_dir, archive_path, input_dir)


if __name__ == "__main__":
    unittest.main()
