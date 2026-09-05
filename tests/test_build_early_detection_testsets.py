import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_early_detection_testsets import (
    build_cutoff_datasets,
    parse_time_seconds,
    truncate_post,
)


def _post():
    return {
        "source": {
            "tweet id": "event-1",
            "label": 1,
            "content": "source",
            "created_at": "2024-01-01T00:00:00+00:00",
        },
        "comment": [
            {
                "comment id": 0,
                "parent": -1,
                "content": "late sibling",
                "created_at": "2024-01-01T05:00:00+00:00",
                "stance_label": 0,
                "state": 0,
                "hop": 1,
            },
            {
                "comment id": 1,
                "parent": -1,
                "content": "early reply",
                "created_at": "2024-01-01T01:00:00+00:00",
                "stance_label": 1,
                "state": 1,
                "hop": 1,
            },
            {
                "comment id": 2,
                "parent": 1,
                "content": "nested reply",
                "created_at": "2024-01-01T02:00:00+00:00",
                "stance_label": 1,
                "state": 0,
                "hop": 2,
            },
        ],
        "state": {
            "1-hop": {"state_0": 1, "state_1": 1},
            "2-hop": {"state_0": 1, "state_1": 0},
        },
        "centrality": {"PageRank": [0.1, 0.2, 0.3, 0.4]},
    }


class TimestampParsingTests(unittest.TestCase):
    def test_parses_drweibo_two_digit_year_datetime(self):
        first = parse_time_seconds("13-7-14 23:28")
        second = parse_time_seconds("13-7-15 00:28")
        self.assertEqual(second - first, 3600)

    def test_parses_weibo_datetime(self):
        first = parse_time_seconds("Mon Jan 01 08:00:00 +0800 2024")
        second = parse_time_seconds("Mon Jan 01 09:00:00 +0800 2024")
        self.assertEqual(second - first, 3600)

    def test_parses_unix_milliseconds_in_auto_mode(self):
        self.assertEqual(parse_time_seconds(1_700_000_000_000), 1_700_000_000)


class TruncatePostTests(unittest.TestCase):
    def test_filters_and_reindexes_comments(self):
        result, stats = truncate_post(_post(), 3)
        comments = result["comment"]

        self.assertEqual([item["comment id"] for item in comments], [0, 1])
        self.assertEqual([item["parent"] for item in comments], [-1, 0])
        self.assertEqual(
            [item["early_detection_original_comment_id"] for item in comments],
            [1, 2],
        )
        self.assertEqual(comments[1]["hop"], 2)
        self.assertEqual(comments[1]["state"], 0)
        self.assertEqual(
            result["state"],
            {
                "1-hop": {"state_0": 0, "state_1": 1},
                "2-hop": {"state_0": 1, "state_1": 0},
            },
        )
        self.assertNotIn("centrality", result)
        self.assertEqual(stats["retained_comment_count"], 2)

    def test_drops_child_whose_late_parent_was_removed(self):
        post = _post()
        post["comment"][2]["parent"] = 0
        result, stats = truncate_post(post, 3)
        self.assertEqual(len(result["comment"]), 1)
        self.assertEqual(stats["dropped_orphan_count"], 1)

    def test_zero_hours_retains_source_only(self):
        result, _ = truncate_post(_post(), 0)
        self.assertEqual(result["comment"], [])
        self.assertEqual(result["state"], {})

    def test_builds_one_raw_directory_per_cutoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "test" / "raw"
            input_dir.mkdir(parents=True)
            with (input_dir / "event-1.json").open("w", encoding="utf-8") as file_obj:
                json.dump(_post(), file_obj)

            report = build_cutoff_datasets(
                input_dir, root / "early", cutoffs=[0, 3]
            )

            self.assertTrue((root / "early" / "0h" / "raw" / "event-1.json").is_file())
            self.assertTrue((root / "early" / "3h" / "raw" / "event-1.json").is_file())
            self.assertEqual(report["cutoffs"]["0h"]["events"], 1)
            self.assertEqual(report["cutoffs"]["3h"]["retained_comments"], 2)


if __name__ == "__main__":
    unittest.main()
