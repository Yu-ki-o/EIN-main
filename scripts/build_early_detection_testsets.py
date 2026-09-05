"""Build timestamp-truncated test sets for early rumor detection.

The input directory must be an existing ``test/raw`` directory produced by
this project.  Every output cutoff has the same JSON schema and can therefore
be passed to ``TreeDataset`` or ``ResGCNTreeDataset`` directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from statistics import mean


DEFAULT_CUTOFFS = (0.0, 1.0, 3.0, 6.0, 12.0, 24.0)
DEFAULT_TIME_FIELDS = (
    "created_at",
    "created at",
    "create_time",
    "createtime",
    "publish_time",
    "post_time",
    "timestamp",
    "time_stamp",
    "datetime",
    "date",
    "time",
    "t",
)
WEIBO_TIME_FORMATS = (
    "%a %b %d %H:%M:%S %z %Y",
    "%y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
)


class TimestampError(ValueError):
    """Raised when a post does not provide a usable node timestamp."""


def _field_value(record, field):
    value = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(field)
        value = value[part]
    return value


def find_time_field(record, explicit_field=None):
    if explicit_field:
        try:
            _field_value(record, explicit_field)
        except KeyError as exc:
            raise TimestampError(
                "missing requested time field {!r}; available keys: {}".format(
                    explicit_field, sorted(record.keys())
                )
            ) from exc
        return explicit_field

    for candidate in DEFAULT_TIME_FIELDS:
        if candidate in record and record[candidate] not in (None, ""):
            return candidate
    raise TimestampError(
        "could not find a time field; available keys: {}. "
        "Pass --source-time-field/--comment-time-field explicitly.".format(
            sorted(record.keys())
        )
    )


def _numeric_seconds(value, unit):
    number = float(value)
    if not math.isfinite(number):
        raise TimestampError("timestamp must be finite, got {!r}".format(value))
    if unit == "seconds":
        return number
    if unit == "milliseconds":
        return number / 1_000.0
    if unit == "minutes":
        return number * 60.0
    if unit == "hours":
        return number * 3_600.0
    if unit != "auto":
        raise TimestampError("unsupported numeric time unit: {}".format(unit))

    magnitude = abs(number)
    if magnitude >= 1e14:
        return number / 1_000_000.0
    if magnitude >= 1e11:
        return number / 1_000.0
    return number


def parse_time_seconds(value, numeric_unit="auto", datetime_format=None):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _numeric_seconds(value, numeric_unit)

    if not isinstance(value, str):
        raise TimestampError(
            "unsupported timestamp type {} for {!r}".format(
                type(value).__name__, value
            )
        )
    text = value.strip()
    if not text:
        raise TimestampError("empty timestamp")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return _numeric_seconds(text, numeric_unit)

    if datetime_format:
        try:
            return datetime.strptime(text, datetime_format).timestamp()
        except ValueError as exc:
            raise TimestampError(
                "timestamp {!r} does not match --datetime-format {!r}".format(
                    text, datetime_format
                )
            ) from exc

    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(iso_text).timestamp()
    except ValueError:
        pass
    for time_format in WEIBO_TIME_FORMATS:
        try:
            return datetime.strptime(text, time_format).timestamp()
        except ValueError:
            continue
    raise TimestampError(
        "unsupported timestamp {!r}; pass --datetime-format if needed".format(text)
    )


def _relative_hours(
    post,
    source_time_field=None,
    comment_time_field=None,
    numeric_unit="auto",
    datetime_format=None,
    comment_times_relative=False,
):
    comments = post.get("comment", [])
    if not isinstance(comments, list):
        raise ValueError("post['comment'] must be a list")

    if comment_times_relative:
        root_seconds = 0.0
    else:
        source = post.get("source")
        if not isinstance(source, dict):
            raise ValueError("post['source'] must be an object")
        source_field = find_time_field(source, source_time_field)
        root_seconds = parse_time_seconds(
            _field_value(source, source_field), numeric_unit, datetime_format
        )

    values = {}
    detected_comment_field = comment_time_field
    for comment in comments:
        if detected_comment_field is None:
            detected_comment_field = find_time_field(comment)
        try:
            raw_time = _field_value(comment, detected_comment_field)
        except KeyError as exc:
            raise TimestampError(
                "comment id {!r} is missing time field {!r}".format(
                    comment.get("comment id"), detected_comment_field
                )
            ) from exc
        seconds = parse_time_seconds(raw_time, numeric_unit, datetime_format)
        delay = seconds if comment_times_relative else seconds - root_seconds
        values[int(comment["comment id"])] = delay / 3_600.0
    return values, detected_comment_field


def _drop_orphans(selected):
    selected_ids = {int(comment["comment id"]) for comment in selected}
    dropped = 0
    changed = True
    while changed:
        changed = False
        kept = []
        for comment in selected:
            parent = int(comment["parent"])
            if parent == -1 or parent in selected_ids:
                kept.append(comment)
                continue
            # Whether the parent was cut off or absent in the original file,
            # retaining the child would create an invalid early graph.
            selected_ids.discard(int(comment["comment id"]))
            dropped += 1
            changed = True
        selected = kept
    return selected, dropped


def _refresh_comment_metadata(comments):
    by_id = {int(comment["comment id"]): comment for comment in comments}
    unresolved = set(by_id)
    while unresolved:
        progressed = False
        for comment_id in list(unresolved):
            comment = by_id[comment_id]
            parent_id = int(comment["parent"])
            if parent_id != -1 and parent_id in unresolved:
                continue
            if parent_id == -1:
                comment["hop"] = 1
                parent_state = 0
            else:
                parent = by_id.get(parent_id)
                if parent is None:
                    raise ValueError(
                        "comment {} references missing parent {}".format(
                            comment_id, parent_id
                        )
                    )
                comment["hop"] = int(parent["hop"]) + 1
                parent_state = int(parent.get("state", 0))

            if "stance_label" in comment:
                stance = int(comment["stance_label"])
                if stance in (0, 1):
                    comment["state"] = parent_state ^ stance
            unresolved.remove(comment_id)
            progressed = True
        if not progressed:
            raise ValueError("cycle detected in comment parent links")


def _build_state_summary(comments):
    counts = {}
    for comment in comments:
        if "state" not in comment:
            continue
        state = int(comment["state"])
        if state not in (0, 1):
            continue
        hop = int(comment["hop"])
        hop_counts = counts.setdefault(hop, {"state_0": 0, "state_1": 0})
        hop_counts["state_{}".format(state)] += 1
    return {"{}-hop".format(hop): counts[hop] for hop in sorted(counts)}


def truncate_post(
    post,
    cutoff_hours,
    source_time_field=None,
    comment_time_field=None,
    numeric_unit="auto",
    datetime_format=None,
    comment_times_relative=False,
    negative_tolerance_seconds=1.0,
):
    """Return a cutoff copy of one post and truncation statistics."""
    if cutoff_hours < 0:
        raise ValueError("cutoff_hours must be non-negative")
    result = copy.deepcopy(post)
    comments = result.get("comment", [])
    relative_hours, detected_field = _relative_hours(
        result,
        source_time_field=source_time_field,
        comment_time_field=comment_time_field,
        numeric_unit=numeric_unit,
        datetime_format=datetime_format,
        comment_times_relative=comment_times_relative,
    )

    tolerance_hours = negative_tolerance_seconds / 3_600.0
    invalid_negative = [
        comment_id
        for comment_id, delay in relative_hours.items()
        if delay < -tolerance_hours
    ]
    if invalid_negative:
        raise TimestampError(
            "{} comment(s) precede the source timestamp, including ids {}".format(
                len(invalid_negative), invalid_negative[:5]
            )
        )

    selected = [
        comment
        for comment in comments
        if max(0.0, relative_hours[int(comment["comment id"])]) <= cutoff_hours
    ]
    selected, orphan_count = _drop_orphans(selected)

    old_to_new = {
        int(comment["comment id"]): new_id
        for new_id, comment in enumerate(selected)
    }
    for new_id, comment in enumerate(selected):
        old_id = int(comment["comment id"])
        old_parent = int(comment["parent"])
        comment["comment id"] = new_id
        comment["parent"] = -1 if old_parent == -1 else old_to_new[old_parent]
        comment["early_detection_original_comment_id"] = old_id
        comment["early_detection_delay_hours"] = max(
            0.0, relative_hours[old_id]
        )

    _refresh_comment_metadata(selected)
    result["comment"] = selected
    result["state"] = _build_state_summary(selected)
    # RAGCL may otherwise reuse centrality computed on the complete graph.
    result.pop("centrality", None)
    result["early_detection"] = {
        "cutoff_hours": float(cutoff_hours),
        "original_comment_count": len(comments),
        "retained_comment_count": len(selected),
        "dropped_orphan_count": orphan_count,
        "comment_time_field": detected_field,
        "comment_times_relative": bool(comment_times_relative),
    }
    return result, result["early_detection"]


def cutoff_name(cutoff):
    value = float(cutoff)
    if value.is_integer():
        return "{}h".format(int(value))
    return "{}h".format(str(value).replace(".", "p"))


def build_cutoff_datasets(
    input_dir,
    output_root,
    cutoffs=DEFAULT_CUTOFFS,
    source_time_field=None,
    comment_time_field=None,
    numeric_unit="auto",
    datetime_format=None,
    comment_times_relative=False,
    overwrite=False,
):
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError("no JSON files found in {}".format(input_dir))

    cutoff_values = [float(value) for value in cutoffs]
    if len(set(cutoff_values)) != len(cutoff_values):
        raise ValueError("cutoffs must be unique")
    output_dirs = {}
    for cutoff in cutoff_values:
        raw_dir = output_root / cutoff_name(cutoff) / "raw"
        cutoff_dir = raw_dir.parent
        if cutoff_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    "{} already exists; pass --overwrite to replace it".format(
                        cutoff_dir
                    )
                )
            shutil.rmtree(cutoff_dir)
        raw_dir.mkdir(parents=True)
        output_dirs[cutoff] = raw_dir

    summaries = {
        cutoff: {"events": 0, "original_comments": 0, "retained_comments": 0,
                 "dropped_orphans": 0, "retention_ratios": []}
        for cutoff in cutoff_values
    }
    for path in files:
        with path.open("r", encoding="utf-8") as file_obj:
            post = json.load(file_obj)
        for cutoff in cutoff_values:
            try:
                truncated, stats = truncate_post(
                    post,
                    cutoff,
                    source_time_field=source_time_field,
                    comment_time_field=comment_time_field,
                    numeric_unit=numeric_unit,
                    datetime_format=datetime_format,
                    comment_times_relative=comment_times_relative,
                )
            except Exception as exc:
                raise type(exc)("{}: {}".format(path, exc)) from exc
            destination = output_dirs[cutoff] / path.name
            with destination.open("w", encoding="utf-8") as file_obj:
                json.dump(truncated, file_obj, indent=2, ensure_ascii=False)

            summary = summaries[cutoff]
            original = stats["original_comment_count"]
            retained = stats["retained_comment_count"]
            summary["events"] += 1
            summary["original_comments"] += original
            summary["retained_comments"] += retained
            summary["dropped_orphans"] += stats["dropped_orphan_count"]
            summary["retention_ratios"].append(
                retained / original if original else 1.0
            )

    report = {"input_dir": str(input_dir.resolve()), "cutoffs": {}}
    for cutoff in cutoff_values:
        summary = summaries[cutoff]
        ratios = summary.pop("retention_ratios")
        summary["mean_event_retention_ratio"] = mean(ratios)
        report["cutoffs"][cutoff_name(cutoff)] = summary
    report_path = output_root / "summary.json"
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build timestamp-truncated copies of a test/raw directory."
    )
    parser.add_argument("--input-dir", required=True, help="existing full test/raw")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--cutoffs", nargs="+", type=float, default=list(DEFAULT_CUTOFFS)
    )
    parser.add_argument("--source-time-field")
    parser.add_argument("--comment-time-field")
    parser.add_argument(
        "--numeric-unit",
        choices=("auto", "seconds", "milliseconds", "minutes", "hours"),
        default="auto",
    )
    parser.add_argument(
        "--datetime-format",
        help="optional strptime format, e.g. '%%Y-%%m-%%d %%H:%%M:%%S'",
    )
    parser.add_argument(
        "--comment-times-relative",
        action="store_true",
        help="treat comment times as delays from the source instead of datetimes",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_cutoff_datasets(
        args.input_dir,
        args.output_root,
        cutoffs=args.cutoffs,
        source_time_field=args.source_time_field,
        comment_time_field=args.comment_time_field,
        numeric_unit=args.numeric_unit,
        datetime_format=args.datetime_format,
        comment_times_relative=args.comment_times_relative,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
