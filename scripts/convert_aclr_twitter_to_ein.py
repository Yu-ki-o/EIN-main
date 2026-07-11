"""Convert ACLR4RUMOR Twitter TSV files into EIN source JSON files.

Expected input files:
  dataset/ACLR4RUMOR-NAACL2022/data/Twitter/Twitter_data_all.txt
  dataset/ACLR4RUMOR-NAACL2022/data/Twitter/Twitter_label_all.txt

Output:
  dataset/Twitter/source/<root_id>.json

This only creates EIN-compatible text-tree source files. It does not add
LLM stance labels or derived EIN state/hop statistics.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict, deque
from pathlib import Path


DEFAULT_DATA = Path(
    "dataset/ACLR4RUMOR-NAACL2022/data/Twitter/Twitter_data_all.txt"
)
DEFAULT_LABELS = Path(
    "dataset/ACLR4RUMOR-NAACL2022/data/Twitter/Twitter_label_all.txt"
)
DEFAULT_OUTPUT = Path("dataset/Twitter/source")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ACLR4RUMOR Twitter data to EIN source JSON."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing output directory before writing JSON files.",
    )
    parser.add_argument(
        "--include_unlabeled",
        action="store_true",
        help="Also write threads without graph labels. These are not trainable by EIN.",
    )
    return parser.parse_args()


def read_labels(path: Path) -> dict[str, int]:
    labels = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed label line {line_number} in {path}: {line!r}"
                )
            root_id, label = parts
            labels[str(root_id)] = int(label)
    return labels


def read_threads(path: Path) -> OrderedDict[str, list[dict[str, str]]]:
    threads: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 4)
            if len(parts) != 5:
                raise ValueError(
                    f"Malformed data line {line_number} in {path}: {line!r}"
                )
            root_id, parent_index, current_index, placeholder, text = parts
            threads.setdefault(str(root_id), []).append(
                {
                    "parent_index": parent_index,
                    "current_index": current_index,
                    "placeholder": placeholder,
                    "content": text,
                }
            )
    return threads


def parse_index(value: str) -> int:
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Expected numeric post index, got {value!r}") from exc


def ordered_comment_rows(
    root_id: str,
    root_index: int,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_index = {
        parse_index(row["current_index"]): row
        for row in rows
        if row["parent_index"].strip().lower() != "none"
    }
    children = {index: [] for index in by_index}
    children[root_index] = []
    for index, row in by_index.items():
        parent_index = parse_index(row["parent_index"])
        if parent_index != root_index and parent_index not in by_index:
            raise ValueError(
                f"Thread {root_id} has missing parent index {parent_index} "
                f"for current index {index}"
            )
        children.setdefault(parent_index, []).append(index)

    for child_list in children.values():
        child_list.sort()

    ordered = []
    queue = deque(children.get(root_index, []))
    while queue:
        index = queue.popleft()
        ordered.append(by_index[index])
        queue.extend(children.get(index, []))

    if len(ordered) != len(by_index):
        raise ValueError(
            f"Thread {root_id} is not a connected tree: "
            f"visited {len(ordered)} of {len(by_index)} comments"
        )
    return ordered


def convert_thread(root_id: str, rows: list[dict[str, str]], label: int) -> dict:
    root_rows = [
        row
        for row in rows
        if row["parent_index"].strip().lower() == "none"
    ]
    if len(root_rows) != 1:
        raise ValueError(
            f"Thread {root_id} should contain exactly one root, got {len(root_rows)}"
        )

    root_row = root_rows[0]
    root_index = parse_index(root_row["current_index"])
    comment_rows = ordered_comment_rows(root_id, root_index, rows)

    index_to_comment_id = {}
    for comment_id, row in enumerate(comment_rows):
        index_to_comment_id[parse_index(row["current_index"])] = comment_id

    comments = []
    for comment_id, row in enumerate(comment_rows):
        parent_index = parse_index(row["parent_index"])
        parent = (
            -1
            if parent_index == root_index
            else index_to_comment_id[parent_index]
        )
        comments.append(
            {
                "comment id": comment_id,
                "parent": parent,
                "content": row["content"],
                "original_index": parse_index(row["current_index"]),
                "placeholder": row["placeholder"],
            }
        )

    return {
        "source": {
            "tweet id": str(root_id),
            "label": int(label),
            "content": root_row["content"],
            "original_index": root_index,
            "placeholder": root_row["placeholder"],
        },
        "comment": comments,
    }


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    if not args.labels.exists():
        raise FileNotFoundError(args.labels)

    if args.output.exists():
        if not args.force:
            raise FileExistsError(
                f"{args.output} already exists; pass --force to replace it."
            )
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    labels = read_labels(args.labels)
    threads = read_threads(args.data)
    missing_labels = sorted(set(threads) - set(labels))
    if missing_labels and not args.include_unlabeled:
        print(
            "Skipping {} unlabeled thread(s); first skipped root id: {}".format(
                len(missing_labels),
                missing_labels[0],
            )
        )

    total_comments = 0
    converted_threads = 0
    for root_id, rows in threads.items():
        if root_id not in labels and not args.include_unlabeled:
            continue
        label = labels.get(root_id)
        post = convert_thread(root_id, rows, label) if label is not None else None
        if post is None:
            root_rows = [
                row
                for row in rows
                if row["parent_index"].strip().lower() == "none"
            ]
            if len(root_rows) != 1:
                raise ValueError(
                    f"Thread {root_id} should contain exactly one root, got {len(root_rows)}"
                )
            post = convert_thread(root_id, rows, 0)
            del post["source"]["label"]
        total_comments += len(post["comment"])
        converted_threads += 1
        output_path = args.output / f"{root_id}.json"
        output_path.write_text(
            json.dumps(post, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

    print(
        "Converted {} threads and {} comments into {}".format(
            converted_threads,
            total_comments,
            args.output,
        )
    )


if __name__ == "__main__":
    main()
