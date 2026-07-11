"""Prepare the converted Twitter dataset for EIN training.

This script is intended to run on the GPU server after
``dataset/Twitter/source/*.json`` has been copied into the project. It can also
rebuild those JSON files from the ACLR4RUMOR TSV files when they are present.

Pipeline:
  1. Ensure ``dataset/<dataset>/source`` exists.
  2. Optionally annotate missing ``stance_label`` values with an English LLM.
  3. Derive per-comment ``state`` / ``hop`` and graph-level hop statistics.
  4. Write a Twitter EIN config ready for ``main.py``.

The LLM step is resumable: existing ``stance_label`` values are skipped unless
``--overwrite_stance`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_aclr_twitter_to_ein import (  # noqa: E402
    convert_thread,
    read_labels,
    read_threads,
)


DEFAULT_ACLR_DATA = Path(
    "dataset/ACLR4RUMOR-NAACL2022/data/Twitter/Twitter_data_all.txt"
)
DEFAULT_ACLR_LABELS = Path(
    "dataset/ACLR4RUMOR-NAACL2022/data/Twitter/Twitter_label_all.txt"
)
DEFAULT_TEMPLATE = Path("configs/EIN/Pheme_BiGCN_UncertaintySemanticChange.yaml")
DEFAULT_CONFIG_OUTPUT = Path(
    "configs/EIN/Twitter_BiGCN_UncertaintySemanticChange_word2vec.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate and preprocess Twitter source JSON for EIN."
    )
    parser.add_argument("--dataset", default="Twitter")
    parser.add_argument("--source_dir", type=Path, default=None)
    parser.add_argument("--convert_from_aclr", action="store_true")
    parser.add_argument("--force_convert", action="store_true")
    parser.add_argument("--aclr_data", type=Path, default=DEFAULT_ACLR_DATA)
    parser.add_argument("--aclr_labels", type=Path, default=DEFAULT_ACLR_LABELS)

    parser.add_argument("--skip_stance", action="store_true")
    parser.add_argument("--overwrite_stance", action="store_true")
    parser.add_argument("--model_name", default="google/gemma-2-9b-it")
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--torch_dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument(
        "--limit_files",
        type=int,
        default=None,
        help="Debug option: process only the first N JSON files.",
    )

    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--config_template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--config_output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    parser.add_argument("--tokenize_mode", default="naive")
    parser.add_argument("--result_name", default="Twitter_UncertaintySemanticChange_bigcn_word2vec")
    return parser.parse_args()


def dataset_root(dataset: str) -> Path:
    return Path("dataset") / dataset


def default_source_dir(dataset: str) -> Path:
    return dataset_root(dataset) / "source"


def list_source_files(source_dir: Path, limit_files: int | None = None) -> list[Path]:
    files = sorted(source_dir.glob("*.json"))
    if limit_files is not None:
        files = files[: max(0, int(limit_files))]
    return files


def ensure_dataset_dirs(dataset: str) -> None:
    root = dataset_root(dataset)
    for child in ("source", "splits", "dataset", "dataset_cache"):
        (root / child).mkdir(parents=True, exist_ok=True)


def convert_from_aclr(args: argparse.Namespace, source_dir: Path) -> None:
    existing = list(source_dir.glob("*.json"))
    if existing and not args.force_convert:
        print(
            f"Using existing source JSON files in {source_dir}; "
            "pass --force_convert to rebuild them."
        )
        return

    if not args.aclr_data.exists():
        raise FileNotFoundError(args.aclr_data)
    if not args.aclr_labels.exists():
        raise FileNotFoundError(args.aclr_labels)

    source_dir.mkdir(parents=True, exist_ok=True)
    if args.force_convert:
        for path in source_dir.glob("*.json"):
            path.unlink()

    labels = read_labels(args.aclr_labels)
    threads = read_threads(args.aclr_data)
    missing = sorted(set(threads) - set(labels))
    if missing:
        print(
            "Skipping {} unlabeled ACLR thread(s); first skipped root id: {}".format(
                len(missing),
                missing[0],
            )
        )

    converted = 0
    comments = 0
    for root_id, rows in threads.items():
        if root_id not in labels:
            continue
        post = convert_thread(root_id, rows, labels[root_id])
        comments += len(post["comment"])
        converted += 1
        (source_dir / f"{root_id}.json").write_text(
            json.dumps(post, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
    print(f"Converted {converted} threads and {comments} comments into {source_dir}")


def clean_sentence(text: str) -> str:
    return re.sub(r"http\S+", "", str(text))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")


def torch_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def load_llm(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub.errors import GatedRepoError

    token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=args.device_map,
            torch_dtype=torch_dtype(args.torch_dtype),
            token=token,
        )
    except (GatedRepoError, OSError) as exc:
        raise RuntimeError(
            "\nCannot load the stance LLM. If you use google/gemma-2-9b-it, "
            "first accept the model terms on Hugging Face and run:\n"
            "  export HF_TOKEN=your_token\n"
            "  huggingface-cli login --token \"$HF_TOKEN\"\n"
            "Or pass a non-gated model with --model_name.\n"
        ) from exc

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def stance_prompt(source_sentence: str, response_sentence: str, parent: int) -> str:
    if parent == -1:
        return (
            f"Source post: '{source_sentence}'\n"
            f"Response comment: '{response_sentence}'\n"
            "Based on the content of the response comment, determine its attitude "
            "towards the source post and choose one option: "
            "The response comment believes the source post: 0; "
            "The response comment does not believe or doubts the source post: 1.\n"
            "If the response comment only mentions someone with @ and has no other "
            "content, consider it believing the source post.\n"
            "Return only one digit, 0 or 1."
        )
    return (
        f"Source sentence: '{source_sentence}'\n"
        f"Response sentence: '{response_sentence}'\n"
        "Based on the content of the response sentence, determine its attitude "
        "towards the source sentence and choose one option: "
        "The response sentence agrees with the source sentence: 0; "
        "The response sentence disagrees with or doubts the source sentence: 1.\n"
        "If the response sentence only mentions someone with @ and has no other "
        "content, consider it agreeing with the source sentence.\n"
        "Return only one digit, 0 or 1."
    )


@torch.no_grad()
def predict_stance(model, tokenizer, prompt: str, max_new_tokens: int) -> int:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    match = re.search(r"\b[01]\b", decoded)
    if match:
        return int(match.group(0))
    digits = re.findall(r"[01]", decoded)
    if digits:
        return int(digits[-1])
    raise ValueError(f"Could not parse stance label from model output: {decoded!r}")


def annotate_stance(args: argparse.Namespace, source_files: list[Path]) -> None:
    model, tokenizer = load_llm(args)
    total_done = 0
    total_skipped = 0
    for path in tqdm(source_files, desc="Annotating stance"):
        data = load_json(path)
        comments = data.get("comment", [])
        comment_map = {comment["comment id"]: comment for comment in comments}
        changed = False
        file_done = 0
        for comment in comments:
            if "stance_label" in comment and not args.overwrite_stance:
                total_skipped += 1
                continue
            parent_id = int(comment["parent"])
            if parent_id == -1:
                source_sentence = data["source"]["content"]
            else:
                source_sentence = comment_map[parent_id]["content"]
            reply_sentence = comment["content"]
            prompt = stance_prompt(
                clean_sentence(source_sentence),
                clean_sentence(reply_sentence),
                parent_id,
            )
            comment["stance_label"] = predict_stance(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
            )
            changed = True
            total_done += 1
            file_done += 1
            if args.save_every > 0 and file_done % args.save_every == 0:
                save_json(path, data)
        if changed:
            save_json(path, data)
    print(f"Stance annotation finished: new={total_done}, skipped={total_skipped}")


def preprocess_one(data: dict, path: Path) -> int:
    comments = data.get("comment", [])
    comment_map = {int(comment["comment id"]): comment for comment in comments}
    unresolved = set(comment_map)
    max_hop = 0

    while unresolved:
        progressed = False
        for comment_id in sorted(list(unresolved)):
            comment = comment_map[comment_id]
            if "stance_label" not in comment:
                raise ValueError(f"{path} comment {comment_id} is missing stance_label")
            stance = int(comment["stance_label"])
            if stance not in (0, 1):
                raise ValueError(
                    f"{path} comment {comment_id} has invalid stance_label {stance}"
                )

            parent_id = int(comment["parent"])
            if parent_id == -1:
                parent_state = 0
                parent_hop = 0
            else:
                parent = comment_map[parent_id]
                if "state" not in parent or "hop" not in parent:
                    continue
                parent_state = int(parent["state"])
                parent_hop = int(parent["hop"])

            comment["state"] = parent_state if stance == 0 else 1 - parent_state
            comment["hop"] = parent_hop + 1
            max_hop = max(max_hop, int(comment["hop"]))
            unresolved.remove(comment_id)
            progressed = True

        if not progressed:
            raise ValueError(f"{path} has unresolved parent/state dependencies")

    hop_counts = {}
    for comment in comments:
        hop = int(comment["hop"])
        state = int(comment["state"])
        hop_counts.setdefault(hop, {"state_0": 0, "state_1": 0})
        hop_counts[hop][f"state_{state}"] += 1
    data["state"] = {
        f"{hop}-hop": hop_counts[hop]
        for hop in sorted(hop_counts)
    }
    return max_hop


def preprocess_source(source_files: list[Path]) -> int:
    max_hop = 0
    for path in tqdm(source_files, desc="Building state/hop"):
        data = load_json(path)
        max_hop = max(max_hop, preprocess_one(data, path))
        save_json(path, data)
    print(f"State/hop preprocessing finished. max_hop={max_hop}")
    return max_hop


def max_depth_from_tree(source_files: list[Path]) -> int:
    max_hop = 0
    for path in source_files:
        data = load_json(path)
        depths = {-1: 0}
        for comment in data.get("comment", []):
            parent = int(comment["parent"])
            depths[int(comment["comment id"])] = depths[parent] + 1
            max_hop = max(max_hop, depths[int(comment["comment id"])])
    return max_hop


def write_config(args: argparse.Namespace, max_hop: int) -> None:
    if not args.config_template.exists():
        raise FileNotFoundError(args.config_template)
    config = yaml.safe_load(args.config_template.read_text(encoding="utf-8"))
    config["dataset"] = args.dataset
    config["language"] = "en"
    config["word_embedding"] = "word2vec"
    config["tokenize_mode"] = args.tokenize_mode
    config["result_name"] = args.result_name
    config["max_hop"] = int(max_hop)
    config["vertical_path_attention_max_distance"] = int(max_hop)
    config["semantic_tree_transformer_max_depth"] = int(max_hop)
    config["classification_class_weights"] = [1.0, 1.0]
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Wrote config: {args.config_output}")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir or default_source_dir(args.dataset)
    ensure_dataset_dirs(args.dataset)

    if args.convert_from_aclr or not list(source_dir.glob("*.json")):
        convert_from_aclr(args, source_dir)

    source_files = list_source_files(source_dir, args.limit_files)
    if not source_files:
        raise FileNotFoundError(f"No JSON files found in {source_dir}")
    print(f"Found {len(source_files)} source JSON files in {source_dir}")

    if not args.skip_stance:
        annotate_stance(args, source_files)
    else:
        print("Skipping stance annotation.")

    if not args.skip_preprocess:
        max_hop = preprocess_source(source_files)
    else:
        print("Skipping state/hop preprocessing.")
        max_hop = max_depth_from_tree(source_files)
        print(f"Tree max_hop from parent links: {max_hop}")

    write_config(args, max_hop)
    print("Done. Next training command:")
    print(f"  python main.py --config_filename {args.config_output} --device cuda:0")


if __name__ == "__main__":
    main()
