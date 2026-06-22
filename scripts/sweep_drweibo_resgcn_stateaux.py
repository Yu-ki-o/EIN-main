#!/usr/bin/env python3
"""One-seed hyperparameter sweep for DRWeibo ResGCN StateAuxSameDiff.

This script intentionally keeps the base optimizer/model hyperparameters from
the source config, such as lr, weight_decay, and dropout. The sweep presets only
touch parameters introduced by StateAuxSameDiff.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/EIN/DRWeibo_ResGCN_StateAuxSameDiff_word2vec.yaml"
DEFAULT_CONFIG_OUT_DIR = REPO_ROOT / "configs/sweeps/drweibo_resgcn_stateaux"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/EIN/DRWeibo/sweep_drweibo_resgcn_stateaux"
DEFAULT_SWEEP_NAME_PREFIX = "sweep_drweibo_resgcn_stateaux"
EXPECTED_DATASET = "DRWeibo"
EXPECTED_BASE_MODEL = "ResGCN_StateAuxSameDiff"

INTRODUCED_PARAM_ORDER = [
    "state_aux_temp",
    "state_aux_detach_route",
    "lambda_node_state_aux",
    "lambda_edge_same_aux",
    "lambda_same_diff_sep",
    "same_path_blend",
    "diff_path_blend",
    "same_diff_gate_blend",
    "same_dulreg_blend",
]

FIXED_TRACE_PARAMS = [
    "lr",
    "weight_decay",
    "dropout",
    "batch_size",
    "hidden_dim",
    "n_layers_conv",
]

SELECTION_METRICS = ("val_loss", "val_acc", "val_auc", "val_f1")

PRESET_GRIDS = {
    "compact": {
        "lambda_node_state_aux": [0.05, 0.1, 0.2],
        "lambda_edge_same_aux": [0.05, 0.1, 0.2],
        "lambda_same_diff_sep": [0.0, 0.005, 0.01, 0.02],
    },
    "blend": {
        "same_path_blend": [0.5, 1.0],
        "diff_path_blend": [0.5, 1.0],
        "same_diff_gate_blend": [0.0, 0.5, 1.0],
        "same_dulreg_blend": [0.0, 0.5, 1.0],
    },
    "route": {
        "state_aux_temp": [0.7, 1.0, 1.5],
        "state_aux_detach_route": [False, True],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep StateAuxSameDiff-only hyperparameters on {} with a "
            "single seed by default."
        ).format(EXPECTED_DATASET)
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Base YAML config.")
    parser.add_argument("--seed", type=int, default=0, help="Single seed to run.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_GRIDS),
        default="compact",
        help="StateAuxSameDiff-only sweep preset.",
    )
    parser.add_argument(
        "--grid-json",
        default=None,
        help=(
            "Optional JSON file mapping introduced parameter names to value lists. "
            "When set, it overrides --preset."
        ),
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Prefix used in result_name and output files.",
    )
    parser.add_argument(
        "--config-out-dir",
        default=str(DEFAULT_CONFIG_OUT_DIR),
        help="Directory for generated trial configs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for aggregate CSV/JSONL results.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="First trial index to run.")
    parser.add_argument("--max-trials", type=int, default=None, help="Maximum trials to run.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed trials with an existing summary or CSV row.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned trials without writing configs or training.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default=None,
        help=(
            "Validation metric used to select the checkpoint for test. "
            "Defaults to the config value, or val_loss when absent."
        ),
    )
    return parser.parse_args()


def load_yaml(path):
    if yaml is None:
        return load_flat_yaml(path)
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.load(file_obj, Loader=yaml.FullLoader)


def parse_flat_yaml_value(value):
    value = value.strip()
    if value == "" or value.lower() in {"null", "none"}:
        return None
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    if value == "[]":
        return []
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_flat_yaml(path):
    config = {}
    with open(path, "r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.split("#", 1)[0].strip()
            config[key.strip()] = parse_flat_yaml_value(value)
    return config


def dump_config(config, file_obj):
    if yaml is not None:
        yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)
        return

    for key, value in config.items():
        if value is None:
            text = "null"
        elif isinstance(value, bool):
            text = "True" if value else "False"
        elif isinstance(value, list):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        file_obj.write("{}: {}\n".format(key, text))


def load_grid(args):
    if args.grid_json is None:
        return deepcopy(PRESET_GRIDS[args.preset])

    with open(args.grid_json, "r", encoding="utf-8") as file_obj:
        grid = json.load(file_obj)

    unknown = sorted(set(grid) - set(INTRODUCED_PARAM_ORDER))
    if unknown:
        raise ValueError(
            "grid-json contains non-StateAuxSameDiff parameters: {}".format(
                ", ".join(unknown)
            )
        )
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError("grid-json value for {} must be a non-empty list".format(key))
    return grid


def iter_trials(grid):
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def safe_name_part(value):
    text = str(value)
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or "x"


def trial_hash(overrides):
    payload = json.dumps(overrides, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def build_trial_name(sweep_name, trial_index, overrides):
    compact_parts = []
    aliases = {
        "lambda_node_state_aux": "lnode",
        "lambda_edge_same_aux": "ledge",
        "lambda_same_diff_sep": "lsep",
        "same_path_blend": "same",
        "diff_path_blend": "diff",
        "same_diff_gate_blend": "gate",
        "same_dulreg_blend": "dulreg",
        "state_aux_temp": "temp",
        "state_aux_detach_route": "detach",
    }
    for key in sorted(overrides):
        compact_parts.append("{}{}".format(aliases.get(key, key), safe_name_part(overrides[key])))
    readable = "_".join(compact_parts)
    if len(readable) > 96:
        readable = trial_hash(overrides)
    return "{}_t{:04d}_{}".format(sweep_name, trial_index, readable)


def result_name_to_dir(result_name):
    return str(result_name).strip().replace("/", "_").replace("\\", "_")


def selection_metric_for(config):
    metric = config.get("selection_metric", "val_loss")
    if metric is None:
        metric = "val_loss"
    metric = str(metric).strip()
    if metric not in SELECTION_METRICS:
        raise ValueError(
            "selection_metric must be one of {}, got {}".format(
                ", ".join(SELECTION_METRICS), metric
            )
        )
    return metric


def summary_path_for(config):
    result_name = result_name_to_dir(config["result_name"])
    selection_metric = selection_metric_for(config)
    return (
        REPO_ROOT
        / "experiments"
        / config["model_name"]
        / config["dataset"]
        / result_name
        / "summary_{}.txt".format(safe_name_part(selection_metric))
    )


def summary_candidate_paths_for(config):
    summary_path = summary_path_for(config)
    if selection_metric_for(config) != "val_loss":
        return [summary_path]
    legacy_path = summary_path.parent / "summary.txt"
    if legacy_path == summary_path:
        return [summary_path]
    return [summary_path, legacy_path]


def parse_summary(summary_path):
    if not summary_path.exists():
        return None
    text = summary_path.read_text(encoding="utf-8")
    match = re.search(
        r"Seed\s+(?P<seed>\d+):\s+Acc\s+(?P<acc>[0-9.]+)\s+\|\s+"
        r"AUC\s+(?P<auc>[0-9.]+)\s+\|\s+F1\s+(?P<f1>[0-9.]+)",
        text,
    )
    if match is None:
        return None
    return {
        "seed": int(match.group("seed")),
        "acc": float(match.group("acc")),
        "auc": float(match.group("auc")),
        "f1": float(match.group("f1")),
    }


def read_existing_trial_keys(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        keys = set()
        for row in reader:
            result_name = row.get("result_name")
            if not result_name:
                continue
            selection_metric = row.get("selection_metric") or "val_loss"
            keys.add((result_name, selection_metric))
        return keys


def append_result(csv_path, jsonl_path, row, fieldnames):
    csv_exists = csv_path.exists()
    if csv_exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []
        if existing_fieldnames != fieldnames:
            merged_rows = []
            for existing_row in existing_rows:
                merged_row = {field: existing_row.get(field) for field in fieldnames}
                if not merged_row.get("selection_metric"):
                    merged_row["selection_metric"] = "val_loss"
                merged_rows.append(merged_row)
            with open(csv_path, "w", encoding="utf-8", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(merged_rows)

    with open(csv_path, "a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)

    with open(jsonl_path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def make_row(config, trial_index, result_name, overrides, result):
    row = {
        "trial_index": trial_index,
        "result_name": result_name,
        "seed": config["seed"],
        "selection_metric": selection_metric_for(config),
        "acc": result.get("acc"),
        "auc": result.get("auc"),
        "f1": result.get("f1"),
    }
    for key in FIXED_TRACE_PARAMS:
        row[key] = config.get(key)
    for key in INTRODUCED_PARAM_ORDER:
        row[key] = config.get(key)
    row["overrides"] = json.dumps(overrides, ensure_ascii=False, sort_keys=True)
    return row


def summarize_results(results, config):
    lines = [
        "Experiment setting:",
        "Target dataset: {}".format(config["dataset"]),
        "Mode: {}".format(config.get("experiment_mode", "id")),
        "OOD source datasets: {}".format(config.get("ood_source_datasets", [])),
        "Validation domain: {}".format(config.get("ood_val_domain", "source")),
        "Checkpoint selection metric: {}".format(selection_metric_for(config)),
        "",
        "Seed results:",
    ]
    for result in results:
        lines.append(
            "Seed {seed}: Acc {acc:.4f} | AUC {auc:.4f} | F1 {f1:.4f}".format(
                **result
            )
        )

    lines.extend(["", "Average results over {} runs:".format(len(results))])
    for metric in ["acc", "auc", "f1"]:
        values = [float(result[metric]) for result in results]
        mean = sum(values) / len(values)
        lines.append("{}: {:.2f}+/-{:.2f} (%)".format(metric.upper(), mean * 100, 0.0))

    summary = "\n".join(lines)
    print(summary)

    summary_path = summary_path_for(config)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print("Summary saved to: {}".format(summary_path))


def validate_base_config(config):
    if config.get("dataset") != EXPECTED_DATASET:
        raise ValueError(
            "Expected dataset {}, got {}".format(EXPECTED_DATASET, config.get("dataset"))
        )
    if config.get("base_model") != EXPECTED_BASE_MODEL:
        raise ValueError(
            "Expected base_model {}, got {}".format(
                EXPECTED_BASE_MODEL, config.get("base_model")
            )
        )


def run_trial(config):
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import argparse as argparse_module
    import torch
    import supervisor as supervisor_module

    args = argparse_module.Namespace(**config)
    supervisor_name = "EIN_{}_supervisor".format(args.base_model)
    supervisor_fn = getattr(supervisor_module, supervisor_name)
    result = supervisor_fn(args)
    if result is not None:
        result["seed"] = args.seed
        summarize_results([result], config)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    base_config = load_yaml(args.config)
    validate_base_config(base_config)
    if args.selection_metric is not None:
        base_config["selection_metric"] = args.selection_metric
    else:
        base_config["selection_metric"] = selection_metric_for(base_config)

    grid = load_grid(args)
    if args.sweep_name:
        sweep_name = args.sweep_name
    else:
        sweep_name = "{}_{}_seed{}".format(
            DEFAULT_SWEEP_NAME_PREFIX,
            args.preset if args.grid_json is None else "custom",
            args.seed,
        )
        if base_config["selection_metric"] != "val_loss":
            sweep_name = "{}_{}".format(
                sweep_name, safe_name_part(base_config["selection_metric"])
            )
    output_dir = Path(args.output_dir)
    config_out_dir = Path(args.config_out_dir)
    csv_path = output_dir / "results.csv"
    jsonl_path = output_dir / "results.jsonl"
    fieldnames = (
        [
            "trial_index",
            "result_name",
            "seed",
            "selection_metric",
            "acc",
            "auc",
            "f1",
        ]
        + FIXED_TRACE_PARAMS
        + INTRODUCED_PARAM_ORDER
        + ["overrides"]
    )

    planned = list(iter_trials(grid))
    end_index = len(planned)
    if args.max_trials is not None:
        end_index = min(end_index, args.start_index + args.max_trials)
    selected = list(enumerate(planned))[args.start_index:end_index]

    print("Base config: {}".format(Path(args.config).resolve()))
    print("Preset/grid keys: {}".format(", ".join(grid.keys())))
    print("Keeping fixed: {}".format(", ".join(FIXED_TRACE_PARAMS)))
    print("Seed per trial: {}".format(args.seed))
    print("Checkpoint selection metric: {}".format(base_config["selection_metric"]))
    print("Selected trials: {} / {}".format(len(selected), len(planned)))

    if args.dry_run:
        for trial_index, overrides in selected:
            trial_name = build_trial_name(sweep_name, trial_index, overrides)
            print("[dry-run] {:04d} {} {}".format(trial_index, trial_name, overrides))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config_out_dir.mkdir(parents=True, exist_ok=True)
    completed = read_existing_trial_keys(csv_path) if args.resume else set()

    for trial_index, overrides in selected:
        trial_config = deepcopy(base_config)
        trial_config.update(overrides)
        trial_config["seed"] = args.seed
        result_name = build_trial_name(sweep_name, trial_index, overrides)
        trial_config["result_name"] = result_name

        generated_config = config_out_dir / "{}.yaml".format(result_name)

        if args.resume and (result_name, selection_metric_for(trial_config)) in completed:
            print("[skip-csv] {:04d} {}".format(trial_index, result_name))
            continue

        if args.resume:
            parsed = None
            parsed_path = None
            for summary_path in summary_candidate_paths_for(trial_config):
                parsed = parse_summary(summary_path)
                if parsed is not None:
                    parsed_path = summary_path
                    break
            if parsed is not None:
                print(
                    "[skip-summary] {:04d} {} summary={}".format(
                        trial_index, result_name, parsed_path
                    )
                )
                row = make_row(trial_config, trial_index, result_name, overrides, parsed)
                append_result(csv_path, jsonl_path, row, fieldnames)
                continue

        with open(generated_config, "w", encoding="utf-8") as file_obj:
            dump_config(trial_config, file_obj)

        print("[run] {:04d} {} config={}".format(trial_index, result_name, generated_config))
        result = run_trial(trial_config)
        if result is None:
            print("[warn] trial returned no result: {}".format(result_name))
            continue

        row = make_row(trial_config, trial_index, result_name, overrides, result)
        append_result(csv_path, jsonl_path, row, fieldnames)
        print(
            "[done] {:04d} acc={:.4f} auc={:.4f} f1={:.4f}".format(
                trial_index,
                result["acc"],
                result["auc"],
                result["f1"],
            )
        )

    print("Aggregate CSV: {}".format(csv_path))
    print("Aggregate JSONL: {}".format(jsonl_path))


if __name__ == "__main__":
    main()
