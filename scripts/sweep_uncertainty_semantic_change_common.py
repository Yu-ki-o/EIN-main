#!/usr/bin/env python3
"""Shared one-seed sweeper for UncertaintySemanticChange models."""

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
DEFAULT_FIXED_FUSION_MODE = "original_change_vertical"
SELECTION_METRICS = ("val_loss", "val_acc", "val_auc", "val_f1")

SWEEP_PARAM_ORDER = [
    "use_uncertainty_sampling",
    "relation_hidden_dim",
    "relation_temperature",
    "stance_route_temperature",
    "stance_route_hard",
    "lambda_edge_relation_aux",
    "lambda_edge_relation_warmup",
    "lambda_view_mi_aux",
    "use_ds_mass_routing",
    "ds_unknown_prior",
    "lambda_ds_unknown_edge_aux",
    "uncertainty_sample_temperature",
    "uncertainty_keep_floor",
    "uncertainty_sampling_warmup_epochs",
    "use_degree_importance",
    "degree_importance_strength",
    "semantic_change_hidden_dim",
    "use_node_keep_in_change_pool",
    "use_semantic_tree_transformer",
    "semantic_tree_transformer_layers",
    "semantic_tree_transformer_heads",
    "semantic_tree_transformer_ffn_dim",
    "semantic_tree_transformer_dropout",
    "semantic_tree_transformer_max_depth",
    "semantic_tree_depth_dim",
    "semantic_tree_transformer_pool",
    "vertical_path_attention_heads",
    "vertical_path_attention_uncertainty_scale",
    "vertical_path_attention_residual_gate",
    "vertical_path_attention_dropout",
    "classification_fusion_hidden_dim",
    "classification_class_weights",
    "use_global_ds_fusion",
    "global_ds_unknown_prior",
    "global_ds_temperature",
    "global_ds_fusion_rule",
    "global_ds_hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "patience",
    "n_layers_conv",
]

FIXED_TRACE_PARAMS = [
    "dataset",
    "base_model",
    "result_group",
    "lr",
    "weight_decay",
    "dropout",
    "batch_size",
    "hidden_dim",
    "n_layers_conv",
    "global_pool",
    "max_hop",
    "classification_fusion_mode",
    "classification_fusion_hidden_dim",
    "use_uncertainty_sampling",
    "use_semantic_tree_transformer",
    "use_vertical_path_attention",
]

BASE_PRESET_GRIDS = {
    "routing": {
        "uncertainty_keep_floor": [0.0, 0.05, 0.1],
        "degree_importance_strength": [0.5, 1.0, 2.0],
    },
    "sampling": {
        "uncertainty_sample_temperature": [0.3, 0.5, 0.8],
        "uncertainty_sampling_warmup_epochs": [0, 5, 10],
    },
    "relation": {
        "lambda_edge_relation_aux": [0.05, 0.1, 0.2],
        "lambda_edge_relation_warmup": [0.2, 0.5, 1.0],
    },
    "temperature": {
        "relation_temperature": [0.7, 1.0, 1.5],
        "stance_route_temperature": [0.3, 0.5, 0.8],
    },
    "vertical": {
        "vertical_path_attention_uncertainty_scale": [0.5, 1.0, 2.0],
        "vertical_path_attention_residual_gate": [0.3, 0.6, 0.9],
    },
    "change_pool": {
        "use_node_keep_in_change_pool": [False, True],
    },
    "semantic_tree": {
        "semantic_tree_transformer_layers": [1, 2],
        "semantic_tree_transformer_dropout": [0.2, 0.3, 0.4],
        "semantic_tree_transformer_max_depth": [40, 72],
        "lambda_view_mi_aux": [0.0, 0.01, 0.02],
    },
    "semantic_tree_capacity": {
        "semantic_tree_transformer_heads": [4, 8],
        "semantic_tree_transformer_ffn_dim": [256, 512],
        "semantic_tree_depth_dim": [64, 128],
        "semantic_tree_transformer_pool": ["mean", "root"],
    },
}

ALIASES = {
    "use_uncertainty_sampling": "usamp",
    "relation_hidden_dim": "rhid",
    "relation_temperature": "rtemp",
    "stance_route_temperature": "stemp",
    "stance_route_hard": "shard",
    "lambda_edge_relation_aux": "laux",
    "lambda_edge_relation_warmup": "lwarm",
    "lambda_view_mi_aux": "lmi",
    "use_ds_mass_routing": "ds",
    "ds_unknown_prior": "du",
    "lambda_ds_unknown_edge_aux": "ldsu",
    "uncertainty_sample_temperature": "samp",
    "uncertainty_keep_floor": "keep",
    "uncertainty_sampling_warmup_epochs": "warm",
    "use_degree_importance": "degimp",
    "degree_importance_strength": "deg",
    "semantic_change_hidden_dim": "chid",
    "use_node_keep_in_change_pool": "poolkeep",
    "use_semantic_tree_transformer": "stree",
    "semantic_tree_transformer_layers": "stlayers",
    "semantic_tree_transformer_heads": "stheads",
    "semantic_tree_transformer_ffn_dim": "stffn",
    "semantic_tree_transformer_dropout": "stdrop",
    "semantic_tree_transformer_max_depth": "stdepth",
    "semantic_tree_depth_dim": "stde",
    "semantic_tree_transformer_pool": "stpool",
    "vertical_path_attention_heads": "vheads",
    "vertical_path_attention_uncertainty_scale": "vunc",
    "vertical_path_attention_residual_gate": "vgate",
    "vertical_path_attention_dropout": "vdrop",
    "classification_fusion_hidden_dim": "fhid",
    "classification_class_weights": "cw",
    "use_global_ds_fusion": "gds",
    "global_ds_unknown_prior": "gdu",
    "global_ds_temperature": "gdt",
    "global_ds_fusion_rule": "gdrule",
    "global_ds_hidden_dim": "gdhid",
    "dropout": "drop",
    "lr": "lr",
    "weight_decay": "wd",
    "patience": "pat",
    "n_layers_conv": "layers",
}


def build_preset_grids(defaults):
    grids = deepcopy(BASE_PRESET_GRIDS)
    grids.update(deepcopy(defaults.get("extra_preset_grids", {})))
    return grids


def parse_args(defaults):
    preset_grids = build_preset_grids(defaults)
    fixed_fusion_mode = defaults.get(
        "fixed_fusion_mode",
        DEFAULT_FIXED_FUSION_MODE,
    )
    parser = argparse.ArgumentParser(
        description=(
            "Sweep UncertaintySemanticChange hyperparameters on {} with "
            "one seed. classification_fusion_mode is fixed to {}."
        ).format(defaults["expected_dataset"], fixed_fusion_mode)
    )
    parser.add_argument(
        "--config",
        default=str(defaults["config"]),
        help="Base YAML config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(defaults.get("default_seed", 4)),
        help="Single seed to run.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(preset_grids),
        default=defaults.get("default_preset", "routing"),
        help="Sweep preset.",
    )
    parser.add_argument(
        "--grid-json",
        default=None,
        help=(
            "Optional JSON file mapping sweep parameter names to value lists. "
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
        default=str(defaults["config_out_dir"]),
        help="Directory for generated trial configs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(defaults["output_dir"]),
        help="Directory for aggregate CSV/JSONL results.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First trial index to run.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum trials to run.",
    )
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
        "--device",
        default=None,
        help="Override config device, e.g. cuda:0, cuda:1, 0, 1, or cpu.",
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


def normalize_device_arg(device):
    device = str(device).strip()
    if not device:
        return device
    lowered = device.lower()
    if lowered == "cpu":
        return "cpu"
    if lowered == "cuda":
        return "cuda"
    if lowered.isdigit():
        return "cuda:{}".format(lowered)
    if lowered.startswith("gpu") and lowered[3:].isdigit():
        return "cuda:{}".format(lowered[3:])
    if lowered.startswith("cuda"):
        if lowered[4:].isdigit():
            return "cuda:{}".format(lowered[4:])
        return lowered
    return device


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
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
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


def load_grid(args, defaults):
    preset_grids = build_preset_grids(defaults)
    if args.grid_json is None:
        return deepcopy(preset_grids[args.preset])

    with open(args.grid_json, "r", encoding="utf-8") as file_obj:
        grid = json.load(file_obj)

    blocked = sorted(set(grid) - set(SWEEP_PARAM_ORDER))
    if blocked:
        raise ValueError(
            "grid-json contains unsupported or fixed parameters: {}".format(
                ", ".join(blocked)
            )
        )
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                "grid-json value for {} must be a non-empty list".format(key)
            )
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
    for key in sorted(overrides):
        compact_parts.append(
            "{}{}".format(ALIASES.get(key, key), safe_name_part(overrides[key]))
        )
    readable = "_".join(compact_parts)
    if len(readable) > 96:
        readable = trial_hash(overrides)
    return "{}_t{:04d}_{}".format(sweep_name, trial_index, readable)


def result_name_to_dir(result_name):
    return str(result_name).strip().replace("/", "_").replace("\\", "_")


def result_group_to_path(result_group):
    if result_group is None:
        return None
    parts = []
    for part in re.split(r"[\\/]+", str(result_group)):
        safe_part = safe_name_part(part)
        if safe_part:
            parts.append(safe_part)
    if not parts:
        return None
    return Path(*parts)


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


def summary_path_for(config, include_group=True):
    result_name = result_name_to_dir(config["result_name"])
    selection_metric = selection_metric_for(config)
    summary_dir = (
        REPO_ROOT
        / "experiments"
        / config["model_name"]
        / config["dataset"]
    )
    if include_group:
        result_group = result_group_to_path(config.get("result_group"))
        if result_group is not None:
            summary_dir = summary_dir / result_group
    return summary_dir / result_name / "summary_{}.txt".format(
        safe_name_part(selection_metric)
    )


def summary_candidate_paths_for(config):
    paths = []
    for include_group in (True, False):
        summary_path = summary_path_for(config, include_group=include_group)
        if summary_path not in paths:
            paths.append(summary_path)
        if selection_metric_for(config) == "val_loss":
            legacy_path = summary_path.parent / "summary.txt"
            if legacy_path not in paths:
                paths.append(legacy_path)
        if config.get("result_group") is None:
            break
    return paths


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
    for key in SWEEP_PARAM_ORDER:
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


def validate_base_config(config, defaults):
    if config.get("dataset") != defaults["expected_dataset"]:
        raise ValueError(
            "Expected dataset {}, got {}".format(
                defaults["expected_dataset"], config.get("dataset")
            )
        )
    if config.get("base_model") != defaults["expected_base_model"]:
        raise ValueError(
            "Expected base_model {}, got {}".format(
                defaults["expected_base_model"], config.get("base_model")
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


def run_cli(defaults):
    args = parse_args(defaults)
    base_config = load_yaml(args.config)
    validate_base_config(base_config, defaults)

    fixed_fusion_mode = defaults.get(
        "fixed_fusion_mode",
        DEFAULT_FIXED_FUSION_MODE,
    )
    base_config["classification_fusion_mode"] = fixed_fusion_mode
    base_config["result_group"] = defaults.get("result_group", defaults["sweep_name_prefix"])
    fixed_selection_metric = defaults.get("fixed_selection_metric")
    if fixed_selection_metric is not None:
        base_config["selection_metric"] = selection_metric_for(
            {"selection_metric": fixed_selection_metric}
        )
        if (
            args.selection_metric is not None
            and args.selection_metric != base_config["selection_metric"]
        ):
            print(
                "Ignoring --selection-metric {}; fixed to {} by this script.".format(
                    args.selection_metric,
                    base_config["selection_metric"],
                )
            )
    elif args.selection_metric is not None:
        base_config["selection_metric"] = args.selection_metric
    else:
        base_config["selection_metric"] = selection_metric_for(base_config)
    if args.device is not None:
        base_config["device"] = normalize_device_arg(args.device)

    grid = load_grid(args, defaults)
    sweep_name = args.sweep_name
    if sweep_name is None:
        sweep_name = "{}_{}_seed{}".format(
            defaults["sweep_name_prefix"],
            args.preset if args.grid_json is None else "custom",
            args.seed,
        )
        if base_config["selection_metric"] != "val_loss":
            sweep_name = "{}_{}".format(
                sweep_name,
                safe_name_part(base_config["selection_metric"]),
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
        + SWEEP_PARAM_ORDER
        + ["overrides"]
    )

    planned = list(iter_trials(grid))
    end_index = len(planned)
    if args.max_trials is not None:
        end_index = min(end_index, args.start_index + args.max_trials)
    selected = list(enumerate(planned))[args.start_index:end_index]

    print("Base config: {}".format(Path(args.config).resolve()))
    print("Dataset/base_model: {}/{}".format(base_config["dataset"], base_config["base_model"]))
    print("Preset/grid keys: {}".format(", ".join(grid.keys())))
    print("Result group: {}".format(base_config["result_group"]))
    print("Fixed fusion mode: {}".format(base_config["classification_fusion_mode"]))
    print("Seed per trial: {}".format(args.seed))
    print("Checkpoint selection metric: {}".format(base_config["selection_metric"]))
    if args.device is not None:
        print("Device override: {}".format(base_config["device"]))
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
        trial_config["classification_fusion_mode"] = fixed_fusion_mode
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
                        trial_index,
                        result_name,
                        parsed_path,
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
