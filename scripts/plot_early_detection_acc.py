#!/usr/bin/env python3
"""Plot early rumor detection accuracy from the Markdown summary.

The SEE series is always taken from the ``SEE without TTT Accuracy`` table,
while all other models are read from the main ``Test Accuracy`` table.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        # Embed TrueType fonts so text stays sharp in LaTeX PDF output.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    }
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "logs" / "early_detection_acc_summary.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "early_detection_figures"
DATASETS = ("DRWeibo", "PHEME")

COLORS = {
    "BiGCN": "#4C78A8",
    "ResGCN": "#72B7B2",
    "RAGCL": "#F58518",
    "EIN": "#54A24B",
    "SEE": "#B279A2",
    "Our Model": "#E45756",
}
MARKERS = {
    "BiGCN": "o",
    "ResGCN": "v",
    "RAGCL": "s",
    "EIN": "^",
    "SEE": "D",
    "Our Model": "*",
}
DISPLAY_NAMES = {"Our Model": "Ours","SEE": "GARD"}
PANEL_LABELS = {
    "DRWeibo": "(a) DRWeibo",
    "PHEME": "(b) PHEME",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot model accuracy at different detection deadlines."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Markdown accuracy summary (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures (default: %(default)s)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="Output formats (default: png pdf)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to plot (default: DRWeibo PHEME)",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution")
    parser.add_argument(
        "--figure-width",
        type=float,
        default=3.35,
        help="Source figure width in inches (default: 3.35)",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=2.35,
        help="Source figure height in inches (default: 2.35)",
    )
    return parser.parse_args()


def split_markdown_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells):
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_summary(path):
    """Extract the main and SEE-without-TTT tables for both datasets."""
    tables = {dataset: {} for dataset in DATASETS}
    current_dataset = None
    current_table = None
    headers = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if line.startswith("## ") and not line.startswith("### "):
            heading = line[3:].strip()
            current_dataset = heading if heading in tables else None
            current_table = None
            headers = None
            continue

        if line.startswith("### "):
            heading = line[4:].strip().lower()
            if current_dataset and heading.startswith("test accuracy"):
                current_table = "main"
            elif current_dataset and heading.startswith("see without ttt accuracy"):
                current_table = "see_without_ttt"
            else:
                current_table = None
            headers = None
            continue

        if not current_dataset or not current_table or not line.startswith("|"):
            continue

        cells = split_markdown_row(line)
        if is_separator_row(cells):
            continue
        if headers is None:
            headers = cells
            tables[current_dataset][current_table] = {
                "headers": headers,
                "rows": [],
            }
            continue
        if len(cells) != len(headers):
            raise ValueError(
                f"Malformed table row in {path}: expected {len(headers)} cells, "
                f"got {len(cells)}: {raw_line}"
            )
        tables[current_dataset][current_table]["rows"].append(cells)

    for dataset in DATASETS:
        missing = {"main", "see_without_ttt"} - tables[dataset].keys()
        if missing:
            raise ValueError(f"Missing {dataset} table(s): {', '.join(sorted(missing))}")
    return tables


def normalized_model_name(name):
    if name.startswith("SEE"):
        return "SEE"
    return name


def build_plot_data(dataset_tables):
    """Replace the main SEE/TTT column with the SEE/without-TTT values."""
    main = dataset_tables["main"]
    see_table = dataset_tables["see_without_ttt"]
    main_deadlines = [row[0] for row in main["rows"]]
    see_deadlines = [row[0] for row in see_table["rows"]]

    series = {}
    for column, raw_name in enumerate(main["headers"][1:], start=1):
        model = normalized_model_name(raw_name)
        if model == "SEE":
            series[model] = {
                "deadlines": see_deadlines,
                "accuracy": [float(row[1]) for row in see_table["rows"]],
            }
        else:
            series[model] = {
                "deadlines": main_deadlines,
                "accuracy": [float(row[column]) for row in main["rows"]],
            }
    return series


def numeric_deadline(deadline):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([hm])", deadline)
    if not match:
        raise ValueError(f"Unsupported detection deadline: {deadline}")
    value = float(match.group(1))
    return value * 60 if match.group(2) == "h" else value


def plot_dataset(
    dataset, series, output_dir, formats, dpi, figure_width, figure_height
):
    if dataset == "PHEME":
        series = {model: data for model, data in series.items() if model != "ResGCN"}

    # Use the union because SEE and the main table may evaluate different
    # deadlines. Evaluated checkpoints are equally spaced for legibility.
    deadlines = sorted(
        {
            deadline
            for model_data in series.values()
            for deadline in model_data["deadlines"]
        },
        key=numeric_deadline,
    )
    deadline_positions = {deadline: index for index, deadline in enumerate(deadlines)}
    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height),
        facecolor="white",
    )
    ax.set_facecolor("white")

    for model, model_data in series.items():
        is_ours = model == "Our Model"
        x_values = [
            deadline_positions[deadline] for deadline in model_data["deadlines"]
        ]
        ax.plot(
            x_values,
            model_data["accuracy"],
            label=DISPLAY_NAMES.get(model, model),
            color=COLORS.get(model),
            marker=MARKERS.get(model, "o"),
            markersize=9 if is_ours else 7,
            linewidth=3.0 if is_ours else 2.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )

    all_values = [
        value for model_data in series.values() for value in model_data["accuracy"]
    ]
    padding = max(1.0, (max(all_values) - min(all_values)) * 0.10)
    ax.set_ylim(min(all_values) - padding, max(all_values) + padding)
    deadline_numbers = [
        re.fullmatch(r"(\d+(?:\.\d+)?)[hm]", deadline).group(1)
        for deadline in deadlines
    ]
    deadline_unit = "hours" if dataset == "DRWeibo" else "mins"
    ax.set_xticks(range(len(deadlines)), deadline_numbers)
    ax.set_xlabel(
        f"Detection Deadline ({deadline_unit})",
        fontsize=14,
        labelpad=3,
    )
    if dataset == "DRWeibo":
        ax.set_ylabel("Accuracy (%)", fontsize=14, labelpad=3)
    ax.set_title(PANEL_LABELS[dataset], fontsize=12, fontweight="semibold", pad=3)
    ax.tick_params(axis="both", labelsize=12, width=1.1, length=4)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)
    legend_options = {
        "frameon": False,
        "ncol": 3,
        "fontsize": 11.5,
        "handlelength": 1.25,
        "handletextpad": 0.35,
        "columnspacing": 0.6,
        "labelspacing": 0.3,
        "borderaxespad": 0.25,
    }
    if dataset == "PHEME":
        extra_artists = ()
    else:
        legend = ax.legend(loc="best", **legend_options)
        extra_artists = ()
    fig.tight_layout(pad=0.35)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset.lower()}_early_detection_acc"
    outputs = []
    for output_format in formats:
        output_path = output_dir / f"{stem}.{output_format}"
        save_options = {"dpi": dpi} if output_format == "png" else {}
        fig.savefig(
            output_path,
            bbox_inches="tight",
            bbox_extra_artists=extra_artists,
            pad_inches=0.02,
            facecolor="white",
            transparent=False,
            **save_options,
        )
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def main():
    args = parse_args()
    tables = parse_summary(args.input)
    generated = []
    for dataset in args.datasets:
        series = build_plot_data(tables[dataset])
        generated.extend(
            plot_dataset(
                dataset,
                series,
                args.output_dir,
                args.formats,
                args.dpi,
                args.figure_width,
                args.figure_height,
            )
        )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
