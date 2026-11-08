"""Confusion matrices of a cross-dataset report in one figure (issue 16), ready for the thesis.

Run from the repository root, after src.evaluation.report:

    python -m src.evaluation.confusion_figure outputs/results/experiment_b.json

Uses the groups of the report that fit a source x target grid: one in_domain_ref group per
dataset (the diagonal) and one cross group per source -> target pair. Each panel is the
confusion matrix summed over the seeds and normalized by row: the share (%) of the epochs of
each true stage that went to each predicted stage. Grid rows are the training dataset, grid
columns the test dataset. Also prints the macro-F1 of every entry (seed) of every group.

Output: outputs/figures/<report name>_confusion.png (300 dpi, text width; text in Portuguese).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.data.figures import INK, MUTED, NAMES, OUT, WIDTH, save_figure, thesis_style
from src.data.format import STAGES

DATASETS = ["sleep_edf", "isruc", "hmc"]
# one-hue sequential ramp (blue, the reference palette), white = 0%
RAMP = LinearSegmentedColormap.from_list(
    "share", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
WHITE_TEXT_ABOVE = 55  # % of the row: from here on the cell is dark enough for white text


def pt(x, decimals=1):
    """Number with a decimal comma, as in the thesis tables."""
    return f"{x:.{decimals}f}".replace(".", ",")


def panel_key(group):
    """(training dataset, test dataset) of a group, or None if it is not part of the grid."""
    if group["scenario"] == "in_domain_ref":
        return group["dataset"], group["dataset"]
    if group["scenario"] == "cross":
        return group["dataset"], group["target"]
    return None


def row_shares(group):
    """Confusion matrix summed over the entries, as % of each true stage (rows)."""
    counts = group.get("confusion_sum")
    if counts is None:
        counts = np.sum([entry["confusion"] for entry in group["entries"]], axis=0)
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    return 100 * np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def mean_macro_f1(group):
    """Mean macro-F1 (%) over the entries of a group, as in the report tables."""
    return 100 * float(np.mean([entry["values"]["macro_f1"] for entry in group["entries"]]))


def print_entries(report):
    """Macro-F1 of every entry of every group, to see how much the seeds differ."""
    for group in report["groups"]:
        values = [f"{entry['name']} {pt(100 * entry['values']['macro_f1'])}" for entry in group["entries"]]
        print(f"{group['label']:<24} " + " | ".join(values))


def plot(report):
    """3 x 3 grid of row-normalized confusion matrices; missing pairs are left blank."""
    panels = {}
    for group in report["groups"]:
        key = panel_key(group)
        if key is not None:
            panels[key] = group
    if not panels:
        raise SystemExit("the report has no in_domain_ref or cross groups")

    thesis_style()
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, WIDTH * 1.08), constrained_layout=True)
    ticks = np.arange(len(STAGES))
    image = None
    for r, source in enumerate(DATASETS):
        for c, target in enumerate(DATASETS):
            ax = axes[r, c]
            group = panels.get((source, target))
            if group is None:
                ax.axis("off")
                continue
            share = row_shares(group)
            image = ax.imshow(share, cmap=RAMP, vmin=0, vmax=100)
            for i in range(len(STAGES)):
                for j in range(len(STAGES)):
                    ax.text(j, i, f"{share[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                            color="white" if share[i, j] >= WHITE_TEXT_ABOVE else INK)
            pair = (f"{NAMES[source]} (dentro da base)" if source == target
                    else f"{NAMES[source]} → {NAMES[target]}")
            ax.set_title(f"{pair}\nmacro-F1 {pt(mean_macro_f1(group))}%", fontsize=7)
            ax.set_xticks(ticks, STAGES if r == len(DATASETS) - 1 else [])
            ax.set_yticks(ticks, STAGES if c == 0 else [])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
    for c in range(len(DATASETS)):
        axes[-1, c].set_xlabel("Estágio previsto")
    for r in range(len(DATASETS)):
        axes[r, 0].set_ylabel("Estágio real")
    if image is not None:
        bar = fig.colorbar(image, ax=axes, shrink=0.5, aspect=30, pad=0.02)
        bar.set_label("% das épocas do estágio real", color=MUTED)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Confusion matrices of a cross-dataset report.")
    parser.add_argument("report", type=Path, help="JSON written by src.evaluation.report")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print_entries(report)
    fig = plot(report)
    name = f"{report['name']}_confusion"
    save_figure(fig, name)
    print(f"written: {OUT / (name + '.png')}")


if __name__ == "__main__":
    main()