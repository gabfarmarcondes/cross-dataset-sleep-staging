"""Figures and tables for the data sanity check (issue 8), ready for the thesis.

Run from the repository root (or step by step in notebooks/08_data_sanity_check.ipynb):

    python -m src.data.figures

Outputs in outputs/figures/ (PNG at 300 dpi; text inside the figures is in Portuguese, for the thesis):
    class_distribution.png/.csv/.tex   epochs per stage and dataset
    example_epochs.png/.csv            one typical epoch of each stage per dataset
    psd_per_stage.png                  median power spectrum per stage, filter off and on
    diagnostics/<recording_id>.png     whole-night overview of single recordings (not for the thesis)
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from src.data.dataset import SleepRecordings, recording_paths
from src.data.format import FS, N_CLASSES, STAGES
from src.read_data import ROOT

DATASETS = ["sleep_edf", "isruc", "hmc"]
NAMES = {"sleep_edf": "Sleep-EDF", "isruc": "ISRUC", "hmc": "HMC"}
# Categorical slots 1-3 of the validated palette (colorblind-safe for any pair of the three);
# a dataset keeps its color in every figure
COLORS = {"sleep_edf": "#2a78d6", "isruc": "#eb6834", "hmc": "#1baf7a"}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

OUT = ROOT / "outputs" / "figures"
CM = 1 / 2.54
WIDTH = 16 * CM                # text width of an A4 page with 3 cm and 2 cm margins
PSD_WINDOW = 10 * FS           # 10-s Welch windows -> 0.1 Hz resolution
TYPICAL_BAND = (0.5, 30.0)     # band used to choose the most typical epoch
HYPNOGRAM_ORDER = [0, 4, 1, 2, 3]  # W, REM, N1, N2, N3 from top to bottom


def thesis_style():
    """Small, quiet styling for figures printed at text width."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save_figure(fig, name, folder=OUT):
    """Save as PNG at 300 dpi (print resolution at text width)."""
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png")


# ----------------------------------------------------------------------------- statistics

def collect_dataset(dataset):
    """One pass over a dataset: class counts, per-epoch spectra (filter off and on), epoch locations."""
    paths = recording_paths(dataset)
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    spectra = {False: [], True: []}
    labels, where = [], []
    for i, path in enumerate(paths):
        for filtered in (False, True):
            rec = SleepRecordings([path], filter=filtered)[0]
            freqs, power = welch(rec["x"], fs=FS, nperseg=PSD_WINDOW, axis=-1)
            spectra[filtered].append(power.astype(np.float32))
        counts += np.bincount(rec["y"], minlength=N_CLASSES)
        labels.append(rec["y"])
        where.append(np.column_stack([np.full(len(rec["y"]), i), np.arange(len(rec["y"]))]))
    return {
        "dataset": dataset, "paths": paths, "freqs": freqs, "counts": counts,
        "spectra": {k: np.concatenate(v) for k, v in spectra.items()},
        "labels": np.concatenate(labels), "where": np.concatenate(where),
    }


def median_spectra(stats, filtered):
    """Median power spectrum (dB) of each stage present in the dataset."""
    power = stats["spectra"][filtered]
    return {k: 10 * np.log10(np.median(power[stats["labels"] == k], axis=0))
            for k in range(N_CLASSES) if np.any(stats["labels"] == k)}


def typical_epochs(stats):
    """For each stage, the epoch whose spectrum (filter on, 0.5-30 Hz) is closest to the stage median."""
    band = (stats["freqs"] >= TYPICAL_BAND[0]) & (stats["freqs"] <= TYPICAL_BAND[1])
    log_power = np.log10(stats["spectra"][True][:, band])
    chosen = {}
    for k in range(N_CLASSES):
        idx = np.flatnonzero(stats["labels"] == k)
        if idx.size == 0:
            continue
        distance = np.mean((log_power[idx] - np.median(log_power[idx], axis=0)) ** 2, axis=1)
        rec_i, pos = stats["where"][idx[np.argmin(distance)]]
        rec = SleepRecordings([stats["paths"][rec_i]], filter=True)[0]
        chosen[k] = {"x": rec["x"][pos], "recording_id": rec["recording_id"],
                     "epoch_idx": int(rec["epoch_idx"][pos])}
    return chosen


# ----------------------------------------------------------------------------- tables

def pt_number(n):
    return f"{n:,}".replace(",", ".")


def pt_percent(p):
    return f"{p:.1f}".replace(".", ",")


def write_class_tables(all_stats, folder=OUT):
    """class_distribution.csv (counts and %) and class_distribution.tex (tabular for the thesis)."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "class_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "recordings", "epochs"] + list(STAGES) + [f"{s}_pct" for s in STAGES])
        for s in all_stats:
            c = s["counts"]
            w.writerow([s["dataset"], len(s["paths"]), c.sum()] + c.tolist()
                       + [round(100 * v / c.sum(), 2) for v in c])
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\hline",
             r"Base & Registros & Épocas & W (\%) & N1 (\%) & N2 (\%) & N3 (\%) & REM (\%) \\", r"\hline"]
    for s in all_stats:
        c = s["counts"]
        pct = " & ".join(pt_percent(100 * v / c.sum()) for v in c)
        lines.append(f"{NAMES[s['dataset']]} & {len(s['paths'])} & {pt_number(int(c.sum()))} & {pct} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (folder / "class_distribution.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_example_table(examples, folder=OUT):
    """example_epochs.csv: which recording and epoch each example comes from (for the figure caption)."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "example_epochs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "stage", "recording_id", "epoch_idx"])
        for d, ex in examples.items():
            for k, e in ex.items():
                w.writerow([d, STAGES[k], e["recording_id"], e["epoch_idx"]])


# ----------------------------------------------------------------------------- figures

def plot_class_distribution(all_stats):
    """Grouped bars: share of epochs per stage, one bar per dataset."""
    fig, ax = plt.subplots(figsize=(WIDTH, 5.5 * CM))
    width = 0.26
    for d, s in enumerate(all_stats):
        share = 100 * s["counts"] / s["counts"].sum()
        x = np.arange(N_CLASSES) + (d - 1) * width
        ax.bar(x, share, width, color=COLORS[s["dataset"]], edgecolor="white", linewidth=0.8,
               label=f"{NAMES[s['dataset']]} ({len(s['paths'])} registros)", zorder=2)
    ax.set_xticks(np.arange(N_CLASSES), STAGES)
    ax.set_ylabel("Épocas (%)")
    ax.yaxis.grid(True, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.set_ylim(0, None)
    fig.tight_layout()
    return fig


def plot_example_epochs(examples):
    """Grid stage x dataset with the typical epoch of each (filter on, robust-scaled units)."""
    datasets = [d for d in DATASETS if d in examples]
    lim = max(np.abs(e["x"]).max() for ex in examples.values() for e in ex.values())
    lim = np.ceil(lim)
    t = np.arange(3000) / FS
    fig, axes = plt.subplots(N_CLASSES, len(datasets), figsize=(WIDTH, 15 * CM),
                             sharex=True, sharey=True, squeeze=False)
    for col, d in enumerate(datasets):
        axes[0, col].set_title(NAMES[d], fontweight="bold")
        for k in range(N_CLASSES):
            ax = axes[k, col]
            ax.axhline(0, color=GRID, linewidth=0.5, zorder=0)
            if k in examples[d]:
                e = examples[d][k]
                ax.plot(t, e["x"], color=INK, linewidth=0.35)
                ax.text(1.0, 1.0, f"{e['recording_id']} · época {e['epoch_idx']}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6, color=MUTED)
            else:
                ax.text(0.5, 0.5, "sem épocas", transform=ax.transAxes, ha="center", va="center", color=MUTED)
            if col == 0:
                ax.set_ylabel(STAGES[k], rotation=0, ha="right", va="center", fontweight="bold")
    axes[0, 0].set_ylim(-lim, 1.4 * lim)  # headroom at the top for the identifier
    half = lim // 2
    axes[0, 0].set_yticks([-half, 0, half])  # -(lim // 2), not (-lim) // 2, which rounds down to -4
    axes[0, 0].set_xlim(0, 30)
    axes[-1, len(datasets) // 2].set_xlabel("Tempo (s)")
    fig.supylabel("Amplitude normalizada (IQR)", fontsize=8)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    return fig


def plot_psd_per_stage(all_stats):
    """Median spectrum per stage (columns) and condition (rows: filter off, filter on); one line per dataset."""
    fig, axes = plt.subplots(2, N_CLASSES, figsize=(WIDTH, 8.5 * CM), sharex=True, sharey=True)
    rows = [(False, "Sem filtro"), (True, "Com filtro (0,3–35 Hz)")]
    for r, (filtered, row_name) in enumerate(rows):
        for s in all_stats:
            freqs = s["freqs"]
            keep = freqs > 0
            for k, spectrum in median_spectra(s, filtered).items():
                axes[r, k].plot(freqs[keep], spectrum[keep], color=COLORS[s["dataset"]], linewidth=1.0,
                                label=NAMES[s["dataset"]] if (r, k) == (0, 0) else None)
        axes[r, 0].set_ylabel(f"{row_name}\nPotência (dB)")
        for k in range(N_CLASSES):
            ax = axes[r, k]
            for f in (0.3, 35):
                ax.axvline(f, color=AXIS, linewidth=0.5, zorder=0)
            ax.set_xscale("log")
            ax.yaxis.grid(True, zorder=0)
            if r == 0:
                ax.set_title(STAGES[k], fontweight="bold")
    axes[0, 0].set_xlim(0.1, 50)
    unfiltered = np.concatenate([np.concatenate(list(median_spectra(s, False).values())) for s in all_stats])
    axes[0, 0].set_ylim(unfiltered.min() - 5, unfiltered.max() + 5)
    for ax in axes[1]:
        ax.set_xticks([0.3, 1, 3, 10, 35], ["0,3", "1", "3", "10", "35"])
    axes[1, N_CLASSES // 2].set_xlabel("Frequência (Hz)")
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    return fig


def plot_recording_overview(dataset, recording_id):
    """Whole-night overview of one recording: hypnogram, amplitude and out-of-band power per epoch."""
    path = next(p for p in recording_paths(dataset) if p.stem == recording_id)
    with np.load(path, allow_pickle=False) as f:
        x, y, epoch_idx = f["x"], f["y"], f["epoch_idx"]
    hours = epoch_idx * 30 / 3600
    freqs, power = welch(x, fs=FS, nperseg=PSD_WINDOW, axis=-1)
    total = power.sum(axis=1)
    low = 100 * power[:, freqs < 0.3].sum(axis=1) / total
    high = 100 * power[:, freqs > 35].sum(axis=1) / total
    level = {stage: i for i, stage in enumerate(HYPNOGRAM_ORDER)}

    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 11 * CM), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1]})
    axes[0].step(hours, [level[v] for v in y], where="post", color=INK, linewidth=0.7)
    axes[0].set_yticks(range(5), [STAGES[k] for k in HYPNOGRAM_ORDER])
    axes[0].set_ylim(4.5, -0.5)
    axes[0].set_title(f"{NAMES[dataset]} · {recording_id}", loc="left", fontweight="bold")
    axes[1].plot(hours, np.percentile(np.abs(x), 99, axis=1), color=INK, linewidth=0.6)
    axes[1].set_ylim(0, None)
    axes[1].set_ylabel("p99 de |x| (µV)")
    axes[2].plot(hours, low, color=INK, linewidth=0.6, label="< 0,3 Hz")
    axes[2].plot(hours, high, color="#898781", linewidth=0.6, label="> 35 Hz")
    axes[2].set_ylabel("Potência (%)")
    axes[2].set_ylim(0, 100)
    axes[2].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    axes[2].set_xlabel("Tempo desde o início (h)")
    for ax in axes:
        ax.yaxis.grid(True, zorder=0)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- main

FLAGGED = [  # recordings singled out during issues 5-7, plus the first recording of each dataset for comparison
    ("hmc", "SN018"), ("hmc", "SN002"),
    ("sleep_edf", "SC4111"), ("sleep_edf", "SC4001"),
    ("isruc", "isruc_6"), ("isruc", "isruc_13"), ("isruc", "isruc_2"),
]


def main():
    thesis_style()
    all_stats = []
    for d in DATASETS:
        print(f"reading {d} ...")
        all_stats.append(collect_dataset(d))

    write_class_tables(all_stats)
    save_figure(plot_class_distribution(all_stats), "class_distribution")

    examples = {s["dataset"]: typical_epochs(s) for s in all_stats}
    write_example_table(examples)
    save_figure(plot_example_epochs(examples), "example_epochs")
    save_figure(plot_psd_per_stage(all_stats), "psd_per_stage")

    for d, rid in FLAGGED:
        if not any(p.stem == rid for p in recording_paths(d)):
            print(f"skipping {rid}: not found")
            continue
        fig = plot_recording_overview(d, rid)
        save_figure(fig, rid, OUT / "diagnostics")
        plt.close(fig)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()