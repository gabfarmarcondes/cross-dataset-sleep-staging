"""Compare training runs side by side (official vs. standard batch norm). From the
repository root:

    python scripts/compare_runs.py outputs/runs/sleep_edf_fold0_sequence_seed0 \\
                                   outputs/runs/sleep_edf_fold0_sequence_seed0_bnstandard

Prints, for each run, the numbers that matter for choosing a setting:
    best validation macro-F1 and its epoch
    mean and standard deviation of the validation macro-F1 over the last 10 epochs
    oscillation: mean change of the validation macro-F1 from one epoch to the next, last 10
    gap: training minus validation macro-F1, averaged over the last 10 epochs
    seconds per epoch
and saves a figure with the training and validation macro-F1 per epoch, one panel per
run, in outputs/figures/compare_<run 1>__<run 2>.png (text in Portuguese).
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.figures import CM, INK, MUTED, WIDTH, save_figure, thesis_style  # noqa: E402

LAST = 10  # epochs at the end used for the "stable" numbers
BN_NAMES = {"official": "BN oficial (escala e deslocamento)", "standard": "BN padrão (estatísticas do lote)"}


def load_run(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "history.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    history = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return run_dir.name, history, info


def summary(history):
    valid, train = history["valid_macro_f1"], history["train_macro_f1"]
    last = slice(-LAST, None)
    best = int(np.argmax(valid))
    return {
        "epochs": len(valid),
        "best valid macro-F1": f"{valid[best]:.1%} (epoch {best + 1})",
        f"valid macro-F1, last {LAST}": f"{valid[last].mean():.1%} ± {valid[last].std():.1%}",
        f"oscillation, last {LAST}": f"{np.abs(np.diff(valid[-LAST - 1:])).mean():.1%}",
        f"gap train - valid, last {LAST}": f"{(train[last] - valid[last]).mean():+.1%}",
        "seconds per epoch": f"{history['seconds'].mean():.0f}",
    }


def title(name, info):
    batch_norm = info.get("model", {}).get("batch_norm", "standard")  # older runs: standard
    return BN_NAMES.get(batch_norm, name)


def plot(runs):
    thesis_style()
    fig, axes = plt.subplots(1, len(runs), figsize=(WIDTH, 5.5 * CM), sharey=True, squeeze=False)
    for ax, (name, history, info) in zip(axes[0], runs):
        epochs = history["epoch"]
        ax.plot(epochs, history["train_macro_f1"], color=MUTED, linewidth=0.9, linestyle="--",
                label="treino")
        ax.plot(epochs, history["valid_macro_f1"], color=INK, linewidth=1.1, label="validação")
        best = int(np.argmax(history["valid_macro_f1"]))
        ax.plot(epochs[best], history["valid_macro_f1"][best], "o", color=INK, markersize=4,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.set_title(title(name, info))
        ax.set_xlabel("época de treino")
        ax.grid(axis="y")
    axes[0, 0].set_ylabel("macro-F1")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}".replace(".", ","))
    axes[0, -1].legend(loc="lower right")
    fig.tight_layout()
    return fig


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    runs = [load_run(path) for path in sys.argv[1:]]
    rows = {name: summary(history) for name, history, _ in runs}
    keys = list(next(iter(rows.values())))
    width = max(len(k) for k in keys)
    for name, _, info in runs:
        print(f"{name}: {title(name, info)}")
    for key in keys:
        print(f"  {key:<{width}}  " + "  |  ".join(f"{rows[name][key]:>20}" for name, _, _ in runs))
    fig = plot(runs)
    out = "compare_" + "__".join(name for name, _, _ in runs)
    save_figure(fig, out)
    print(f"figure: outputs/figures/{out}.png")


if __name__ == "__main__":
    main()