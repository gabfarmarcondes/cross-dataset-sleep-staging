"""Regenerate every table and figure of the thesis Experiments chapter (issue 18).

Run from the repository root, on the machine where the experiments ran, after all the runs
of experiments A, B and C are finished (see the README, "Reproducing the thesis results"):

    python -m src.evaluation.export                       # everything
    python -m src.evaluation.export --dry-run             # only print the commands
    python -m src.evaluation.export --skip-data-figures   # skip the slow figures of the data

It runs, in this order, one process per step:
    src.evaluation.report            experiment_a, experiment_a_runs, experiment_b, experiment_c
    src.evaluation.paired            experiment_a_pares, experiment_c_dentro, experiment_c_entre
    src.evaluation.confusion_figure  experiment_b_confusion.png
    src.data.figures                 class_distribution.png, example_epochs.png, psd_per_stage.png
Runs that are already evaluated are not evaluated again, so nothing is trained or tested anew.
Then it writes outputs/results/environment.tex (hardware and library versions of this machine)
and outputs/results/thesis_manifest.md, which links each table and figure of the chapter to
the file it comes from and says what was formatted by hand in the thesis.

Group labels are the row and column titles of the generated tables, in Portuguese, as in
the thesis. The labels used by --pair must match the labels of the report it reads.
"""
import argparse
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

from src.read_data import ROOT

RESULTS = Path("outputs") / "results"
FIGURES = Path("outputs") / "figures"

# report name -> groups (LABEL=DATASET:SCENARIO[:TARGET][@OPTIONS], see src/evaluation/report.py)
REPORTS = {
    # experiment A: pooled over the 5 folds, as in the TinySleepNet paper
    "experiment_a": [
        "CNN=sleep_edf:in_domain_pooled@epoch",
        "CNN + LSTM=sleep_edf:in_domain_pooled@sequence",
        "CNN + LSTM, sem aumento=sleep_edf:in_domain_pooled@sequence,noaug",
    ],
    # experiment A, one value per run, for the run-by-run comparison
    "experiment_a_runs": [
        "CNN=sleep_edf:in_domain@epoch",
        "CNN + LSTM=sleep_edf:in_domain@sequence",
        "Sem aumento=sleep_edf:in_domain@sequence,noaug",
    ],
    # experiment B: reference within each dataset (on the target test subjects) and the 6 pairs
    "experiment_b": [
        "Sleep-EDF (dentro)=sleep_edf:in_domain_ref",
        "ISRUC (dentro)=isruc:in_domain_ref",
        "HMC (dentro)=hmc:in_domain_ref",
        "Sleep-EDF → ISRUC=sleep_edf:cross:isruc",
        "Sleep-EDF → HMC=sleep_edf:cross:hmc",
        "ISRUC → Sleep-EDF=isruc:cross:sleep_edf",
        "HMC → Sleep-EDF=hmc:cross:sleep_edf",
        "ISRUC → HMC=isruc:cross:hmc",
        "HMC → ISRUC=hmc:cross:isruc",
    ],
    # experiment C: with and without the band-pass filter
    "experiment_c": [
        "CNN + LSTM, com filtro=sleep_edf:in_domain_pooled",
        "CNN + LSTM, sem filtro=sleep_edf:in_domain_pooled@nofilter",
        "Com filtro=sleep_edf:in_domain",
        "Sem filtro=sleep_edf:in_domain@nofilter",
        "Sleep-EDF → ISRUC, com filtro=sleep_edf:cross:isruc",
        "Sleep-EDF → ISRUC, sem filtro=sleep_edf:cross:isruc@nofilter",
        "Sleep-EDF → HMC, com filtro=sleep_edf:cross:hmc",
        "Sleep-EDF → HMC, sem filtro=sleep_edf:cross:hmc@nofilter",
    ],
}

# (output name, report it reads, pairs (A, B, header of A - B))
PAIRED = [
    ("experiment_a_pares", "experiment_a_runs", [
        ("CNN + LSTM", "CNN", r"$\Delta$ LSTM"),
        ("CNN + LSTM", "Sem aumento", r"$\Delta$ aumento"),
    ]),
    ("experiment_c_dentro", "experiment_c", [
        ("Com filtro", "Sem filtro", r"$\Delta$"),
    ]),
    ("experiment_c_entre", "experiment_c", [
        ("Sleep-EDF → ISRUC, com filtro", "Sleep-EDF → ISRUC, sem filtro", r"$\Delta$ ISRUC"),
        ("Sleep-EDF → HMC, com filtro", "Sleep-EDF → HMC, sem filtro", r"$\Delta$ HMC"),
    ]),
]

DATA_FIGURES = ["class_distribution", "example_epochs", "psd_per_stage"]  # written by src.data.figures

# Experiments chapter, in order: (thesis label, generated file, what the thesis adds by hand)
THESIS = [
    ("fig:distribuicao-estagios", FIGURES / "class_distribution.png", ""),
    ("fig:epocas-exemplo", FIGURES / "example_epochs.png", ""),
    ("fig:psd-estagios", FIGURES / "psd_per_stage.png", ""),
    ("tab:ambiente", RESULTS / "environment.tex", ""),
    ("tab:exp-a", RESULTS / "experiment_a_metrics.tex",
     "row of the TinySleepNet paper (Supratak and Guo, 2020) added by hand"),
    ("tab:pares", RESULTS / "experiment_a_pares.tex", ""),
    ("tab:exp-a-estagios", RESULTS / "experiment_a_f1_stages.tex",
     "row of the TinySleepNet paper added by hand"),
    ("tab:exp-b", RESULTS / "experiment_b_metrics.tex",
     "macro-F1 only; the reference of each pair is the '(dentro)' row of its target, "
     "and the difference is the pair minus that reference"),
    ("tab:exp-b-estagios", RESULTS / "experiment_b_f1_stages.tex", ""),
    ("fig:confusao-b", FIGURES / "experiment_b_confusion.png", ""),
    ("tab:exp-c-pares", RESULTS / "experiment_c_dentro.tex", ""),
    ("tab:exp-c-entre", RESULTS / "experiment_c_entre.tex", "headers grouped by pair by hand"),
    ("tab:exp-c-estagios", RESULTS / "experiment_c_f1_stages.tex",
     "only the pooled and cross rows; filter in its own column"),
]


def report_command(name, groups):
    args = ["--name", name]
    for group in groups:
        args += ["--group", group]
    return "src.evaluation.report", args


def paired_command(name, report, pairs):
    args = [str(RESULTS / f"{report}.json")]
    for a, b, header in pairs:
        args += ["--pair", a, b, header]
    return "src.evaluation.paired", args + ["--name", name]


def steps(data_figures=True):
    """(module, arguments) of every step, in the order they must run."""
    commands = [report_command(name, groups) for name, groups in REPORTS.items()]
    commands += [paired_command(*spec) for spec in PAIRED]
    commands.append(("src.evaluation.confusion_figure", [str(RESULTS / "experiment_b.json")]))
    if data_figures:
        commands.append(("src.data.figures", []))
    return commands


def produced_files():
    """Every file the steps write that the thesis uses."""
    files = set()
    for name in REPORTS:
        files |= {RESULTS / f"{name}{suffix}" for suffix in ("_metrics.tex", "_f1_stages.tex", ".json")}
    files |= {RESULTS / f"{name}.tex" for name, _, _ in PAIRED}
    files.add(FIGURES / "experiment_b_confusion.png")
    files |= {FIGURES / f"{name}.png" for name in DATA_FIGURES}
    files.add(RESULTS / "environment.tex")
    return files


# ----------------------------------------------------------------------------- environment

def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def cpu_description(cpuinfo, threads):
    """'model, N núcleos (M threads)' from /proc/cpuinfo; the model without its '24-Cores' suffix."""
    blocks = [dict(re.findall(r"^([^:\n]+?)\s*:\s*(.*)$", block, flags=re.M))
              for block in cpuinfo.strip().split("\n\n") if block.strip()]
    model = next((b["model name"] for b in blocks if "model name" in b), platform.processor() or "---")
    model = re.sub(r"\s+\d+-Cores?(\s+Processor)?$", "", model.strip())
    cores = {(b["physical id"], b["core id"]) for b in blocks if "physical id" in b and "core id" in b}
    if not cores:
        return f"{model}, {threads} threads"
    return f"{model}, {len(cores)} núcleos ({threads} threads)"


def gigabytes(n_bytes):
    return f"{n_bytes / 1024 ** 3:.0f}~GB"


def environment_rows():
    """(component, configuration or version) of this machine, for the thesis table."""
    import matplotlib
    import mne
    import numpy
    import scipy
    import sklearn
    import torch

    rows = []
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        rows.append(("GPU", f"{props.name}, {gigabytes(props.total_memory)}"))
    else:
        rows.append(("GPU", "---"))
    rows.append(("CPU", cpu_description(read_text("/proc/cpuinfo"), os.cpu_count())))
    mem = re.search(r"^MemTotal:\s*(\d+) kB", read_text("/proc/meminfo"), flags=re.M)
    rows.append(("Memória RAM", gigabytes(int(mem.group(1)) * 1024) if mem else "---"))
    pretty = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', read_text("/etc/os-release"), flags=re.M)
    rows.append(("Sistema operacional", pretty.group(1) if pretty else platform.platform()))
    rows.append(("Python", platform.python_version()))
    cuda = f" (CUDA {torch.version.cuda})" if torch.version.cuda else ""
    rows.append(("PyTorch", torch.__version__.split("+")[0] + cuda))
    for name, module in [("NumPy", numpy), ("SciPy", scipy), ("scikit-learn", sklearn),
                         ("MNE", mne), ("Matplotlib", matplotlib)]:
        rows.append((name, module.__version__))
    return rows


def environment_table(rows):
    lines = [r"\begin{tabular}{ll}", r"\hline",
             r"\textbf{Componente} & \textbf{Configuração ou versão} \\", r"\hline"]
    lines += [f"{component} & {value} \\\\" for component, value in rows]
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- manifest

def manifest(root=ROOT):
    lines = ["# Experiments chapter: where each table and figure comes from", "",
             "Written by `python -m src.evaluation.export` (issue 18).", "",
             "| Thesis label | File | Formatted by hand in the thesis | Found |", "|---|---|---|---|"]
    for label, path, note in THESIS:
        found = "yes" if (root / path).exists() else "**missing**"
        lines.append(f"| `{label}` | `{path}` | {note or '—'} | {found} |")
    return "\n".join(lines) + "\n"


def run(module, args, root=ROOT):
    print(f"\n$ python -m {module} {shlex.join(args)}", flush=True)
    subprocess.run([sys.executable, "-m", module, *args], cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description="Regenerate the tables and figures of the thesis.")
    parser.add_argument("--dry-run", action="store_true", help="only print the commands")
    parser.add_argument("--skip-data-figures", action="store_true",
                        help="skip src.data.figures (reads every recording of the three datasets)")
    args = parser.parse_args()
    commands = steps(data_figures=not args.skip_data_figures)
    if args.dry_run:
        for module, arguments in commands:
            print(f"python -m {module} {shlex.join(arguments)}")
        return
    for module, arguments in commands:
        run(module, arguments)
    (ROOT / RESULTS).mkdir(parents=True, exist_ok=True)
    (ROOT / RESULTS / "environment.tex").write_text(environment_table(environment_rows()), encoding="utf-8")
    text = manifest()
    (ROOT / RESULTS / "thesis_manifest.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print("written: outputs/results/environment.tex, outputs/results/thesis_manifest.md")


if __name__ == "__main__":
    main()
