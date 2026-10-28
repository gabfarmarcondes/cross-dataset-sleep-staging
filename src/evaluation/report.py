"""Results tables of an experiment, in one command (issue 13). From the repository root:

    python -m src.evaluation.report --name in_domain \\
        --group "Sleep-EDF=sleep_edf:in_domain" \\
        --group "ISRUC=isruc:in_domain" \\
        --group "HMC=hmc:in_domain"

A group is LABEL=DATASET:SCENARIO[:TARGET]:
    in_domain       runs of DATASET trained with --fold (all folds and seeds), each tested on
                    its fold's test subjects; one value per run
    cross:TARGET    runs of DATASET trained with --source, tested on TARGET's target.test
                    subjects; one value per run (seed)
    in_domain_ref   the in-domain runs of DATASET again, but only on DATASET's target.test
                    subjects, pooled over the folds of each seed: the in-domain reference for
                    the cross-dataset scenarios whose target is DATASET; one value per seed

Only runs with the chosen mode, batch norm and filter setting are used (--mode, --batch-norm,
--filter; defaults from configs/experiment.yaml). Runs not yet evaluated are evaluated first
(src/evaluation/evaluate.py). The tables show mean ± standard deviation over the values.

Writes in outputs/results/:
    <name>.json            every value of every run, summaries and summed confusion matrices
    <name>.csv             one row per group (fractions, not percentages)
    <name>_metrics.tex     accuracy, macro-F1 and kappa         (tabular, for \\input{} in the thesis)
    <name>_f1_stages.tex   F1 per stage
    <name>_cost.tex        parameters, training and inference time
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import yaml

from src.data.format import STAGES
from src.data.splits import load_splits
from src.evaluation.evaluate import evaluate_run, file_suffix
from src.evaluation.metrics import compute_metrics, normalize_rows, summarize
from src.models.tinysleepnet import BATCH_NORMS
from src.read_data import ROOT
from src.training.loop import MODES
from src.training.train import EXPERIMENT_FILE, RUNS_DIR

RESULTS_DIR = ROOT / "outputs" / "results"
SCENARIOS = ("in_domain", "cross", "in_domain_ref")
METRICS = ["accuracy", "macro_f1", "kappa"] + [f"f1_{s}" for s in STAGES]
COSTS = ["parameters", "seconds_per_training_epoch", "epochs_run", "inference_seconds_per_recording"]


# ----------------------------------------------------------------------------- groups and runs

def parse_group(spec):
    """'LABEL=DATASET:SCENARIO[:TARGET]' -> dict."""
    label, sep, rest = spec.partition("=")
    parts = rest.split(":")
    if not sep or len(parts) not in (2, 3) or parts[1] not in SCENARIOS:
        raise ValueError(f"group {spec!r}: expected LABEL=DATASET:SCENARIO[:TARGET], "
                         f"SCENARIO in {SCENARIOS}")
    dataset, scenario = parts[0], parts[1]
    target = parts[2] if len(parts) == 3 else None
    if (scenario == "cross") != (target is not None):
        raise ValueError(f"group {spec!r}: a target goes with the cross scenario, and only with it")
    return {"label": label.strip(), "spec": spec, "dataset": dataset, "scenario": scenario,
            "target": target}


def find_runs(dataset, scenario, mode, batch_norm, filtered, runs_dir=RUNS_DIR):
    """Finished runs of a dataset and scenario with the given settings, by name."""
    found = []
    for run_json in sorted(runs_dir.glob("*/run.json")):
        info = json.loads(run_json.read_text(encoding="utf-8"))
        split = info["split"]
        right_split = split.get("source") if scenario == "cross" else split.get("fold") is not None
        if (info["dataset"] == dataset and right_split and info["mode"] == mode
                and info.get("model", {}).get("batch_norm", "standard") == batch_norm
                and info["preprocessing"]["filter"] == filtered
                and "result" in info and (run_json.parent / "best.pt").exists()):
            found.append((run_json.parent, info))
    return found


def evaluation_of(run_dir, target, force, log):
    path = run_dir / f"evaluation{file_suffix(target)}.json"
    if force or not path.exists():
        return evaluate_run(run_dir, target, log=log)
    return json.loads(path.read_text(encoding="utf-8"))


def values_of(metrics):
    out = {k: metrics[k] for k in ("accuracy", "macro_f1", "kappa")}
    out.update({f"f1_{s}": metrics["f1"][s] for s in STAGES})
    return out


def read_predictions(run_dir):
    with open(run_dir / "predictions.csv", encoding="utf-8") as f:
        return [(int(r["subject_id"]), int(r["y_true"]), int(r["y_pred"])) for r in csv.DictReader(f)]


def collect(group, runs, force, log):
    """Per-value metrics (and costs) of one group."""
    entries = []
    if group["scenario"] in ("in_domain", "cross"):
        for run_dir, info in runs:
            result = evaluation_of(run_dir, group["target"], force, log)
            entries.append({"name": info["run"], "seed": info["seed"],
                            "fold": info["split"].get("fold"),
                            "values": values_of(result["metrics"]),
                            "confusion": result["metrics"]["confusion"],
                            "cost": {k: result["cost"][k] for k in COSTS}})
        return entries

    # in_domain_ref: pool the folds of each seed, keep only the target.test subjects
    target_test = set(load_splits(group["dataset"])["cross_dataset"]["target"]["test"])
    by_seed = defaultdict(dict)
    for run_dir, info in runs:
        by_seed[info["seed"]][info["split"]["fold"]] = run_dir
    k = load_splits(group["dataset"])["in_domain"]["k"]
    for seed, folds in sorted(by_seed.items()):
        if len(folds) < k:
            log(f"  {group['label']}: seed {seed} has {len(folds)} of {k} folds, skipped")
            continue
        rows = []
        for fold in sorted(folds):
            evaluation_of(folds[fold], None, force, log)
            rows += [r for r in read_predictions(folds[fold]) if r[0] in target_test]
        missing = target_test - {r[0] for r in rows}
        if missing:
            raise ValueError(f"{group['label']}: seed {seed} has no predictions for subjects {sorted(missing)}")
        metrics = compute_metrics([r[1] for r in rows], [r[2] for r in rows])
        entries.append({"name": f"seed {seed} (folds pooled)", "seed": seed, "fold": None,
                        "values": values_of(metrics), "confusion": metrics["confusion"],
                        "cost": None})
    return entries


def summarize_group(entries):
    summary = {m: summarize([e["values"][m] for e in entries]) for m in METRICS}
    with_cost = [e for e in entries if e["cost"]]
    summary.update({c: summarize([e["cost"][c] for e in with_cost]) for c in COSTS})
    confusion = np.sum([e["confusion"] for e in entries], axis=0) if entries else np.zeros((5, 5))
    return summary, np.asarray(confusion, dtype=int)


# ----------------------------------------------------------------------------- formatting

def pt(value, digits):
    """Number in the Portuguese format (decimal comma, dot for thousands)."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def mean_std(stats, scale=1.0, digits=1):
    if stats["n"] == 0:
        return "---"
    text = pt(stats["mean"] * scale, digits)
    if not np.isnan(stats["std"]):
        text += r" $\pm$ " + pt(stats["std"] * scale, digits)
    return text


def tabular(header, rows, align):
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\hline", " & ".join(header) + r" \\", r"\hline"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_tables(groups):
    """The three tabulars, with values in percent (kappa as a fraction)."""
    metrics = tabular(
        [r"Cenário", "$n$", r"Acurácia (\%)", r"Macro-F1 (\%)", r"$\kappa$"],
        [[g["label"], str(g["summary"]["macro_f1"]["n"]),
          mean_std(g["summary"]["accuracy"], 100), mean_std(g["summary"]["macro_f1"], 100),
          mean_std(g["summary"]["kappa"], 1, 2)] for g in groups],
        "lrrrr")
    stages = tabular(
        [r"Cenário"] + [rf"{s} (\%)" for s in STAGES],
        [[g["label"]] + [mean_std(g["summary"][f"f1_{s}"], 100) for s in STAGES] for g in groups],
        "l" + "r" * len(STAGES))
    cost = tabular(
        [r"Cenário", "Parâmetros", r"Treino (s/época de treino)", r"Épocas de treino",
         r"Inferência (s/registro)"],
        [[g["label"],
          pt(g["summary"]["parameters"]["mean"], 0),  # the same in every run
          mean_std(g["summary"]["seconds_per_training_epoch"], 1, 0),
          mean_std(g["summary"]["epochs_run"], 1, 0),
          mean_std(g["summary"]["inference_seconds_per_recording"], 1, 2)]
         for g in groups if g["scenario"] != "in_domain_ref"],
        "lrrrr")
    return metrics, stages, cost


# ----------------------------------------------------------------------------- main

def parse_args():
    with open(EXPERIMENT_FILE, encoding="utf-8") as f:
        experiment = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description="Results tables of an experiment.")
    parser.add_argument("--name", required=True, help="file name in outputs/results/")
    parser.add_argument("--group", action="append", required=True,
                        help="LABEL=DATASET:SCENARIO[:TARGET]; repeat for more rows")
    parser.add_argument("--mode", choices=MODES, default=experiment["training"]["mode"])
    parser.add_argument("--batch-norm", choices=BATCH_NORMS,
                        default=experiment.get("model", {}).get("batch_norm", "official"))
    parser.add_argument("--filter", choices=["on", "off"],
                        default="on" if experiment["preprocessing"]["filter"] else "off")
    parser.add_argument("--force", action="store_true", help="evaluate again even if already evaluated")
    return parser.parse_args()


def main():
    args = parse_args()
    filtered = args.filter == "on"
    groups = []
    for spec in args.group:
        group = parse_group(spec)
        runs = find_runs(group["dataset"], group["scenario"], args.mode, args.batch_norm, filtered)
        if not runs:
            raise SystemExit(f"{group['label']}: no finished {args.mode} runs of {group['dataset']} "
                             f"({group['scenario']}, {args.batch_norm} batch norm, filter {args.filter})")
        print(f"{group['label']}: {len(runs)} runs")
        entries = collect(group, runs, args.force, log=lambda message: print("  " + message))
        group["entries"] = entries
        group["summary"], confusion = summarize_group(entries)
        group["confusion_sum"] = confusion.tolist()
        group["confusion_normalized"] = normalize_rows(confusion).tolist()
        groups.append(group)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"name": args.name, "created": datetime.now().isoformat(timespec="seconds"),
              "mode": args.mode, "batch_norm": args.batch_norm, "filter": args.filter,
              "groups": groups}
    (RESULTS_DIR / f"{args.name}.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    with open(RESULTS_DIR / f"{args.name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        keys = METRICS + COSTS
        writer.writerow(["label", "dataset", "scenario", "target", "n"]
                        + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        for g in groups:
            s = g["summary"]
            writer.writerow([g["label"], g["dataset"], g["scenario"], g["target"] or "",
                             s["macro_f1"]["n"]]
                            + [f"{s[k][stat]:.4f}" for k in keys for stat in ("mean", "std")])
    for suffix, text in zip(("metrics", "f1_stages", "cost"), latex_tables(groups)):
        (RESULTS_DIR / f"{args.name}_{suffix}.tex").write_text(text, encoding="utf-8")

    print()
    for g in groups:
        s = g["summary"]
        print(f"{g['label']:<28} n={s['macro_f1']['n']}  "
              f"accuracy {s['accuracy']['mean']:.1%}  macro-F1 {s['macro_f1']['mean']:.1%} "
              f"(± {s['macro_f1']['std']:.1%})  kappa {s['kappa']['mean']:.3f}")
    print(f"\nwritten: outputs/results/{args.name}.json, .csv, _metrics.tex, _f1_stages.tex, _cost.tex")


if __name__ == "__main__":
    main()