"""Results tables of an experiment, in one command (issue 13). From the repository root:

    python -m src.evaluation.report --name in_domain \\
        --group "Sleep-EDF=sleep_edf:in_domain" \\
        --group "ISRUC=isruc:in_domain" \\
        --group "HMC=hmc:in_domain"

A group is LABEL=DATASET:SCENARIO[:TARGET][@OPTION[,OPTION]]:
    in_domain         runs of DATASET trained with --fold (all folds and seeds), each tested
                      on its fold's test subjects; one value per run
    in_domain_pooled  the same runs, with the test epochs of the folds of each seed pooled
                      before computing the metrics, as the official TinySleepNet code does
                      (predict.py); one value per seed
    in_domain_ref     the same runs, pooled per seed, but only on DATASET's target.test
                      subjects: the in-domain reference for the cross-dataset scenarios whose
                      target is DATASET; one value per seed
    cross:TARGET      runs of DATASET trained with --source, tested on TARGET's target.test
                      subjects; one value per run (seed)
Options after @: a mode (epoch or sequence) overrides --mode for that group, noaug
takes the runs trained with --no-augment instead (issue 15), and nofilter the runs trained
with --no-filter (issue 17). So CNN-only and CNN + LSTM, with and without data augmentation,
or with and without the band-pass filter, can be rows of the same table:

    python -m src.evaluation.report --name experiment_a \
        --group "CNN=sleep_edf:in_domain_pooled@epoch" \
        --group "CNN + LSTM=sleep_edf:in_domain_pooled@sequence" \
        --group "CNN + LSTM, no augmentation=sleep_edf:in_domain_pooled@sequence,noaug"

    python -m src.evaluation.report --name experiment_c \
        --group "Filtered=sleep_edf:in_domain" \
        --group "Unfiltered=sleep_edf:in_domain@nofilter"

Only runs with the chosen mode, batch norm and filter setting (--mode, --batch-norm,
--filter; defaults from configs/experiment.yaml; nofilter groups: filter off) and with the
`training` block of experiment.yaml (augmentation off for noaug groups) are used: a quick
test with --max-epochs never enters a table. Runs not yet evaluated are evaluated first
(src/evaluation/evaluate.py). The tables show mean ± standard deviation over the values;
src/evaluation/paired.py compares two groups run by run (same fold and seed).

Writes in outputs/results/:
    <name>.json            every value of every run, summaries and summed confusion matrices
    <name>.csv             one row per group (fractions, not percentages)
    <name>_metrics.tex     accuracy, macro-F1 and kappa         (tabular, for \\input{} in the thesis)
    <name>_f1_stages.tex   F1 per stage
    <name>_cost.tex        parameters, training and inference time (over the runs behind each group)
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
from src.training.augment import without_augmentation
from src.training.loop import MODES
from src.training.train import EXPERIMENT_FILE, RUNS_DIR, same_training

RESULTS_DIR = ROOT / "outputs" / "results"
SCENARIOS = ("in_domain", "in_domain_pooled", "in_domain_ref", "cross")
SETTINGS = ("noaug", "nofilter")  # group options besides a mode
METRICS = ["accuracy", "macro_f1", "kappa"] + [f"f1_{s}" for s in STAGES]
COSTS = ["parameters", "seconds_per_training_epoch", "epochs_run", "inference_seconds_per_recording"]


# ----------------------------------------------------------------------------- groups and runs

def parse_group(spec):
    """'LABEL=DATASET:SCENARIO[:TARGET][@OPTION[,OPTION]]' -> dict (mode None: use --mode;
    filter None: use --filter)."""
    label, sep, rest = spec.partition("=")
    rest, _, options = rest.partition("@")
    options = [o.strip() for o in options.split(",")] if options else []
    modes = [o for o in options if o in MODES]
    parts = rest.split(":")
    if (not sep or len(parts) not in (2, 3) or parts[1] not in SCENARIOS or len(modes) > 1
            or any(o not in MODES and o not in SETTINGS for o in options)):
        raise ValueError(f"group {spec!r}: expected LABEL=DATASET:SCENARIO[:TARGET][@OPTION[,OPTION]], "
                         f"SCENARIO in {SCENARIOS}, OPTION a mode in {MODES} or one of {SETTINGS}")
    dataset, scenario = parts[0], parts[1]
    target = parts[2] if len(parts) == 3 else None
    if (scenario == "cross") != (target is not None):
        raise ValueError(f"group {spec!r}: a target goes with the cross scenario, and only with it")
    return {"label": label.strip(), "spec": spec, "dataset": dataset, "scenario": scenario,
            "target": target, "mode": modes[0] if modes else None, "augment": "noaug" not in options,
            "filter": False if "nofilter" in options else None}


def find_runs(dataset, scenario, mode, batch_norm, filtered, training, runs_dir=RUNS_DIR):
    """Finished runs of a dataset and scenario with the given settings, by name.

    training: the `training` block the runs must have been trained with (mode aside).
    """
    found = []
    for run_json in sorted(runs_dir.glob("*/run.json")):
        info = json.loads(run_json.read_text(encoding="utf-8"))
        split = info["split"]
        right_split = split.get("source") if scenario == "cross" else split.get("fold") is not None
        if (info["dataset"] == dataset and right_split and info["mode"] == mode
                and info.get("model", {}).get("batch_norm", "standard") == batch_norm
                and info["preprocessing"]["filter"] == filtered
                and same_training(info["training"], training)
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


def cost_of(result):
    return {k: result["cost"][k] for k in COSTS} if "cost" in result else None


def collect(group, runs, force, log):
    """(entries, costs) of one group: the metric values, and the cost of every run used."""
    entries, costs = [], []
    if group["scenario"] in ("in_domain", "cross"):
        for run_dir, info in runs:
            result = evaluation_of(run_dir, group["target"], force, log)
            entries.append({"name": info["run"], "seed": info["seed"],
                            "fold": info["split"].get("fold"),
                            "values": values_of(result["metrics"]),
                            "confusion": result["metrics"]["confusion"]})
            costs.append(cost_of(result))
        return entries, [c for c in costs if c]

    # pooled scenarios: the test epochs of the k folds of each seed together; in_domain_ref
    # keeps only the target.test subjects
    splits = load_splits(group["dataset"])
    if group["scenario"] == "in_domain_ref":
        keep = set(splits["cross_dataset"]["target"]["test"])
    else:
        keep = {int(s) for s in splits["recordings"]}
    by_seed = defaultdict(dict)
    for run_dir, info in runs:
        by_seed[info["seed"]][info["split"]["fold"]] = run_dir
    k = splits["in_domain"]["k"]
    for seed, folds in sorted(by_seed.items()):
        if len(folds) < k:
            log(f"  {group['label']}: seed {seed} has {len(folds)} of {k} folds, skipped")
            continue
        rows = []
        for fold in sorted(folds):
            costs.append(cost_of(evaluation_of(folds[fold], None, force, log)))
            rows += [r for r in read_predictions(folds[fold]) if r[0] in keep]
        missing = keep - {r[0] for r in rows}
        if missing:
            raise ValueError(f"{group['label']}: seed {seed} has no predictions for subjects {sorted(missing)}")
        metrics = compute_metrics([r[1] for r in rows], [r[2] for r in rows])
        entries.append({"name": f"seed {seed} (folds pooled)", "seed": seed, "fold": None,
                        "values": values_of(metrics), "confusion": metrics["confusion"]})
    return entries, [c for c in costs if c]


def summarize_group(entries, costs):
    """Metrics over the entries; costs over the runs (in pooled groups, one per fold run)."""
    summary = {m: summarize([e["values"][m] for e in entries]) for m in METRICS}
    summary.update({c: summarize([cost[c] for cost in costs]) for c in COSTS})
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
         for g in groups],
        "lrrrr")
    return metrics, stages, cost


# ----------------------------------------------------------------------------- main

def parse_args():
    with open(EXPERIMENT_FILE, encoding="utf-8") as f:
        experiment = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description="Results tables of an experiment.")
    parser.add_argument("--name", required=True, help="file name in outputs/results/")
    parser.add_argument("--group", action="append", required=True,
                        help="LABEL=DATASET:SCENARIO[:TARGET][@OPTION[,OPTION]]; repeat for more rows")
    parser.add_argument("--mode", choices=MODES, default=experiment["training"]["mode"])
    parser.add_argument("--batch-norm", choices=BATCH_NORMS,
                        default=experiment.get("model", {}).get("batch_norm", "official"))
    parser.add_argument("--filter", choices=["on", "off"],
                        default="on" if experiment["preprocessing"]["filter"] else "off",
                        help="filter setting of the groups without the nofilter option")
    parser.add_argument("--force", action="store_true", help="evaluate again even if already evaluated")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(EXPERIMENT_FILE, encoding="utf-8") as f:
        training = yaml.safe_load(f)["training"]
    groups = []
    for spec in args.group:
        group = parse_group(spec)
        group["mode"] = group["mode"] or args.mode
        if group["filter"] is None:
            group["filter"] = args.filter == "on"
        expected = training if group["augment"] else without_augmentation(training)
        runs = find_runs(group["dataset"], group["scenario"], group["mode"], args.batch_norm,
                         group["filter"], expected)
        if not runs:
            raise SystemExit(f"{group['label']}: no finished {group['mode']} runs of {group['dataset']} "
                             f"({group['scenario']}, {args.batch_norm} batch norm, "
                             f"filter {'on' if group['filter'] else 'off'}, "
                             f"{'with' if group['augment'] else 'without'} augmentation, other training "
                             f"settings as in experiment.yaml) in {RUNS_DIR}; train them first")
        print(f"{group['label']}: {len(runs)} runs")
        entries, costs = collect(group, runs, args.force, log=lambda message: print("  " + message))
        group["entries"], group["costs"] = entries, costs
        group["summary"], confusion = summarize_group(entries, costs)
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
        writer.writerow(["label", "dataset", "scenario", "target", "mode", "augment", "filter", "n"]
                        + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        for g in groups:
            s = g["summary"]
            writer.writerow([g["label"], g["dataset"], g["scenario"], g["target"] or "", g["mode"],
                             int(g["augment"]), int(g["filter"]), s["macro_f1"]["n"]]
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
