"""Run-by-run comparison of groups of a report (issue 17). From the repository root, after
src.evaluation.report:

    python -m src.evaluation.paired outputs/results/experiment_c.json \\
        --pair "Filtered" "Unfiltered"

Each --pair A B [HEADER] compares two groups of the report on the entries they have in
common: the same fold and seed (in_domain groups, one entry per run) or the same seed
(cross and pooled groups, one entry per seed). The difference is A - B, for each pair of
entries; HEADER is the column title of the difference in the LaTeX table (default
$\\Delta$). With several --pair options, the table has every group once and one difference
column per pair, e.g. for experiment A (in_domain groups, one value per run):

    python -m src.evaluation.paired outputs/results/<report>.json \\
        --pair "CNN + LSTM" "CNN" '$\\Delta$ LSTM' \\
        --pair "CNN + LSTM" "CNN + LSTM, no augmentation" '$\\Delta$ aumento'

Entries that are not in every group of the table are left out, and the script says which.
Prints every value and, for each pair, the mean ± standard deviation of the difference and
in how many pairs A was higher, lower or equal. Writes outputs/results/<--name>.tex
(default <report name>_paired): one row per fold and seed, then the mean ± standard
deviation of every column, in % (differences in percentage points); macro-F1 unless
--metric.
"""
import argparse
import json
import math
from pathlib import Path

from src.evaluation.metrics import summarize
from src.evaluation.report import METRICS, RESULTS_DIR, mean_std, pt


def entry_values(group, metric):
    """{(fold, seed): value} of a group; fold is None outside the in_domain scenario."""
    values = {}
    for entry in group["entries"]:
        key = (entry["fold"], entry["seed"])
        if key in values:
            raise ValueError(f"{group['label']}: two entries for fold {key[0]}, seed {key[1]}")
        values[key] = entry["values"][metric]
    return values


def sort_key(key):
    fold, seed = key
    return (-1 if fold is None else fold, seed)


def compare(report, pairs, metric="macro_f1"):
    """The comparison table of the given pairs [(A, B, header)] of group labels.

    Returns labels (the groups, each once, in order), keys (the (fold, seed) entries common
    to all of them), values {label: {key: value}}, deltas [(header, A, B, [A - B per key])]
    and left_out {label: keys of that group that are not in the table}.
    """
    groups = {g["label"]: g for g in report["groups"]}
    labels = []
    for a, b, _ in pairs:
        for label in (a, b):
            if label not in groups:
                raise ValueError(f"no group {label!r} in the report; its groups are {list(groups)}")
            if label not in labels:
                labels.append(label)
    values = {label: entry_values(groups[label], metric) for label in labels}
    common = set.intersection(*(set(v) for v in values.values()))
    if not common:
        raise ValueError(f"the groups {labels} have no fold and seed in common")
    keys = sorted(common, key=sort_key)
    left_out = {label: sorted(set(v) - common, key=sort_key) for label, v in values.items()
                if set(v) - common}
    deltas = [(header, a, b, [values[a][k] - values[b][k] for k in keys]) for a, b, header in pairs]
    return {"labels": labels, "keys": keys, "values": values, "deltas": deltas, "left_out": left_out}


def tally(differences):
    """(higher, lower, equal): in how many pairs A was above, below or equal to B (NaN skipped)."""
    valid = [d for d in differences if not math.isnan(d)]
    return sum(d > 0 for d in valid), sum(d < 0 for d in valid), sum(d == 0 for d in valid)


def signed(value, digits=1):
    """Difference with its sign, in the Portuguese format: +0,4 / $-$1,2 / 0,0."""
    if math.isnan(value):
        return "---"
    text = pt(abs(value), digits)
    if text.strip("0,") == "":
        return text  # rounds to zero: no sign
    return ("+" if value > 0 else "$-$") + text


def percent(value):
    return "---" if math.isnan(value) else pt(100 * value, 1)


def latex_table(table):
    """tabular: fold (in_domain only) and seed, the value of every group, the differences."""
    with_fold = table["keys"][0][0] is not None
    header = (["Fold"] if with_fold else []) + ["Semente"] + table["labels"]
    header += [h for h, *_ in table["deltas"]]
    rows = []
    for i, key in enumerate(table["keys"]):
        row = ([str(key[0])] if with_fold else []) + [str(key[1])]
        row += [percent(table["values"][label][key]) for label in table["labels"]]
        row += [signed(100 * d[i]) for *_, d in table["deltas"]]
        rows.append(row)
    mean_row = [r"\multicolumn{2}{c}{Média $\pm$ desvio}" if with_fold else r"Média $\pm$ desvio"]
    mean_row += [mean_std(summarize([table["values"][label][k] for k in table["keys"]]), 100)
                 for label in table["labels"]]
    for *_, differences in table["deltas"]:
        s = summarize(differences)
        mean_row.append(signed(100 * s["mean"])
                        + ("" if math.isnan(s["std"]) else r" $\pm$ " + pt(100 * s["std"], 1)))
    align = ("cc" if with_fold else "c") + "c" * (len(table["labels"]) + len(table["deltas"]))
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\hline", " & ".join(header) + r" \\", r"\hline"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\hline", " & ".join(mean_row) + r" \\", r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def print_table(table, metric):
    keys = table["keys"]
    with_fold = keys[0][0] is not None
    print(f"{metric} (%), {len(keys)} pairs ({'fold and seed' if with_fold else 'seed'})")
    names = table["labels"] + [f"{a} - {b}" for _, a, b, _ in table["deltas"]]
    print(("fold " if with_fold else "") + "seed  " + " | ".join(names))
    for i, key in enumerate(keys):
        cells = [f"{100 * table['values'][label][key]:.1f}" for label in table["labels"]]
        cells += [f"{100 * d[i]:+.1f}" for *_, d in table["deltas"]]
        print((f"{key[0]:>4} " if with_fold else "") + f"{key[1]:>4}  " + " | ".join(cells))
    for label, missing in table["left_out"].items():
        print(f"left out of {label} (not in every group): {missing}")
    for _, a, b, differences in table["deltas"]:
        s = summarize(differences)
        higher, lower, equal = tally(differences)
        print(f"{a} - {b}: {100 * s['mean']:+.1f} ± {100 * s['std']:.1f} points; "
              f"{a} higher in {higher}, lower in {lower}, equal in {equal} of {len(differences)} pairs")


def parse_pairs(raw):
    pairs = []
    for values in raw:
        if len(values) not in (2, 3):
            raise SystemExit(f"--pair takes A B [HEADER], got {values}")
        a, b = values[:2]
        pairs.append((a, b, values[2] if len(values) == 3 else r"$\Delta$"))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Run-by-run comparison of groups of a report.")
    parser.add_argument("report", type=Path, help="JSON written by src.evaluation.report")
    parser.add_argument("--pair", nargs="+", action="append", required=True, metavar="LABEL",
                        help="A B [HEADER]: compare group A with group B (A - B); repeat for more")
    parser.add_argument("--metric", choices=METRICS, default="macro_f1")
    parser.add_argument("--name", help="output file name (default: <report name>_paired)")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    try:
        table = compare(report, parse_pairs(args.pair), args.metric)
    except ValueError as error:
        raise SystemExit(str(error))
    print_table(table, args.metric)
    name = args.name or f"{report['name']}_paired"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.tex").write_text(latex_table(table), encoding="utf-8")
    print(f"\nwritten: outputs/results/{name}.tex")


if __name__ == "__main__":
    main()
