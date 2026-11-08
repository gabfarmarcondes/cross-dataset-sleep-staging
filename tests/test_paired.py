"""Unit tests for the run-by-run comparison of report groups (issue 17). Run from the
repository root:

    python -m pytest tests/
"""
import math

import pytest

from src.evaluation.paired import compare, latex_table, parse_pairs, signed, tally


def group(label, values):
    """A report group with one entry per (fold, seed) -> macro-F1."""
    entries = [{"name": f"{label} {fold} {seed}", "fold": fold, "seed": seed,
                "values": {"macro_f1": v}} for (fold, seed), v in values.items()]
    return {"label": label, "entries": entries}


def test_pairs_by_fold_and_seed_and_leaves_out_what_is_not_in_both():
    a = group("A", {(0, 0): 0.80, (0, 1): 0.70, (1, 0): 0.75, (1, 1): 0.60})
    b = group("B", {(1, 0): 0.70, (0, 0): 0.82, (0, 1): 0.70})  # (1, 1) missing
    table = compare({"groups": [a, b]}, [("A", "B", r"$\Delta$")])
    assert table["keys"] == [(0, 0), (0, 1), (1, 0)]  # sorted by fold, then seed
    _, _, _, differences = table["deltas"][0]
    assert differences == pytest.approx([-0.02, 0.0, 0.05])
    assert table["left_out"] == {"A": [(1, 1)]}
    assert tally(differences) == (1, 1, 1)


def test_several_pairs_share_the_groups():
    a, b, c = (group(n, {(0, 0): v}) for n, v in [("A", 0.8), ("B", 0.7), ("C", 0.75)])
    table = compare({"groups": [a, b, c]}, parse_pairs([["A", "B"], ["A", "C", "D"]]))
    assert table["labels"] == ["A", "B", "C"]
    assert [h for h, *_ in table["deltas"]] == [r"$\Delta$", "D"]


def test_unknown_groups_and_groups_without_common_entries_are_errors():
    a = group("A", {(0, 0): 0.8})
    pooled = group("P", {(None, 0): 0.8})
    with pytest.raises(ValueError):
        compare({"groups": [a]}, [("A", "X", "")])
    with pytest.raises(ValueError):
        compare({"groups": [a, pooled]}, [("A", "P", "")])  # per run vs per seed


def test_signs_in_the_portuguese_format():
    assert signed(0.4) == "+0,4"
    assert signed(-1.24) == "$-$1,2"
    assert signed(0.0) == "0,0" and signed(-0.01) == "0,0"
    assert signed(float("nan")) == "---"
    assert tally([0.1, float("nan"), -0.2]) == (1, 1, 0)


def test_latex_table_of_runs_and_of_seeds():
    runs = compare({"groups": [group("A", {(0, 0): 0.80, (0, 1): 0.90}),
                               group("B", {(0, 0): 0.78, (0, 1): 0.85})]}, [("A", "B", "D")])
    text = latex_table(runs)
    assert "Fold & Semente & A & B & D" in text
    assert "0 & 1 & 90,0 & 85,0 & +5,0" in text
    assert r"\multicolumn{2}{c}{Média $\pm$ desvio} & 85,0 $\pm$ 7,1 & 81,5 $\pm$ 4,9 & +3,5 $\pm$ 2,1" in text

    seeds = compare({"groups": [group("A", {(None, 0): 0.70}), group("B", {(None, 0): 0.72})]},
                    [("A", "B", "D")])
    text = latex_table(seeds)
    assert "Fold" not in text and "Semente & A & B & D" in text
    assert r"Média $\pm$ desvio & 70,0 & 72,0 & $-$2,0 \\" in text  # one pair: no deviation
    assert not math.isnan(seeds["deltas"][0][3][0])
