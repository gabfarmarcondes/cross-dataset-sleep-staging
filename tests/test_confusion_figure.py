"""Tests for the confusion-matrix figure of cross-dataset reports (issue 16)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.evaluation.confusion_figure import mean_macro_f1, panel_key, plot, row_shares


def group(scenario, dataset, target=None, confusion=None, macro_f1=(0.7, 0.8)):
    confusion = confusion if confusion is not None else (np.eye(5) * 10 + 1).astype(int).tolist()
    entries = [{"name": f"seed {s}", "seed": s, "fold": None, "confusion": confusion,
                "values": {"macro_f1": f}} for s, f in enumerate(macro_f1)]
    return {"label": f"{dataset}->{target}", "scenario": scenario, "dataset": dataset,
            "target": target, "entries": entries}


def test_panel_key_places_references_on_the_diagonal_and_pairs_off_it():
    assert panel_key(group("in_domain_ref", "isruc")) == ("isruc", "isruc")
    assert panel_key(group("cross", "sleep_edf", "hmc")) == ("sleep_edf", "hmc")
    assert panel_key(group("in_domain_pooled", "sleep_edf")) is None


def test_row_shares_sums_the_entries_and_normalizes_each_true_stage():
    counts = [[8, 2, 0, 0, 0], [1, 1, 2, 0, 0], [0, 0, 5, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 3, 1]]
    share = row_shares(group("cross", "isruc", "hmc", confusion=counts))
    assert share[0].tolist() == pytest.approx([80, 20, 0, 0, 0])
    assert share[1].tolist() == pytest.approx([25, 25, 50, 0, 0])
    assert share[3].tolist() == [0, 0, 0, 0, 0]  # a stage absent from the test set stays at 0
    assert share[4].tolist() == pytest.approx([0, 0, 0, 75, 25])


def test_row_shares_prefers_the_summed_matrix_of_the_report():
    g = group("cross", "isruc", "hmc")
    g["confusion_sum"] = np.diag([1, 2, 3, 4, 5]).tolist()
    assert np.allclose(row_shares(g), np.eye(5) * 100)


def test_mean_macro_f1_is_the_mean_of_the_entries_in_percent():
    assert mean_macro_f1(group("cross", "isruc", "hmc", macro_f1=(0.6, 0.7, 0.8))) == pytest.approx(70)


def test_plot_draws_nine_panels_and_leaves_missing_pairs_blank():
    datasets = ["sleep_edf", "isruc", "hmc"]
    groups = [group("in_domain_ref", d) for d in datasets]
    groups += [group("cross", s, t) for s in datasets for t in datasets if s != t and (s, t) != ("hmc", "isruc")]
    fig = plot({"name": "test", "groups": groups})
    panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert len(panels) == 9
    blank = [ax for ax in panels if not ax.axison]
    assert len(blank) == 1  # hmc -> isruc was left out