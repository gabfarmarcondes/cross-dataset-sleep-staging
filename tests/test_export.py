"""Unit tests for the export of the thesis tables and figures (issue 18). Run from the
repository root:

    python -m pytest tests/
"""
import subprocess

import pytest

from src.evaluation import export
from src.evaluation.report import parse_group

CPUINFO = """processor\t: 0
model name\t: AMD Ryzen Threadripper 7960X 24-Cores
physical id\t: 0
core id\t\t: 0

processor\t: 1
model name\t: AMD Ryzen Threadripper 7960X 24-Cores
physical id\t: 0
core id\t\t: 1

processor\t: 2
model name\t: AMD Ryzen Threadripper 7960X 24-Cores
physical id\t: 0
core id\t\t: 0
"""


def test_every_group_of_every_report_is_valid():
    for groups in export.REPORTS.values():
        labels = [parse_group(spec)["label"] for spec in groups]
        assert len(labels) == len(set(labels))  # paired.py finds a group by its label


def test_every_pair_uses_labels_of_the_report_it_reads():
    for _, report, pairs in export.PAIRED:
        labels = {parse_group(spec)["label"] for spec in export.REPORTS[report]}
        for a, b, _ in pairs:
            assert a in labels and b in labels


def test_paired_tables_compare_entries_of_the_same_kind():
    """In-domain groups pair by fold and seed; pooled and cross groups by seed only."""
    for _, report, pairs in export.PAIRED:
        scenario = {parse_group(s)["label"]: parse_group(s)["scenario"] for s in export.REPORTS[report]}
        for a, b, _ in pairs:
            assert (scenario[a] == "in_domain") == (scenario[b] == "in_domain")


def test_steps_run_the_reports_before_what_reads_them():
    modules = [module for module, _ in export.steps()]
    last_report = max(i for i, m in enumerate(modules) if m == "src.evaluation.report")
    assert all(i > last_report for i, m in enumerate(modules)
               if m in ("src.evaluation.paired", "src.evaluation.confusion_figure"))
    assert "src.data.figures" not in [m for m, _ in export.steps(data_figures=False)]


def test_every_thesis_table_and_figure_comes_from_a_step():
    produced = export.produced_files()
    for label, path, _ in export.THESIS:
        assert path in produced, label
    assert len({label for label, _, _ in export.THESIS}) == len(export.THESIS)


def test_cpu_description_counts_physical_cores_and_threads():
    assert export.cpu_description(CPUINFO, 3) == "AMD Ryzen Threadripper 7960X, 2 núcleos (3 threads)"
    assert export.cpu_description("", 8).endswith("8 threads")


def test_environment_table_has_one_row_per_component():
    rows = export.environment_rows()
    assert [r[0] for r in rows][:4] == ["GPU", "CPU", "Memória RAM", "Sistema operacional"]
    text = export.environment_table(rows)
    assert text.startswith(r"\begin{tabular}{ll}") and text.count(r" \\") == len(rows) + 1


def test_manifest_marks_missing_files(tmp_path):
    (tmp_path / export.FIGURES).mkdir(parents=True)
    (tmp_path / export.FIGURES / "class_distribution.png").touch()
    text = export.manifest(tmp_path)
    assert "| `fig:distribuicao-estagios` | `outputs/figures/class_distribution.png` | — | yes |" in text
    assert text.count("**missing**") == len(export.THESIS) - 1


def test_dry_run_runs_nothing(monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("a dry run must not start any process")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr("sys.argv", ["export", "--dry-run"])
    export.main()
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(export.steps())
    assert lines[0].startswith("python -m src.evaluation.report --name experiment_a ")
    assert "'Sleep-EDF → HMC, com filtro'" in lines[-3]  # labels with spaces stay quoted
