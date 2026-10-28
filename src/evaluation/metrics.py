"""Metrics of the evaluation module (issue 13, D13). No model here: only labels in, numbers out.

    compute_metrics(y_true, y_pred)  one evaluation (one run on its test subjects)
    summarize(values)                mean and standard deviation over runs (folds x seeds)
"""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from src.data.format import N_CLASSES, STAGES

LABELS = list(range(N_CLASSES))


def normalize_rows(counts):
    """Each row divided by its total: the diagonal is the recall of each stage. Empty rows: NaN."""
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, counts / totals, np.nan)


def compute_metrics(y_true, y_pred):
    """Accuracy, macro-F1, Cohen's kappa, F1 per stage and the confusion matrix.

    macro-F1 averages the F1 of the stages present in y_true or y_pred, as sklearn's default
    and the official code (D13). The F1 of a stage absent from both is NaN, not 0, so that
    it is skipped (not counted as an error) when runs are averaged.
    confusion[i][j]: epochs of true stage i predicted as stage j.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_stage = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0.0)
    present = np.isin(LABELS, np.union1d(y_true, y_pred))
    per_stage = np.where(present, per_stage, np.nan)
    counts = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "n_epochs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1": {stage: float(v) for stage, v in zip(STAGES, per_stage)},
        "confusion": counts.tolist(),
        "confusion_normalized": normalize_rows(counts).tolist(),
    }


def summarize(values):
    """Mean and standard deviation (n - 1) of one metric over runs; NaN values are skipped.

    With a single value the standard deviation is NaN: it is not defined.
    """
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return {"mean": float(values.mean()), "std": std, "n": int(len(values))}