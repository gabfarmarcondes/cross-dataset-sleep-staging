"""Unit tests for the evaluation module (issue 13). Run from the repository root:

    python -m pytest tests/
"""
import csv
import math

import numpy as np
import pytest
import torch
from sklearn.metrics import cohen_kappa_score, f1_score

from src.data.splits import load_splits
from src.evaluation.evaluate import evaluation_subjects, predict
from src.evaluation.metrics import compute_metrics, summarize
from src.evaluation.report import collect, mean_std, parse_group, pt
from src.models.tinysleepnet import EpochClassifier, TinySleepNet

CFG = {"batch_size": 15, "seq_length": 20, "epoch_batch_size": 100}


# --- metrics --------------------------------------------------------------------------------

def test_metrics_on_a_small_example():
    y_true = [0, 0, 1, 2, 2, 2, 4]
    y_pred = [0, 1, 1, 2, 2, 0, 4]  # N3 (3) appears in neither
    m = compute_metrics(y_true, y_pred)
    assert m["n_epochs"] == 7
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["kappa"] == pytest.approx(cohen_kappa_score(y_true, y_pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert math.isnan(m["f1"]["N3"])  # absent stage: skipped, not counted as 0
    assert m["f1"]["REM"] == 1.0
    assert m["confusion"][0] == [1, 1, 0, 0, 0]
    assert m["confusion_normalized"][0] == [0.5, 0.5, 0.0, 0.0, 0.0]
    assert all(math.isnan(v) for v in m["confusion_normalized"][3])


def test_summarize_skips_nan_and_uses_n_minus_1():
    s = summarize([0.8, 0.9, float("nan")])
    assert s["n"] == 2 and s["mean"] == pytest.approx(0.85)
    assert s["std"] == pytest.approx(np.std([0.8, 0.9], ddof=1))
    assert math.isnan(summarize([0.8])["std"])


def test_portuguese_number_format():
    assert pt(1516421, 0) == "1.516.421"
    assert mean_std({"mean": 0.854, "std": 0.012, "n": 3}, 100) == r"85,4 $\pm$ 1,2"
    assert mean_std({"mean": 0.854, "std": float("nan"), "n": 1}, 100) == "85,4"
    assert mean_std({"mean": 0.8, "std": 0.02, "n": 3}, 1, 2) == r"0,80 $\pm$ 0,02"


# --- which subjects are tested --------------------------------------------------------------

def test_groups_are_parsed_and_checked():
    g = parse_group("Sleep-EDF → ISRUC=sleep_edf:cross:isruc")
    assert (g["label"], g["dataset"], g["scenario"], g["target"]) == \
        ("Sleep-EDF → ISRUC", "sleep_edf", "cross", "isruc")
    for bad in ["x=sleep_edf:cross", "x=sleep_edf:in_domain:isruc", "x=sleep_edf:test", "sleep_edf"]:
        with pytest.raises(ValueError):
            parse_group(bad)


def test_evaluation_subjects_follow_the_frozen_splits():
    splits = load_splits("sleep_edf")
    in_domain = {"run": "r", "dataset": "sleep_edf", "split": {"fold": 2, "source": False, "subjects": None}}
    assert evaluation_subjects(in_domain) == ("sleep_edf", splits["in_domain"]["folds"][2]["test"])
    source = {"run": "r", "dataset": "sleep_edf", "split": {"fold": None, "source": True, "subjects": None}}
    assert evaluation_subjects(source, "isruc") == ("isruc", load_splits("isruc")["cross_dataset"]["target"]["test"])
    overfit = {"run": "r", "dataset": "sleep_edf", "split": {"fold": None, "source": False, "subjects": [0, 1]}}
    for info, target in [(in_domain, "isruc"), (source, None), (source, "sleep_edf"), (overfit, None)]:
        with pytest.raises(ValueError):
            evaluation_subjects(info, target)


# --- prediction -----------------------------------------------------------------------------

def test_predict_matches_one_pass_over_the_recording():
    """Sequence mode: minibatches of 20 with the state carried = one pass over the night."""
    torch.manual_seed(0)
    model = TinySleepNet().eval()
    recording = {"x": np.random.default_rng(0).standard_normal((45, 3000)).astype(np.float32),
                 "y": np.zeros(45, dtype=np.int64)}
    with torch.no_grad():
        whole = model(torch.from_numpy(recording["x"])[None], [45])[0][0].argmax(dim=-1).numpy()
    assert np.array_equal(predict(model, recording, "sequence", CFG, torch.device("cpu")), whole)
    epoch_model = EpochClassifier().eval()
    with torch.no_grad():
        expected = epoch_model(torch.from_numpy(recording["x"]).unsqueeze(1)).argmax(dim=-1).numpy()
    assert np.array_equal(predict(epoch_model, recording, "epoch", CFG, torch.device("cpu")), expected)


# --- in-domain reference on the target.test subjects ----------------------------------------

def test_in_domain_reference_pools_folds_on_target_test_subjects(tmp_path):
    """Correct predictions only for target.test subjects: the reference must be 100%."""
    splits = load_splits("sleep_edf")
    target_test = set(splits["cross_dataset"]["target"]["test"])
    runs = []
    for fold, part in enumerate(splits["in_domain"]["folds"]):
        run_dir = tmp_path / f"fold{fold}"
        run_dir.mkdir()
        (run_dir / "evaluation.json").write_text("{}", encoding="utf-8")  # already evaluated
        with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["recording_id", "subject_id", "epoch_idx", "y_true", "y_pred"])
            for subject in part["test"]:
                stage = subject % 5
                wrong = (stage + 1) % 5
                for e in range(10):
                    writer.writerow(["r", subject, e, stage, stage if subject in target_test else wrong])
        info = {"run": f"fold{fold}", "seed": 0, "split": {"fold": fold}}
        runs.append((run_dir, info))
    group = parse_group("ref=sleep_edf:in_domain_ref")
    entries = collect(group, runs, force=False, log=lambda message: None)
    assert len(entries) == 1 and entries[0]["values"]["accuracy"] == 1.0
    assert sum(map(sum, entries[0]["confusion"])) == 10 * len(target_test)

    skipped = collect(group, runs[:4], force=False, log=lambda message: None)  # a fold missing
    assert skipped == []