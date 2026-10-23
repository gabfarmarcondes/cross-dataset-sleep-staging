"""Unit tests for the subject-wise splits (issue 9). Run from the repository root:

    python -m pytest tests/
"""
import json

import numpy as np
import pytest

from src.data.splits import SPLITS_DIR, pick_validation, save_splits, stratified_folds

SUBJECTS = list(range(2, 22))  # 20 subjects
WAKE = {s: (s * 37 % 100) / 100 for s in SUBJECTS}  # arbitrary, all different


def frozen_split_files():
    return sorted(SPLITS_DIR.glob("*.json"))


# --- the functions that build the splits ------------------------------------------------

def test_folds_cover_every_subject_once():
    folds = stratified_folds(SUBJECTS, WAKE, 5, np.random.default_rng(0))
    assert sorted(s for f in folds for s in f) == SUBJECTS
    assert [len(f) for f in folds] == [4] * 5


def test_folds_are_reproducible():
    a = stratified_folds(SUBJECTS, WAKE, 5, np.random.default_rng(0))
    b = stratified_folds(SUBJECTS, WAKE, 5, np.random.default_rng(0))
    assert a == b


def test_halves_separate_the_two_most_awake_subjects():
    top_two = sorted(SUBJECTS, key=WAKE.get)[-2:]
    for seed in range(10):
        test, adapt = stratified_folds(SUBJECTS, WAKE, 2, np.random.default_rng(seed))
        assert (top_two[0] in test) != (top_two[1] in test)


def test_validation_is_ten_percent_and_disjoint():
    train, valid = pick_validation(SUBJECTS[:16], np.random.default_rng(0))
    assert len(valid) == 2 and not set(train) & set(valid)
    assert sorted(train + valid) == SUBJECTS[:16]


def test_frozen_file_is_not_overwritten(tmp_path):
    splits = {"dataset": "demo"}
    save_splits(splits, tmp_path)
    with pytest.raises(FileExistsError):
        save_splits(splits, tmp_path)


# --- the frozen files in configs/splits (acceptance criterion) --------------------------

@pytest.mark.parametrize("path", frozen_split_files(), ids=lambda p: p.stem)
def test_no_subject_in_two_partitions(path):
    splits = json.loads(path.read_text(encoding="utf-8"))
    subjects = sorted(int(s) for s in splits["recordings"])

    def check(*partitions):
        joined = [s for p in partitions for s in p]
        assert len(joined) == len(set(joined)), "a subject appears in two partitions"
        assert sorted(joined) == subjects, "a subject is missing"

    tested = []
    for fold in splits["in_domain"]["folds"]:
        check(fold["train"], fold["valid"], fold["test"])
        tested += fold["test"]
    assert sorted(tested) == subjects, "every subject must be tested exactly once across folds"

    check(splits["cross_dataset"]["source"]["train"], splits["cross_dataset"]["source"]["valid"])
    check(splits["cross_dataset"]["target"]["adapt"], splits["cross_dataset"]["target"]["test"])

    recordings = [r for rs in splits["recordings"].values() for r in rs]
    assert len(recordings) == len(set(recordings)), "a recording is assigned to two subjects"