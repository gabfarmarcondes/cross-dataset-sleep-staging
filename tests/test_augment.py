"""Unit tests for the data augmentation (issue 15). Run from the repository root:

    python -m pytest tests/
"""
import numpy as np
import pytest

from src.training.augment import augment, is_augmented, random_shift, shift_signal, without_augmentation
from src.training.loop import minibatches
from src.training.minibatches import PAD_LABEL

N_SAMPLES = 3000
CFG = {"batch_size": 4, "seq_length": 20, "epoch_batch_size": 32, "signal_shift": 0.1, "sequence_skip": 4}


def numbered_recording(n_epochs, n_samples=N_SAMPLES):
    """Each sample holds its position in the night, and each label its epoch number."""
    x = np.arange(n_epochs * n_samples, dtype=np.float64).reshape(n_epochs, n_samples)
    return {"x": x, "y": np.arange(n_epochs)}


def official_shift(x, y, offset):
    """train.py of the official code (augment_signal_full), on its [n, 3000, 1, 1] arrays."""
    x = x[:, :, None, None]
    roll_x = np.roll(x, int(offset))
    if offset < 0:
        x, y = roll_x[:-1], y[:-1]
    if offset > 0:
        x, y = roll_x[1:], y[1:]
    return x[:, :, 0, 0], y


# --- signal shift ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [-299, -150, -1, 1, 7, 299])
def test_shift_equals_the_official_roll(shift):
    r = numbered_recording(12)
    x, y = shift_signal(r["x"], r["y"], shift)
    expected_x, expected_y = official_shift(r["x"], r["y"], shift)
    assert np.array_equal(x, expected_x) and np.array_equal(y, expected_y)


@pytest.mark.parametrize("shift", [-299, -1, 1, 299])
def test_shifted_epochs_keep_their_labels(shift):
    """Every shifted epoch is one continuous piece of the night, at least 90% of it from the
    epoch its label names; the epoch with samples wrapped from the other end is gone."""
    r = numbered_recording(12)
    x, y = shift_signal(r["x"], r["y"], shift)
    assert len(x) == len(y) == 11
    for epoch, label in zip(x, y):
        assert np.all(np.diff(epoch) == 1)  # no jump: nothing wrapped around
        own = np.mean(epoch // N_SAMPLES == label)  # share of samples from the labelled epoch
        assert own == pytest.approx(1 - abs(shift) / N_SAMPLES)
        assert own >= 0.9


def test_shift_is_a_view_and_zero_changes_nothing():
    r = numbered_recording(5)
    x, _ = shift_signal(r["x"], r["y"], 120)
    assert np.shares_memory(x, r["x"])  # no copy of the night at every training epoch
    x, y = shift_signal(r["x"], r["y"], 0)
    assert x is r["x"] and y is r["y"]
    with pytest.raises(ValueError):
        shift_signal(r["x"], r["y"], N_SAMPLES)


def test_random_draws_cover_the_official_ranges():
    rng = np.random.default_rng(0)
    shifts = [random_shift(rng, 0.1, N_SAMPLES) for _ in range(5000)]
    assert max(map(abs, shifts)) <= 299 and min(shifts) < -250 and max(shifts) > 250
    r = numbered_recording(10, n_samples=4)
    skips = [10 - len(a["y"]) for a in (augment([r], rng, 0.0, 4)[0] for _ in range(500))]
    assert set(skips) == {0, 1, 2, 3, 4}


# --- augment() and the training minibatches -------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_augment_shifts_then_skips(seed):
    r = numbered_recording(10, n_samples=100)
    out = augment([r], np.random.default_rng(seed), 0.1, 4)[0]
    rng = np.random.default_rng(seed)  # the same draws, done by hand
    x, y = shift_signal(r["x"], r["y"], random_shift(rng, 0.1, 100))
    n_skip = int(rng.integers(0, 5))
    assert np.array_equal(out["x"], x[n_skip:]) and np.array_equal(out["y"], y[n_skip:])


def test_augmentation_off_returns_the_data_and_draws_nothing():
    recordings = [numbered_recording(6), numbered_recording(4)]
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    out = augment(recordings, rng, 0.0, 0)
    assert rng.bit_generator.state == before  # same shuffling as before issue 15
    assert all(o["x"] is r["x"] and o["y"] is r["y"] for o, r in zip(out, recordings))
    assert without_augmentation(CFG)["signal_shift"] == 0 and without_augmentation(CFG)["sequence_skip"] == 0
    assert is_augmented(CFG, "epoch") and is_augmented({**CFG, "signal_shift": 0.0}, "sequence")
    assert not is_augmented({**CFG, "signal_shift": 0.0}, "epoch")
    assert not is_augmented(without_augmentation(CFG), "sequence")


def real_labels(batches):
    return np.concatenate([y[y != PAD_LABEL] for _, y, _, _ in batches])


@pytest.mark.parametrize("mode", ["epoch", "sequence"])
def test_only_training_is_augmented(mode):
    """Without rng (validation, test) every epoch comes once, unchanged; with rng (training)
    each recording loses one edge epoch (shift) and, in sequence mode, up to 4 more (skip)."""
    recordings = [numbered_recording(n, n_samples=50) for n in (45, 30, 23)]
    total = sum(len(r["y"]) for r in recordings)
    labels = np.sort(np.concatenate([r["y"] for r in recordings]))
    for _ in range(3):  # evaluation never changes
        assert np.array_equal(np.sort(real_labels(minibatches(recordings, mode, CFG))), labels)

    rng = np.random.default_rng(0)
    counts = [len(real_labels(minibatches(recordings, mode, CFG, rng))) for _ in range(30)]
    most_lost = len(recordings) * (1 + (4 if mode == "sequence" else 0))
    assert all(total - most_lost <= c <= total for c in counts)
    assert len(set(counts)) > 1  # drawn anew at every training epoch
    assert all(len(r["y"]) == n for r, n in zip(recordings, (45, 30, 23)))  # data left intact


@pytest.mark.parametrize("mode", ["epoch", "sequence"])
def test_training_minibatches_follow_the_seed(mode):
    recordings = [numbered_recording(n, n_samples=50) for n in (45, 30, 23)]

    def first_epoch(seed):
        return [x for x, _, _, _ in minibatches(recordings, mode, CFG, np.random.default_rng(seed))]

    a, b, c = first_epoch(1), first_epoch(1), first_epoch(2)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    assert not all(p.shape == q.shape and np.array_equal(p, q) for p, q in zip(a, c))


def test_every_training_epoch_is_a_window_of_the_night():
    """Epoch mode: each training epoch is 50 consecutive samples of one recording, labelled
    with the epoch most of them come from (labels stay aligned after shuffling)."""
    recordings = [numbered_recording(n, n_samples=50) for n in (45, 30)]
    recordings[1]["x"] = recordings[1]["x"] + 1e6  # tell the two recordings apart
    for x, y, _, _ in minibatches(recordings, "epoch", CFG, np.random.default_rng(0)):
        assert np.all(np.diff(x, axis=1) == 1)
        position = x[:, 25] % 1e6  # middle sample, position in its night
        assert np.array_equal(position // 50, y)
