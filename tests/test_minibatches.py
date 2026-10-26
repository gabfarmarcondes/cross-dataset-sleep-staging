"""Unit tests for the sequence minibatches (issue 11). Run from the repository root:

    python -m pytest tests/
"""
import numpy as np
import torch

from src.models.tinysleepnet import TinySleepNet
from src.training.minibatches import PAD_LABEL, sequence_minibatches


def fake_recordings(n_epochs, n_samples=4):
    rng = np.random.default_rng(0)
    return [{"x": rng.standard_normal((n, n_samples)).astype(np.float32),
             "y": rng.integers(0, 5, n)} for n in n_epochs]


def test_lengths_and_reset_flags():
    """Groups of 2 recordings: [45, 20] gives 3 minibatches, then [3] alone gives 1."""
    batches = list(sequence_minibatches(fake_recordings([45, 20, 3]), batch_size=2, seq_length=20))
    assert [b[2].tolist() for b in batches] == [[20, 20], [20, 0], [5, 0], [3, 0]]
    assert [b[3] for b in batches] == [True, False, False, True]
    x, y, lengths, _ = batches[0]
    assert x.shape == (2, 20, 4) and x.dtype == np.float32
    assert y.shape == (2, 20) and y.dtype == np.int64


def test_rows_rebuild_each_recording():
    """Following a row across minibatches gives back its recording; padding is 0 / PAD_LABEL."""
    recordings = fake_recordings([45, 20, 3])
    order = [2, 0, 1]
    rows = {i: ([], []) for i in range(len(recordings))}
    group = -1
    for x, y, lengths, reset in sequence_minibatches(recordings, 2, 20, order):
        group += reset
        for row, rec in enumerate(order[2 * group:2 * group + 2]):
            rows[rec][0].append(x[row, :lengths[row]])
            rows[rec][1].append(y[row, :lengths[row]])
        padded = np.arange(20)[None, :] >= lengths[:, None]
        assert np.all(x[padded] == 0) and np.all(y[padded] == PAD_LABEL)
    for i, recording in enumerate(recordings):
        assert np.array_equal(np.concatenate(rows[i][0]), recording["x"])
        assert np.array_equal(np.concatenate(rows[i][1]), recording["y"])


def test_stateful_minibatches_equal_whole_recordings():
    """Minibatches + carried state + reset = each recording processed in one long pass."""
    torch.manual_seed(0)
    model = TinySleepNet().eval()
    recordings = fake_recordings([45, 23, 7], n_samples=3000)
    order, batch_size = [2, 0, 1], 2
    collected = {i: [] for i in range(len(recordings))}
    group = -1
    with torch.no_grad():
        for x, _, lengths, reset in sequence_minibatches(recordings, batch_size, 20, order):
            if reset:
                group += 1
                state = None
            logits, state = model(torch.from_numpy(x), lengths, state)
            for row, rec in enumerate(order[group * batch_size:(group + 1) * batch_size]):
                collected[rec].append(logits[row, :lengths[row]])
        for i, recording in enumerate(recordings):
            whole, _ = model(torch.from_numpy(recording["x"])[None], [len(recording["y"])])
            assert torch.allclose(torch.cat(collected[i]), whole[0], atol=1e-5)