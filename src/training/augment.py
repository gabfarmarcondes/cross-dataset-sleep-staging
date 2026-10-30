"""Data augmentation of the training recordings (issue 15), as in the official TinySleepNet
code (https://github.com/akaraspt/tinysleepnet, train.py and minibatching.py).

Drawn anew at every training epoch, on the training recordings only (validation and test
recordings are never changed):

signal shift (augment_signal_full in the official code)
    the whole night is moved by a random offset of up to +-10% of an epoch (+-300 samples
    at 100 Hz), across epoch boundaries; the epoch at the edge that would receive samples
    wrapped around from the other end of the night is dropped, so every remaining epoch
    keeps its label (at least 90% of it comes from the epoch the label belongs to);
sequence skip (augment_seq), sequence mode only
    0 to 4 epochs are skipped at the start of each recording, so the 20-epoch sequences of
    the stateful LSTM are cut at different places from one training epoch to the next.

Settings, in the `training` block of configs/experiment.yaml (0 turns one off):

    signal_shift: 0.1    # largest offset, as a fraction of an epoch
    sequence_skip: 4     # largest number of epochs skipped at the start

Like the official code, the shift treats the epochs of a recording as one continuous signal,
also where discarded epochs left a gap (the sequence minibatches do the same).
"""


def shift_signal(x, y, shift):
    """Recording x [n_epochs, n_samples] with its signal moved by `shift` samples; labels y.

    shift > 0 moves the signal later: each epoch starts with the last `shift` samples of
    the epoch before it, and the first epoch is dropped. shift < 0 moves it earlier: each
    epoch ends with the first samples of the next one, and the last epoch is dropped.
    Same result as the official np.roll of the whole night followed by dropping the epoch
    with wrapped samples, but as a view: nothing is copied.
    """
    n_epochs, n_samples = x.shape
    if shift == 0:
        return x, y
    if not -n_samples < shift < n_samples:
        raise ValueError(f"shift must be smaller than one epoch ({n_samples} samples), got {shift}")
    start = n_samples - shift if shift > 0 else -shift  # first sample of the first kept epoch
    flat = x.reshape(-1)
    shifted = flat[start:start + (n_epochs - 1) * n_samples].reshape(n_epochs - 1, n_samples)
    return shifted, (y[1:] if shift > 0 else y[:-1])


def random_shift(rng, max_fraction, n_samples):
    """Offset in samples, as drawn by the official code: uniform in +-max_fraction of an
    epoch, truncated towards zero (so at most 299 samples for 0.1 of 3000)."""
    return int(rng.uniform(-max_fraction, max_fraction) * n_samples)


def augment(recordings, rng, signal_shift=0.0, sequence_skip=0):
    """Augmented versions of the recordings for one training epoch: dicts with x and y only.

    The shift comes first, then the skip, as in the official code. With both settings at 0
    the recordings come back unchanged and rng is not used.
    """
    augmented = []
    for recording in recordings:
        x, y = recording["x"], recording["y"]
        if signal_shift > 0:
            x, y = shift_signal(x, y, random_shift(rng, signal_shift, x.shape[1]))
        if sequence_skip > 0:
            n_skip = int(rng.integers(0, sequence_skip + 1))  # 0 to sequence_skip, as randint(5)
            x, y = x[n_skip:], y[n_skip:]
        augmented.append({"x": x, "y": y})
    return augmented


def without_augmentation(training):
    """The training settings with both augmentations turned off (--no-augment)."""
    return {**training, "signal_shift": 0.0, "sequence_skip": 0}


def is_augmented(training, mode):
    """Whether training in this mode changes the data (the skip is used in sequence mode only)."""
    return float(training["signal_shift"]) > 0 or (mode == "sequence" and int(training["sequence_skip"]) > 0)
