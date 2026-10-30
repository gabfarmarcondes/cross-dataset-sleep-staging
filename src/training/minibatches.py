"""Minibatches for the two training modes (issues 11 and 15).

Sequence mode: sequence_minibatches, minibatches of epoch sequences for the stateful LSTM
of TinySleepNet. Same scheme as iterate_batch_multiple_seq_minibatches in the official code
(https://github.com/akaraspt/tinysleepnet, minibatching.py):

- recordings are taken batch_size (15) at a time;
- row i of each minibatch follows recording i of the group, seq_length (20) consecutive
  epochs per minibatch, so the LSTM state of a row can be carried to the next minibatch;
- the group goes on until its longest recording ends; a shorter recording leaves its row
  partly filled, then empty (length 0);
- the first minibatch of each group is flagged with reset=True: start the state from zeros.

    for x, y, lengths, reset in sequence_minibatches(recordings, order=rng.permutation(n)):
        if reset:
            state = None
        logits, state = model(torch.from_numpy(x), lengths, state)

The random skip of the first epochs (augment_seq in the official code) is applied to the
recordings before they get here (src/training/augment.py).

Epoch mode: epoch_minibatches, minibatches of single epochs taken from the recordings
(no concatenated copy of the data, so that each recording can be shifted on its own).
"""
import math

import numpy as np

BATCH_SIZE = 15
SEQ_LENGTH = 20
PAD_LABEL = -1  # label of padded positions; the loss must mask them


def sequence_minibatches(recordings, batch_size=BATCH_SIZE, seq_length=SEQ_LENGTH, order=None):
    """Yield (x, y, lengths, reset) for the stateful LSTM.

    recordings: items with "x" [n_epochs, n_samples] and "y" [n_epochs], e.g. SleepRecordings.
    order: recording indices in the order to use (a permutation for training); None keeps
    the given order (validation and test).

        x        float32 [batch_size, seq_length, n_samples], zeros after each row's length
        y        int64 [batch_size, seq_length], PAD_LABEL after each row's length
        lengths  int64 [batch_size], real epochs in each row (0 to seq_length)
        reset    True on the first minibatch of a new group of recordings
    """
    order = list(range(len(recordings))) if order is None else list(order)
    for start in range(0, len(order), batch_size):
        group = [recordings[i] for i in order[start:start + batch_size]]
        n_samples = group[0]["x"].shape[1]
        n_chunks = math.ceil(max(len(r["y"]) for r in group) / seq_length)
        for chunk in range(n_chunks):
            part = slice(chunk * seq_length, (chunk + 1) * seq_length)
            x = np.zeros((batch_size, seq_length, n_samples), dtype=np.float32)
            y = np.full((batch_size, seq_length), PAD_LABEL, dtype=np.int64)
            lengths = np.zeros(batch_size, dtype=np.int64)
            for row, recording in enumerate(group):
                labels = recording["y"][part]
                n = len(labels)
                x[row, :n] = recording["x"][part]
                y[row, :n] = labels
                lengths[row] = n
            yield x, y, lengths, chunk == 0


def epoch_minibatches(recordings, batch_size, order=None):
    """Yield (x, y, None, None) with single epochs, for the CNN alone.

    The epochs are numbered as if the recordings were concatenated in the given order.
    order: epoch numbers in the order to use (a permutation for training); None keeps
    every epoch in place (validation and test).

        x  float32 [<= batch_size, n_samples]
        y  int64 [<= batch_size]
    """
    sizes = [len(r["y"]) for r in recordings]
    which = np.repeat(np.arange(len(recordings)), sizes)  # recording of each epoch
    where = np.concatenate([np.arange(n) for n in sizes])  # position in that recording
    order = np.arange(len(which)) if order is None else np.asarray(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        pairs = list(zip(which[idx], where[idx]))
        x = np.stack([recordings[r]["x"][e] for r, e in pairs]).astype(np.float32, copy=False)
        y = np.array([recordings[r]["y"][e] for r, e in pairs], dtype=np.int64)
        yield x, y, None, None