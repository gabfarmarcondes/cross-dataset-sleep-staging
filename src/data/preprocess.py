"""Signal operations: resampling (used by the loaders) and the load-time filter and scaling."""
from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from src.data.format import EPOCH_SAMPLES, FS


def resample_to_target(signal, fs_original):
    """Resample a 1-D signal to FS (100 Hz); resample_poly applies an anti-aliasing filter."""
    if fs_original == FS:
        return signal
    ratio = Fraction(FS, int(fs_original))  # 100/200 -> 1/2; 100/256 -> 25/64
    return resample_poly(signal, ratio.numerator, ratio.denominator)


def bandpass(signal, low=0.3, high=35.0, order=4):
    """Zero-phase Butterworth band-pass of a continuous 1-D signal sampled at FS."""
    sos = butter(order, [low, high], btype="bandpass", fs=FS, output="sos")
    # Extend each end by up to one epoch (mirrored) so the filter transient falls outside the signal
    return sosfiltfilt(sos, signal, padlen=min(EPOCH_SAMPLES, len(signal) - 1))


def contiguous_runs(epoch_idx):
    """(start, stop) positions of runs of consecutive epochs; a jump in epoch_idx ends a run."""
    breaks = np.flatnonzero(np.diff(epoch_idx) != 1) + 1
    edges = np.concatenate([[0], breaks, [len(epoch_idx)]])
    return list(zip(edges[:-1], edges[1:]))


def filter_epochs(x, epoch_idx, low=0.3, high=35.0, order=4):
    """Band-pass a recording [n_epochs, 3000], one continuous run of epochs at a time.

    Epochs on both sides of a gap (discarded epochs) were not adjacent in time,
    so the filter never runs across a gap.
    """
    out = np.empty(x.shape, dtype=np.float64)
    for start, stop in contiguous_runs(epoch_idx):
        run = x[start:stop].ravel()  # epochs back to back = the original continuous signal
        out[start:stop] = bandpass(run, low, high, order).reshape(stop - start, -1)
    return out


def robust_scale(x, clip=20.0):
    """Per-recording robust scaling: (x - median) / IQR, then clipping at +-clip."""
    q25, median, q75 = np.percentile(x, [25, 50, 75])
    iqr = q75 - q25
    if iqr <= 0:
        raise ValueError("interquartile range is zero (flat signal)")
    return np.clip((x - median) / iqr, -clip, clip)