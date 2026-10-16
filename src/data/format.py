"""Common preprocessed format: one .npz file per recording.

Every dataset loader writes recordings with save_recording(), and every model
reads them through the same keys, so no code after preprocessing depends on the
dataset. The format is described in docs/data_format.md.
"""
from pathlib import Path

import numpy as np

FS = 100                         # sampling rate after resampling (Hz)
EPOCH_SAMPLES = 30 * FS          # one 30-s epoch = 3000 samples
N_CLASSES = 5                    # 0=W, 1=N1, 2=N2, 3=N3, 4=REM
STAGES = ("W", "N1", "N2", "N3", "REM")  # stage name of each label
DATASETS = {"sleep_edf", "isruc", "hmc"}

ARRAY_KEYS = {"x", "y", "epoch_idx"}
SCALAR_KEYS = {"fs", "fs_original", "channel", "subject_id", "recording_id", "dataset"}


def save_recording(path, x, y, epoch_idx, fs_original, channel, subject_id, recording_id, dataset):
    """Write one recording in the common format and validate the written file.

    x: signal in microvolts, shape [n_epochs, 3000], already resampled to 100 Hz
       (anti-aliasing included) and NOT band-pass filtered or normalized.
    y: labels in the common scheme (0=W, 1=N1, 2=N2, 3=N3, 4=REM), shape [n_epochs].
    epoch_idx: position of each epoch in the original recording, increasing;
       gaps mark discarded epochs (e.g. movement time, unscored).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        epoch_idx=np.asarray(epoch_idx, dtype=np.int64),
        fs=np.int64(FS),
        fs_original=np.float64(fs_original),
        channel=str(channel),
        subject_id=np.int64(subject_id),
        recording_id=str(recording_id),
        dataset=str(dataset),
    )
    try:
        return validate_npz(path)
    except ValueError:
        path.unlink()  # never leave an invalid file behind
        raise


def validate_npz(path):
    """Check that a .npz file follows the common format.

    Raises ValueError with the first problem found; returns a short summary
    (number of epochs and epochs per class) when the file is valid.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        missing = (ARRAY_KEYS | SCALAR_KEYS) - set(f.files)
        if missing:
            raise ValueError(f"{path.name}: missing keys {sorted(missing)}")

        x, y, idx = f["x"], f["y"], f["epoch_idx"]

        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != EPOCH_SAMPLES:
            raise ValueError(f"{path.name}: x must be float32 [n_epochs, {EPOCH_SAMPLES}], got {x.dtype} {x.shape}")
        if x.shape[0] == 0:
            raise ValueError(f"{path.name}: no epochs")
        if not np.isfinite(x).all():
            raise ValueError(f"{path.name}: x contains NaN or inf")
        # MNE returns volts; a typical EEG in volts has a median amplitude around 1e-5
        if np.median(np.abs(x)) < 1e-2:
            raise ValueError(f"{path.name}: x looks like volts; store microvolts")

        if y.dtype != np.int64 or y.shape != (x.shape[0],):
            raise ValueError(f"{path.name}: y must be int64 [{x.shape[0]}], got {y.dtype} {y.shape}")
        if y.min() < 0 or y.max() >= N_CLASSES:
            raise ValueError(f"{path.name}: labels must be in 0..{N_CLASSES - 1}, got {y.min()}..{y.max()}")

        if idx.dtype != np.int64 or idx.shape != y.shape:
            raise ValueError(f"{path.name}: epoch_idx must be int64 with the same shape as y")
        if np.any(np.diff(idx) <= 0):
            raise ValueError(f"{path.name}: epoch_idx must be strictly increasing (chronological order)")

        if int(f["fs"]) != FS:
            raise ValueError(f"{path.name}: fs must be {FS}, got {int(f['fs'])}")
        if str(f["dataset"]) not in DATASETS:
            raise ValueError(f"{path.name}: unknown dataset {str(f['dataset'])!r}")

        return {
            "n_epochs": int(x.shape[0]),
            "class_counts": np.bincount(y, minlength=N_CLASSES).tolist(),
        }