"""Recordings in the common format, with the load-time preprocessing (D7, D8).

The saved .npz files are never changed: the band-pass filter and the robust
scaling are applied here, in memory. The same files therefore give both
conditions of the filtering ablation, switched by `filter` in the config:

    preprocessing:
      filter: true          # band-pass 0.3-35 Hz on/off (D7)
      band: [0.3, 35.0]
      filter_order: 4
      clip: 20.0            # robust scaling, clip at +-20 IQR (D8)

    recordings = SleepRecordings(paths, **cfg["preprocessing"])
"""
from pathlib import Path

import numpy as np

from src.data.format import validate_npz
from src.data.preprocess import filter_epochs, robust_scale
from src.read_data import ROOT, load_config

META_KEYS = ("dataset", "recording_id", "subject_id", "channel")


def recording_paths(dataset):
    """All saved .npz files of one dataset, in name order (the splits will list them explicitly)."""
    _, paths = load_config()
    return sorted((ROOT / paths["processed_root"] / dataset).glob("*.npz"))


class SleepRecordings:
    """Preprocessed recordings; item i is a dict with x [n_epochs, 3000], y, epoch_idx and metadata."""

    def __init__(self, paths, filter=True, band=(0.3, 35.0), filter_order=4, clip=20.0):
        self.filter = filter
        self.recordings = [self._load(Path(p), filter, band, filter_order, clip) for p in paths]

    @staticmethod
    def _load(path, filter, band, order, clip):
        validate_npz(path)
        with np.load(path, allow_pickle=False) as f:
            x, y, epoch_idx = f["x"], f["y"], f["epoch_idx"]
            meta = {key: f[key].item() for key in META_KEYS}
        if filter:
            x = filter_epochs(x, epoch_idx, band[0], band[1], order)  # filter first, then scale
        x = robust_scale(x, clip).astype(np.float32)
        return {"x": x, "y": y, "epoch_idx": epoch_idx, **meta}

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, i):
        return self.recordings[i]