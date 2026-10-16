"""ISRUC-Sleep (Subgroup I) loader: raw .rec files -> common .npz format.

Only N/N.rec (signal) and N/N_1.txt (expert 1) are opened; the subject folders
are never listed. Run from the repository root:
"""
import logging
from collections import Counter

import mne
import numpy as np

from src.data.format import EPOCH_SAMPLES, N_CLASSES, STAGES, save_recording
from src.data.preprocess import resample_to_target
from src.read_data import (ROOT, as_edf, load_config, pick_channel, processed_path,
                           read_isruc_labels, validate_dataset)

DATASET = "isruc"
EPOCH_SEC = 30

log = logging.getLogger(__name__)


def prepare_recording(rec, labels_path, subject_id, cfg):
    """Convert one ISRUC recording to the common format; returns validate_npz's summary."""
    # Signal: a .rec file is an EDF file with another extension, so it is read
    # through a temporary .edf copy. Only the channel names, the sampling rate
    # and the data are used; the header's patient fields are never logged.
    with as_edf(str(rec)) as edf:
        raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
        channel = pick_channel(raw.ch_names, cfg["channel_names"])
        fs = raw.info["sfreq"]
        signal = raw.get_data(picks=[channel], units="uV")[0]  # read before the copy is deleted
    if fs != cfg["sampling_rate"]:
        raise ValueError(f"expected {cfg['sampling_rate']} Hz, got {fs} Hz")
    n_epochs = len(signal) // int(EPOCH_SEC * fs)

    # Labels: expert 1, one per epoch, starting at the beginning of the recording
    codes = read_isruc_labels(labels_path)  # blank lines are skipped
    if len(codes) != n_epochs:
        raise ValueError(f"{n_epochs} epochs in the signal but {len(codes)} labels")
    mapping = cfg["labels"]["mapping"]
    unknown = set(codes) - set(mapping)
    if unknown:
        raise ValueError(f"label codes not in datasets.yaml: {sorted(unknown)}")
    y = np.array([mapping[code] for code in codes], dtype=np.int64)

    # 200 Hz -> 100 Hz on the continuous signal, then cut into 30-s epochs
    x = resample_to_target(signal, fs)[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)

    recording_id = f"isruc_{subject_id}"
    summary = save_recording(
        processed_path(DATASET, recording_id),
        x=x, y=y, epoch_idx=np.arange(n_epochs),
        fs_original=fs, channel=channel, subject_id=subject_id,
        recording_id=recording_id, dataset=DATASET,
    )
    log.info(
        "%s: %d epochs | %s | median |x| %.1f uV | %s",
        recording_id, n_epochs, channel, np.median(np.abs(x)),
        dict(zip(STAGES, summary["class_counts"])),
    )
    summary["channel"] = channel
    return summary


def main():
    datasets, _ = load_config()
    cfg = datasets[DATASET]

    log_path = ROOT / "outputs" / "logs" / f"prepare_{DATASET}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
    )

    files = validate_dataset(DATASET)  # {subject: [(rec, labels)]}; checks the 20 recordings
    totals = np.zeros(N_CLASSES, dtype=np.int64)
    channels = Counter()
    for subject_id, pairs in files.items():
        for rec, labels_path in pairs:
            try:
                summary = prepare_recording(rec, labels_path, subject_id, cfg)
            except ValueError as e:
                raise ValueError(f"ISRUC subject {subject_id}: {e}") from e  # say which one failed
            totals += summary["class_counts"]
            channels[summary["channel"]] += 1

    log.info("TOTAL: %d recordings, %d epochs | %s | channels %s",
             sum(channels.values()), totals.sum(), dict(zip(STAGES, totals.tolist())), dict(channels))


if __name__ == "__main__":
    main()