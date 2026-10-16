"""Sleep-EDF Expanded (SC study) loader: raw EDF files -> common .npz format.

Writes one .npz per recording with save_recording() (see docs/data_format.md)
and logs the class counts of every recording. Run from the repository root:

"""
import logging
from collections import Counter

import mne
import numpy as np

from src.data.format import EPOCH_SAMPLES, FS, N_CLASSES, STAGES, save_recording
from src.read_data import ROOT, load_config, pick_channel, processed_path, validate_dataset

DATASET = "sleep_edf"
EPOCH_SEC = 30
SLEEP_LABELS = [1, 2, 3, 4]   # N1, N2, N3, REM (W=0 is not sleep)
DISCARD = -1                  # label of epochs that are not kept (M, ?, unscored)
UNSCORED = "unscored"         # description of signal after the last annotation

log = logging.getLogger(__name__)


def edf_start(path):
    """Start date and time from the EDF header (bytes 168-183, "dd.mm.yyhh.mm.ss")."""
    with open(path, "rb") as f:
        header = f.read(184)
    return header[168:184].decode("ascii")


def epoch_descriptions(annotations, n_epochs):
    """Hypnogram description of each 30-s epoch of the signal.

    Annotations must be contiguous from the start of the recording and last a
    multiple of 30 s. Annotations past the end of the signal are cut; epochs
    after the last annotation are marked UNSCORED.
    """
    desc = np.full(n_epochs, UNSCORED, dtype=object)
    expected_onset = 0.0
    for onset, duration, text in zip(annotations.onset, annotations.duration, annotations.description):
        if onset != expected_onset or duration % EPOCH_SEC != 0:
            raise ValueError(f"annotation {text!r} at {onset} s ({duration} s) is off the 30-s grid")
        start = int(onset) // EPOCH_SEC
        desc[start:start + int(duration) // EPOCH_SEC] = text  # a slice past the end is cut
        expected_onset = onset + duration
    return desc


def to_labels(desc, mapping):
    """Common labels (0=W ... 4=REM) of each epoch; DISCARD where the mapping is null."""
    labels = [mapping.get(d) for d in desc]
    return np.array([DISCARD if label is None else label for label in labels], dtype=np.int64)


def sleep_window(labels, minutes):
    """First and one-past-last epoch to keep: `minutes` of wake around the sleep period."""
    sleep = np.flatnonzero(np.isin(labels, SLEEP_LABELS))
    if sleep.size == 0:
        raise ValueError("no sleep epochs")
    edge = minutes * 60 // EPOCH_SEC
    return max(sleep[0] - edge, 0), min(sleep[-1] + edge + 1, len(labels))


def prepare_recording(psg, hyp, subject_id, cfg):
    """Convert one PSG/hypnogram pair to the common format; returns validate_npz's summary."""
    if edf_start(psg) != edf_start(hyp):
        raise ValueError(f"PSG and {hyp.name} start at different times")

    # Signal: one channel, in microvolts, cut into 30-s epochs
    raw = mne.io.read_raw_edf(psg, preload=False, verbose="error")
    channel = pick_channel(raw.ch_names, cfg["channel_names"])
    fs = raw.info["sfreq"]
    if fs != FS:
        raise ValueError(f"expected {FS} Hz, got {fs} Hz")
    signal = raw.get_data(picks=[channel], units="uV")[0]
    n_epochs = len(signal) // EPOCH_SAMPLES
    epochs = signal[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)

    # Labels: one per epoch, aligned with the signal by the annotation onsets
    annotations = mne.read_annotations(hyp)
    mapping = cfg["labels"]["mapping"]
    unknown = set(annotations.description) - set(mapping)
    if unknown:
        raise ValueError(f"hypnogram descriptions not in datasets.yaml: {sorted(unknown)}")
    desc = epoch_descriptions(annotations, n_epochs)
    labels = to_labels(desc, mapping)

    # Keep the sleep window, then drop discarded epochs inside it
    start, stop = sleep_window(labels, cfg["trim_wake_minutes"])
    window = np.arange(start, stop)
    discarded = labels[window] == DISCARD
    keep = window[~discarded]
    dropped = Counter(desc[window[discarded]])

    recording_id = psg.name[:6]  # SC4ssN: subject ss, night N
    summary = save_recording(
        processed_path(DATASET, recording_id),
        x=epochs[keep], y=labels[keep], epoch_idx=keep,
        fs_original=fs, channel=channel, subject_id=subject_id,
        recording_id=recording_id, dataset=DATASET,
    )
    log.info(
        "%s: kept %d of %d epochs | trimmed %d | dropped %s | %s",
        recording_id, summary["n_epochs"], n_epochs, n_epochs - len(window),
        dict(dropped) or "none", dict(zip(STAGES, summary["class_counts"])),
    )
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

    files = validate_dataset(DATASET)  # {subject: [(psg, hypnogram), ...]}; checks the 39 recordings
    totals = np.zeros(N_CLASSES, dtype=np.int64)
    n_recordings = 0
    for subject_id, pairs in files.items():
        for psg, hyp in pairs:
            try:
                summary = prepare_recording(psg, hyp, subject_id, cfg)
            except ValueError as e:
                raise ValueError(f"{psg.name}: {e}") from e  # say which recording failed
            totals += summary["class_counts"]
            n_recordings += 1

    log.info("TOTAL: %d recordings, %d epochs | %s",
             n_recordings, totals.sum(), dict(zip(STAGES, totals.tolist())))


if __name__ == "__main__":
    main()