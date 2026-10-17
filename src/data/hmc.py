"""HMC Sleep Staging Database loader: raw EDF files -> common .npz format.

Writes one .npz per recording with save_recording() (see docs/data_format.md)
and logs class counts plus the lights off/on times of every recording.
Run from the repository root:
"""
import logging

import mne
import numpy as np

from src.data.edf_header import start_time
from src.data.format import EPOCH_SAMPLES, FS, N_CLASSES, STAGES, save_recording
from src.data.preprocess import resample_to_target
from src.read_data import ROOT, load_config, pick_channel, processed_path, validate_dataset

DATASET = "hmc"
EPOCH_SEC = 30
LIGHTS = {"Lights off", "Lights on"}  # events in the scoring file, not sleep stages

log = logging.getLogger(__name__)


def first(values):
    """First value of an array as a float, or None if the array is empty."""
    return float(values[0]) if values.size else None


def prepare_recording(psg, hyp, subject_id, cfg):
    """Convert one PSG/scoring pair to the common format; returns validate_npz's summary."""
    # Signal: one channel, in microvolts, resampled 256 -> 100 Hz as a whole
    raw = mne.io.read_raw_edf(psg, preload=False, verbose="error")
    channel = pick_channel(raw.ch_names, cfg["channel_names"])
    fs = raw.info["sfreq"]
    if fs != cfg["sampling_rate"]:
        raise ValueError(f"expected {cfg['sampling_rate']} Hz, got {fs} Hz")
    signal = resample_to_target(raw.get_data(picks=[channel], units="uV")[0], fs)
    signal_sec = raw.n_times / fs

    # Labels: one per stage annotation; lights off/on are kept apart.
    # Onsets count from the start of the scoring file; the header start times
    # give the shift to seconds from the start of the PSG.
    annotations = mne.read_annotations(hyp)
    offset = (start_time(hyp) - start_time(psg)).total_seconds()
    onsets_all = annotations.onset + offset
    # EDF+ can tie an annotation to a channel as "text@@channel" (HMC does it for the
    # lights events); only the text matters here.
    descriptions = np.array([d.split("@@")[0] for d in annotations.description])
    mapping = cfg["labels"]["mapping"]
    unknown = set(descriptions) - set(mapping) - LIGHTS
    if unknown:
        raise ValueError(f"scoring descriptions not in datasets.yaml: {sorted(unknown)}")
    is_stage = np.array([mapping.get(d) is not None for d in descriptions])
    onsets, durations = onsets_all[is_stage], annotations.duration[is_stage]
    ends = onsets + durations
    if np.any(durations % EPOCH_SEC != 0) or np.any(onsets[1:] != ends[:-1]):
        raise ValueError("stage annotations are not contiguous 30-s epochs")
    epochs_per_annotation = (durations // EPOCH_SEC).astype(int)
    y = np.repeat([mapping[d] for d in descriptions[is_stage]], epochs_per_annotation).astype(np.int64)
    n_epochs = len(y)

    # Cut the scored period: from the first stage onset, one 30-s epoch per label
    start = onsets[0] * FS
    if start != int(start):
        raise ValueError(f"first stage onset {onsets[0]} s is not a whole sample at {FS} Hz")
    start = int(start)
    if start < 0:
        raise ValueError(f"scoring starts {-onsets[0]:g} s before the signal")
    if start + n_epochs * EPOCH_SAMPLES > len(signal):
        raise ValueError(f"signal ({signal_sec:.0f} s) is shorter than the scoring ({ends[-1]:.0f} s)")
    x = signal[start:start + n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)

    recording_id = psg.stem  # SN001
    summary = save_recording(
        processed_path(DATASET, recording_id),
        x=x, y=y, epoch_idx=np.arange(n_epochs),
        fs_original=fs, channel=channel, subject_id=subject_id,
        recording_id=recording_id, dataset=DATASET,
    )

    # Lights off/on versus the scored period (positive margin = scoring inside the lights period)
    lights_off = first(onsets_all[descriptions == "Lights off"])
    lights_on = first(onsets_all[descriptions == "Lights on"])
    summary["margin_off"] = None if lights_off is None else onsets[0] - lights_off
    summary["margin_on"] = None if lights_on is None else lights_on - ends[-1]
    log.info(
        "%s: %d epochs (%d annotations) | %s | scoring file starts %+g s | scored %g-%g s of %g s | "
        "lights off %s, on %s | median |x| %.1f uV | %s",
        recording_id, n_epochs, len(onsets), channel, offset, onsets[0], ends[-1], signal_sec,
        lights_off, lights_on, np.median(np.abs(x)), dict(zip(STAGES, summary["class_counts"])),
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

    files = validate_dataset(DATASET)  # {subject: [(psg, scoring)]}; checks the 20 recordings
    totals = np.zeros(N_CLASSES, dtype=np.int64)
    margins_off, margins_on, n_recordings = [], [], 0
    for subject_id, pairs in files.items():
        for psg, hyp in pairs:
            try:
                summary = prepare_recording(psg, hyp, subject_id, cfg)
            except ValueError as e:
                raise ValueError(f"{psg.name}: {e}") from e  # say which recording failed
            totals += summary["class_counts"]
            n_recordings += 1
            if summary["margin_off"] is not None:
                margins_off.append(summary["margin_off"])
            if summary["margin_on"] is not None:
                margins_on.append(summary["margin_on"])

    log.info("TOTAL: %d recordings, %d epochs | %s",
             n_recordings, totals.sum(), dict(zip(STAGES, totals.tolist())))
    log.info("Lights off found in %d recordings, lights on in %d | smallest margin: %s s at the start, "
             "%s s at the end (negative = scored epochs outside lights off/on)",
             len(margins_off), len(margins_on), min(margins_off, default=None), min(margins_on, default=None))


if __name__ == "__main__":
    main()