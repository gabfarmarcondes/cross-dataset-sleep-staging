"""Check that the local datasets match configs/datasets.yaml.

For every recording: the files exist, an accepted EEG channel is present with the
configured sampling rate and, for ISRUC, there is one label per 30-s epoch.
Prints one summary per dataset and exits with status 1 if anything is wrong.

Usage: python scripts/check_datasets.py
"""
import sys
from collections import Counter
from pathlib import Path

import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from read_data import as_edf, load_config, pick_channel, read_isruc_labels, validate_dataset  # noqa: E402

EPOCH_SECONDS = 30


def check_recording(dataset, cfg, sid, signal, annotation):
    """Return (channel name used, list of problems) for one recording."""
    with as_edf(signal) as edf:
        names = mne.io.read_raw_edf(edf, preload=False, verbose="error").ch_names
        channel = pick_channel(names, cfg["channel_names"])
        # Read the header again with only this channel: MNE reports the highest rate of the file otherwise
        raw = mne.io.read_raw_edf(edf, include=[channel], preload=False, verbose="error")
    problems = []
    sfreq = raw.info["sfreq"]
    if sfreq != cfg["sampling_rate"]:
        problems.append(f"subject {sid}, {signal.name}: {channel} at {sfreq} Hz, expected {cfg['sampling_rate']}")
    if dataset == "isruc":
        n_labels = len(read_isruc_labels(annotation))
        n_epochs = raw.n_times // int(sfreq * EPOCH_SECONDS)
        if n_labels != n_epochs:
            problems.append(f"subject {sid}: {n_labels} labels, {n_epochs} epochs in the signal")
    return channel, problems


def check_dataset(dataset, cfg):
    """Print a summary of one dataset; return True if no problem was found."""
    try:
        pairs = validate_dataset(dataset)
    except (FileNotFoundError, ValueError) as error:
        print(f"{dataset}: FAILED - {error}")
        return False
    channels, problems = Counter(), []
    for sid, recordings in pairs.items():
        for signal, annotation in recordings:
            try:
                channel, recording_problems = check_recording(dataset, cfg, sid, signal, annotation)
            except ValueError as error:
                problems.append(f"subject {sid}, {signal.name}: {error}")
                continue
            channels[channel] += 1
            problems += recording_problems
    n_recordings = sum(len(r) for r in pairs.values())
    print(f"{dataset}: {len(pairs)} subjects, {n_recordings}/{cfg['expected_recordings']} recordings")
    print(f"  channels used: {dict(channels)} (sampling rate expected {cfg['sampling_rate']} Hz)")
    print(f"  mismatches: {len(problems)}")
    for problem in problems:
        print(f"    {problem}")
    return not problems


def main():
    datasets, _ = load_config()
    results = [check_dataset(dataset, cfg) for dataset, cfg in datasets.items()]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
