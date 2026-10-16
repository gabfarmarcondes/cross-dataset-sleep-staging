"""Where does one ISRUC recording hit the physical limit of its C3 channel? Per stage and per epoch.

Reads only N/N.rec and N/N_1.txt. Run from the repository root, e.g. for subject 1:
"""
import sys

import mne
import numpy as np

from scripts.inspect_isruc_headers import channel_header
from src.data.format import STAGES
from src.read_data import as_edf, get_subject_files, load_config, pick_channel, read_isruc_labels


def main(sid):
    cfg = load_config()[0]["isruc"]
    (rec, labels_path), = get_subject_files("isruc", sid)
    limit = float(channel_header(rec, cfg["channel_names"])["phys_max"])
    with as_edf(str(rec)) as edf:
        raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
        x = raw.get_data(picks=[pick_channel(raw.ch_names, cfg["channel_names"])], units="uV")[0]
        per_epoch = int(30 * raw.info["sfreq"])
    n = len(x) // per_epoch
    at_limit = np.abs(x[:n * per_epoch]) >= 0.999 * limit
    clipped = at_limit.reshape(n, per_epoch).mean(axis=1)  # fraction of samples at the limit, per epoch
    y = np.array([cfg["labels"]["mapping"][c] for c in read_isruc_labels(labels_path)])

    print(f"isruc_{sid}: {n} epochs | limit {limit:g} uV | {100 * clipped.mean():.1f}% of samples at the limit")
    for k, stage in enumerate(STAGES):
        m = y == k
        if m.any():
            print(f"  {stage:>3}: {m.sum():4d} epochs | {100 * clipped[m].mean():5.1f}% of samples at the limit"
                  f" | {(clipped[m] > 0.1).sum():4d} epochs with >10%")
    bad = np.flatnonzero(clipped > 0.1)
    if bad.size:
        runs = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1)
        longest = max(runs, key=len)
        print(f"  epochs with >10% at the limit: {bad.size} in {len(runs)} runs;"
              f" longest run {len(longest)} epochs ({longest[0]}-{longest[-1]})")


if __name__ == "__main__":
    main(int(sys.argv[1]))