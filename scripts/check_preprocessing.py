"""Acceptance check for issue 7: both filter conditions come from the same, unchanged .npz files.

For each recording, prints the share of signal power below 0.3 Hz and above 35 Hz with
the filter off and on, and the share of clipped samples; then the median per dataset.
Run from the repository root (first 2 recordings per dataset, or all of them):

    python -m scripts.check_preprocessing
    python -m scripts.check_preprocessing all
"""
import hashlib
import sys

import numpy as np
from scipy.signal import welch

from src.data.dataset import SleepRecordings, recording_paths
from src.data.format import FS


def out_of_band(x):
    """Share of the power spectrum below 0.3 Hz and above 35 Hz."""
    freqs, power = welch(x.ravel(), fs=FS, nperseg=10 * FS)
    return power[freqs < 0.3].sum() / power.sum(), power[freqs > 35].sum() / power.sum()


def file_hashes(paths):
    return [hashlib.md5(p.read_bytes()).hexdigest() for p in paths]


def main(n_per_dataset=2):
    for dataset in ("sleep_edf", "isruc", "hmc"):
        paths = recording_paths(dataset)[:n_per_dataset]
        before = file_hashes(paths)
        rows = []
        for path in paths:  # one recording at a time, to keep memory low
            a, b = SleepRecordings([path], filter=False)[0], SleepRecordings([path], filter=True)[0]
            row = (*out_of_band(a["x"]), *out_of_band(b["x"]), np.mean(np.abs(b["x"]) >= 20))
            rows.append(row)
            print(f"{a['recording_id']:>8}: filter off <0.3 Hz {row[0]:6.2%} >35 Hz {row[1]:6.2%} | "
                  f"filter on <0.3 Hz {row[2]:6.2%} >35 Hz {row[3]:6.2%} | "
                  f"clipped {row[4]:.3%} | labels equal {np.array_equal(a['y'], b['y'])}")
        m = np.median(rows, axis=0)
        print(f"{dataset} median of {len(rows)}: filter off <0.3 Hz {m[0]:6.2%} >35 Hz {m[1]:6.2%} | "
              f"filter on <0.3 Hz {m[2]:6.2%} >35 Hz {m[3]:6.2%} | clipped {m[4]:.3%}")
        print(f"{dataset}: files unchanged after loading: {file_hashes(paths) == before}")


if __name__ == "__main__":
    main(None if sys.argv[1:] == ["all"] else 2)