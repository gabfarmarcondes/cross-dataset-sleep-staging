"""Check whether the ISRUC C3 signal hits the physical limit declared in the EDF header.

Reads only N/N.rec through the same helpers as the loader. Run from the repository root:

    python -m scripts.check_isruc_clipping
"""
import mne
import numpy as np

from scripts.inspect_isruc_headers import channel_header
from src.read_data import as_edf, load_config, pick_channel, validate_dataset


def main():
    accepted = load_config()[0]["isruc"]["channel_names"]
    for sid, pairs in validate_dataset("isruc").items():
        rec = pairs[0][0]
        limit = float(channel_header(rec, accepted)["phys_max"])  # declared range is -limit..+limit
        with as_edf(str(rec)) as edf:
            raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
            x = raw.get_data(picks=[pick_channel(raw.ch_names, accepted)], units="uV")[0]
        at_limit = np.mean(np.abs(x) >= 0.999 * limit) * 100
        print(f"isruc_{sid}: limit {limit:g} uV | max |x| {np.abs(x).max():.2f} | "
              f"p99 |x| {np.percentile(np.abs(x), 99):.2f} | at the limit {at_limit:.3f}%")


if __name__ == "__main__":
    main()