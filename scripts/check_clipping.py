"""Quality rule D19: share of samples at the physical limit of the EEG channel, per recording.

The limit is the physical range declared in the EDF header. Run from the repository root
for each dataset:

    python -m scripts.check_clipping sleep_edf
    python -m scripts.check_clipping isruc
    python -m scripts.check_clipping hmc
"""
import sys

import mne
import numpy as np

from src.data.edf_header import channel_header
from src.read_data import as_edf, load_config, pick_channel, validate_dataset

MAX_AT_LIMIT = 5.0                                   # % of samples; above this, exclude (D19)
TO_MICROVOLT = {"uV": 1.0, "µV": 1.0, "mV": 1e3, "V": 1e6}


def main(dataset):
    accepted = load_config()[0][dataset]["channel_names"]
    for sid, pairs in validate_dataset(dataset).items():
        for signal_path, _ in pairs:
            h = channel_header(signal_path, accepted)
            if h["dimension"] not in TO_MICROVOLT:
                raise ValueError(f"{signal_path.name}: unknown unit {h['dimension']!r}")
            scale = TO_MICROVOLT[h["dimension"]]
            low, high = float(h["phys_min"]) * scale, float(h["phys_max"]) * scale
            with as_edf(str(signal_path)) as edf:
                raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
                x = raw.get_data(picks=[pick_channel(raw.ch_names, accepted)], units="uV")[0]
            tol = 0.001 * (high - low)
            at_limit = np.mean((x >= high - tol) | (x <= low + tol)) * 100
            flag = "  <-- above the D19 limit" if at_limit > MAX_AT_LIMIT else ""
            print(f"{dataset} {sid} ({signal_path.name}): range {low:g}..{high:g} uV | "
                  f"p99 |x| {np.percentile(np.abs(x), 99):.2f} | at the limit {at_limit:.3f}%{flag}")


if __name__ == "__main__":
    main(sys.argv[1])