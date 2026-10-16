"""Print the EDF signal-header fields of the C3 channel in each ISRUC .rec file.

Only the signal header is read and printed. The general header holds the patient
and recording identification fields; it is used only for the number of signals.
Run from the repository root:

    python -m scripts.inspect_isruc_headers
"""
from src.read_data import load_config, validate_dataset

# EDF signal header: each field is stored for all signals in a row, in this order
SIGNAL_FIELDS = [("label", 16), ("transducer", 80), ("dimension", 8), ("phys_min", 8),
                 ("phys_max", 8), ("dig_min", 8), ("dig_max", 8), ("prefilter", 80),
                 ("samples_per_record", 8), ("reserved", 32)]


def channel_header(path, accepted_names):
    with open(path, "rb") as f:
        general = f.read(256)
        n_signals = int(general[252:256].decode("ascii"))
        block = f.read(256 * n_signals)
    fields, pos = {}, 0
    for name, size in SIGNAL_FIELDS:
        fields[name] = [block[pos + k * size: pos + (k + 1) * size].decode("latin-1").strip()
                        for k in range(n_signals)]
        pos += size * n_signals
    k = next(i for i, label in enumerate(fields["label"]) if label in accepted_names)
    return {name: fields[name][k] for name, _ in SIGNAL_FIELDS if name != "reserved"}


def main():
    datasets, _ = load_config()
    accepted = datasets["isruc"]["channel_names"]
    for sid, pairs in validate_dataset("isruc").items():
        h = channel_header(pairs[0][0], accepted)
        print(f"isruc_{sid}: {h['label']} | dim {h['dimension']!r} | phys {h['phys_min']}..{h['phys_max']} | "
              f"dig {h['dig_min']}..{h['dig_max']} | prefilter {h['prefilter']!r} | "
              f"transducer {h['transducer']!r} | samples/record {h['samples_per_record']}")


if __name__ == "__main__":
    main()