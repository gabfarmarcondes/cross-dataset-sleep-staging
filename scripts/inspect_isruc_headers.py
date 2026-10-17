"""Print the EDF signal-header fields of the C3 channel in each ISRUC .rec file.

Only the signal header is read and printed. The general header holds the patient
and recording identification fields; it is used only for the number of signals.
Run from the repository root:

    python -m scripts.inspect_isruc_headers
"""
from src.data.edf_header import channel_header
from src.read_data import load_config, validate_dataset


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