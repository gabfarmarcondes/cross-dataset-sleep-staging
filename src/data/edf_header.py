"""
Read EDF header fields directly, without the patient and recording identification fields.

The general header (256 bytes) holds the patient and recording identification in
bytes 8-167; nothing here reads or returns them.
"""
from datetime import datetime

# EDF signal header: each field is stored for all signals in a row, in this order
SIGNAL_FIELDS = [("label", 16), ("transducer", 80), ("dimension", 8), ("phys_min", 8),
                 ("phys_max", 8), ("dig_min", 8), ("dig_max", 8), ("prefilter", 80),
                 ("samples_per_record", 8), ("reserved", 32)]

def edf_start(path):
    # Start date and time from the general header (bytes 168-183, "dd.mm.yyh.mm.ss")
    with open(path, "rb") as f:
        header = f.read(184)
    return header[168:184].decode("ascii")


def start_time(path):
    # Start of the recording as a datetime (one-second resolution)
    return datetime.strptime(edf_start(path), "%d.%m.%y%H.%M.%S")


def channel_header(path, accepted_names):
    # Signal-header fields of the firsts channel whose label si in accepted_names
    with open(path, "rb") as f:
        general = f.read(256)
        n_signals = int(general[252:256].decode("ascii"))
        block = f.read(256 * n_signals)
    fields , pos = {}, 0
    for name, size in SIGNAL_FIELDS:
        fields[name] = [block[pos + k * size: pos + (k + 1) * size].decode("latin-1").strip()
                        for k in range(n_signals)]
        pos += size * n_signals
    k = next(i for i, label in enumerate(fields["label"]) if label in accepted_names)
    return {name: fields[name][k] for name, _ in SIGNAL_FIELDS if name != "reserved"}
        