import shutil
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATASETS_FILE = ROOT / "configs" / "datasets.yaml"
PATHS_FILE = ROOT / "configs" / "paths.yaml"
PATHS_EXAMPLE_FILE = ROOT / "configs" / "paths.example.yaml"


@lru_cache(maxsize=None)
def load_config():
    """Return (datasets, paths) from configs/datasets.yaml and configs/paths.yaml, read once."""
    if not PATHS_FILE.exists():
        raise FileNotFoundError(
            f"{PATHS_FILE} not found. Copy {PATHS_EXAMPLE_FILE} to {PATHS_FILE} "
            "and set data_root to the folder that contains the datasets."
        )
    with open(DATASETS_FILE) as f:
        datasets = yaml.safe_load(f)
    with open(PATHS_FILE) as f:
        paths = yaml.safe_load(f)
    return datasets, paths


def load_subjects(dataset):
    """Return the frozen subject IDs of a dataset (configs/datasets.yaml)."""
    return load_config()[0][dataset]["subjects"]


def get_subject_files(dataset, sid):
    """Return the (signal, annotation) path pairs of one subject, one per recording."""
    datasets, paths = load_config()
    if dataset not in datasets:
        raise ValueError(f"Unknown dataset: {dataset}")
    files = datasets[dataset]["files"]
    # data_root may be absolute or relative to the repository root
    folder = ROOT / paths["data_root"] / datasets[dataset]["subdir"]

    if dataset == "sleep_edf":
        pairs = []
        for psg in sorted(folder.glob(files["signal_pattern"].format(subject=sid))):
            # The hypnogram shares the first 7 characters of its PSG; the 8th (scorer) varies
            hypnograms = list(folder.glob(files["annotation_pattern"].format(prefix=psg.name[:7])))
            if len(hypnograms) != 1:
                raise FileNotFoundError(f"Expected one hypnogram for {psg.name}, found {len(hypnograms)}")
            pairs.append((psg, hypnograms[0]))
    else:
        # Build both paths directly. Never glob or list an ISRUC subject folder:
        # it holds extra files that must not be read.
        pairs = [(folder / files["signal"].format(subject=sid), folder / files["annotation"].format(subject=sid))]

    missing = [str(path) for pair in pairs for path in pair if not path.exists()]
    if not pairs or missing:
        raise FileNotFoundError(f"{dataset} subject {sid}: missing files {missing or 'all'}")
    return pairs

def processed_path(dataset, recording_id):
    """Path of the .npz file of one preprocessed recording."""
    _, paths = load_config()
    return ROOT / paths["processed_root"] / dataset / f"{recording_id}.npz"


def validate_dataset(dataset):
    """Resolve the files of every subject and check the recording count.

    Returns {sid: [(signal, annotation), ...]}.
    """
    pairs = {sid: get_subject_files(dataset, sid) for sid in load_subjects(dataset)}
    n_recordings = sum(len(p) for p in pairs.values())
    expected = load_config()[0][dataset]["expected_recordings"]
    if n_recordings != expected:
        raise ValueError(f"{dataset}: found {n_recordings} recordings, expected {expected}")
    return pairs


@contextmanager
def as_edf(path):
    """Yield a path with the .edf extension, so MNE can open the file.

    MNE (1.13.2) rejects the .rec extension used by ISRUC ("Only EDF files are
    supported, got rec"), so a non-.edf file is copied to a temporary .edf; an
    .edf path is yielded unchanged. Callers need preload=True, because the
    temporary file is deleted when the block exits.
    """
    path = Path(path)
    if path.suffix.lower() == ".edf":
        yield path
        return
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / f"{path.stem}.edf"
        shutil.copyfile(path, copy)
        yield copy


def pick_channel(available_names, accepted_names):
    """Return the first accepted channel name present in available_names."""
    for name in accepted_names:
        if name in available_names:
            return name
    raise ValueError(f"None of the accepted channels {accepted_names} is present; available: {available_names}")


def read_isruc_labels(path):
    """Return the ISRUC stage codes of one recording, one integer per 30-s epoch.

    Blank lines are ignored: some files end with many blank lines and use CRLF.
    """
    with open(path) as f:
        return [int(line) for line in f if line.strip()]
