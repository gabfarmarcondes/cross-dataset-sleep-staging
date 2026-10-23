"""Subject-wise splits (issue 9), frozen in configs/splits/<dataset>.json.

Every partition is a list of subject IDs; "recordings" maps each subject to its
recording IDs, so all nights of a subject always land in the same partition.

    in_domain.folds[i]   train / valid / test     k-fold by subject (k=5); valid = 10% of the
                                                  training subjects (as in TinySleepNet)
    cross_dataset.source train / valid            the dataset used as source (all subjects)
    cross_dataset.target adapt / test             the dataset used as target: adapt is unlabeled
                                                  data for adaptation (TCC 2), test is never trained on

Folds and the adapt/test halves are stratified by each subject's share of wake
epochs, the stage that varies most between subjects. Generate the files once,
from the repository root, and commit them:

    python -m src.data.splits

An existing file is never overwritten: delete it on purpose to regenerate it.
"""
import json
import re
from datetime import date

import numpy as np

from src.data.dataset import recording_paths
from src.read_data import ROOT, load_config

SPLITS_DIR = ROOT / "configs" / "splits"
SEED = 0
K_FOLDS = 5
VALID_FRACTION = 0.10
DATASETS = ["sleep_edf", "isruc", "hmc"]


def stratified_folds(subjects, wake_share, k, rng):
    """Split subjects into k groups with similar wake share.

    Subjects are sorted by wake share and taken in blocks of k; inside each block,
    the k subjects are dealt to the k groups in random order.
    """
    order = sorted(subjects, key=lambda s: (wake_share[s], s))
    groups = [[] for _ in range(k)]
    for start in range(0, len(order), k):
        block = order[start:start + k]
        for group, subject in zip(rng.permutation(k), block):
            groups[group].append(subject)
    return [sorted(g) for g in groups]


def pick_validation(subjects, rng, fraction=VALID_FRACTION):
    """(train, valid): `fraction` of the subjects, at least one, at random."""
    n_valid = max(1, round(fraction * len(subjects)))
    valid = sorted(int(s) for s in rng.choice(subjects, size=n_valid, replace=False))
    return [s for s in subjects if s not in valid], valid


def subject_summary(dataset):
    """{subject: [recording ids]} and {subject: wake share}, for the subjects listed in datasets.yaml."""
    listed = load_config()[0][dataset]["subjects"]
    recordings = {s: [] for s in listed}
    wake, total = {s: 0 for s in listed}, {s: 0 for s in listed}
    for path in recording_paths(dataset):
        with np.load(path, allow_pickle=False) as f:
            sid, y = int(f["subject_id"]), f["y"]
        if sid not in recordings:  # a file left over from a subject no longer in the list
            raise ValueError(f"{path.name}: subject {sid} is not in datasets.yaml; delete the file")
        recordings[sid].append(path.stem)
        wake[sid] += int(np.sum(y == 0))
        total[sid] += len(y)
    missing = [s for s in listed if not recordings[s]]
    if missing:
        raise ValueError(f"{dataset}: no .npz files for subjects {missing}")
    return {s: sorted(r) for s, r in recordings.items()}, {s: wake[s] / total[s] for s in listed}


def make_splits(dataset, seed=SEED, k=K_FOLDS):
    """All splits of one dataset, as a JSON-ready dict."""
    recordings, wake_share = subject_summary(dataset)
    subjects = sorted(recordings)
    rng = np.random.default_rng(seed)

    folds = []
    for test in stratified_folds(subjects, wake_share, k, rng):
        train, valid = pick_validation([s for s in subjects if s not in test], rng)
        folds.append({"train": train, "valid": valid, "test": test})

    source_train, source_valid = pick_validation(subjects, rng)
    test, adapt = stratified_folds(subjects, wake_share, 2, rng)

    return {
        "dataset": dataset,
        "created": date.today().isoformat(),
        "seed": seed,
        "rule": "subject-wise; folds and adapt/test stratified by wake share; valid = 10% of training subjects",
        "recordings": {str(s): recordings[s] for s in subjects},
        "in_domain": {"k": k, "folds": folds},
        "cross_dataset": {"source": {"train": source_train, "valid": source_valid},
                          "target": {"adapt": adapt, "test": test}},
    }


def save_splits(splits, folder=SPLITS_DIR):
    """Write <dataset>.json; refuse to overwrite a frozen file."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{splits['dataset']}.json"
    if path.exists():
        raise FileExistsError(f"{path} is frozen; delete it on purpose to regenerate it")
    text = json.dumps(splits, indent=2)
    # one line per list ([2, 5, 7]) instead of one line per element, so the file is easy to review
    text = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def load_splits(dataset, folder=SPLITS_DIR):
    with open(folder / f"{dataset}.json", encoding="utf-8") as f:
        return json.load(f)


def recordings_of(splits, subjects):
    """Recording IDs of the given subjects (e.g. splits["in_domain"]["folds"][0]["test"])."""
    return [r for s in subjects for r in splits["recordings"][str(s)]]


def main():
    for dataset in DATASETS:
        if (SPLITS_DIR / f"{dataset}.json").exists():
            print(f"{dataset}: frozen, not regenerated")
            continue
        splits = make_splits(dataset)
        _, wake_share = subject_summary(dataset)

        def share(subjects):
            return np.mean([wake_share[s] for s in subjects])

        target = splits["cross_dataset"]["target"]
        print(f"{dataset}: written to {save_splits(splits)}")
        print(f"  target adapt {target['adapt']} (mean wake {share(target['adapt']):.0%})")
        print(f"  target test  {target['test']} (mean wake {share(target['test']):.0%})")
        for i, fold in enumerate(splits["in_domain"]["folds"]):
            print(f"  fold {i}: test {fold['test']} (mean wake {share(fold['test']):.0%}), valid {fold['valid']}")


if __name__ == "__main__":
    main()