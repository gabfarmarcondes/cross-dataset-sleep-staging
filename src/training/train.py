"""Train TinySleepNet on one split of one dataset (issue 12). From the repository root:

    python -m src.training.train --dataset sleep_edf --fold 0         # in-domain, fold 0
    python -m src.training.train --dataset sleep_edf --source         # dataset as source
    python -m src.training.train --dataset sleep_edf --subjects 0 1 --mode epoch   # overfit check
    python -m src.training.train --dataset sleep_edf --fold 0 --batch-norm standard  # batch-norm comparison
    python -m src.training.train --dataset sleep_edf --fold 0 --no-augment   # issue 15 comparison
    python -m src.training.train --dataset sleep_edf --fold 0 --no-filter    # issue 17 comparison

Training and validation subjects come from the frozen splits (configs/splits). The test
subjects (in_domain folds[i].test, cross_dataset target) are never loaded here, so model
selection can only use source labels. With --subjects, validation uses the same
subjects as training: it is a bug check (the model must be able to memorize them), not
an experiment.

Settings: the `model`, `training` and `preprocessing` blocks of configs/experiment.yaml;
the training data are augmented as set there (signal_shift, sequence_skip; issue 15),
unless --no-augment is given, and band-pass filtered as set there (filter), unless
--no-filter is given (issue 17; the run name then ends in _nofilter, and the evaluation of
the run loads its test recordings without the filter too). Output, in
outputs/runs/<run name>/:
    history.csv   one row per training epoch: loss, accuracy, macro-F1 (train and validation)
    best.pt       weights with the best validation macro-F1
    run.json      settings, subjects, recordings and result of the run
    train.log     the console output
"""
import argparse
import json
import logging
from datetime import datetime

import torch
import yaml

from src.data.dataset import SleepRecordings, without_filter
from src.data.splits import load_splits, recordings_of
from src.models.tinysleepnet import BATCH_NORMS, count_parameters
from src.read_data import ROOT, processed_path
from src.training.augment import is_augmented, without_augmentation
from src.training.loop import MODES, fit, make_model
from src.training.seed import set_seed

EXPERIMENT_FILE = ROOT / "configs" / "experiment.yaml"
RUNS_DIR = ROOT / "outputs" / "runs"

log = logging.getLogger(__name__)


def select_subjects(splits, fold=None, source=False, subjects=None):
    """(train, valid) subject lists of a run; never the test or adapt subjects."""
    if subjects:
        unknown = [s for s in subjects if str(s) not in splits["recordings"]]
        if unknown:
            raise ValueError(f"subjects {unknown} are not in the {splits['dataset']} splits")
        return list(subjects), list(subjects)
    if source:
        part = splits["cross_dataset"]["source"]
    else:
        part = splits["in_domain"]["folds"][fold]
    return part["train"], part["valid"]


def same_training(a, b):
    """Whether two `training` blocks are equal, apart from the mode (it is in the run name)."""
    return {k: v for k, v in a.items() if k != "mode"} == {k: v for k, v in b.items() if k != "mode"}


def run_name(dataset, mode, seed, fold=None, source=False, subjects=None, filtered=True,
             batch_norm="official", augmented=True):
    """e.g. sleep_edf_fold0_sequence_seed0; suffixes mark the non-default settings."""
    if subjects:
        split = "subjects-" + "-".join(str(s) for s in subjects)
    elif source:
        split = "source"
    else:
        split = f"fold{fold}"
    name = f"{dataset}_{split}_{mode}_seed{seed}"
    if not filtered:
        name += "_nofilter"
    if batch_norm != "official":
        name += f"_bn{batch_norm}"
    if not augmented:
        name += "_noaug"
    return name


def load_recordings(dataset, splits, subjects, preprocessing):
    paths = [processed_path(dataset, r) for r in recordings_of(splits, subjects)]
    return SleepRecordings(paths, **preprocessing)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinySleepNet on one split.")
    parser.add_argument("--dataset", required=True, choices=["sleep_edf", "isruc", "hmc"])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--fold", type=int, help="in-domain fold (0-4)")
    which.add_argument("--source", action="store_true", help="all subjects, as source dataset")
    which.add_argument("--subjects", type=int, nargs="+", help="overfit check on these subjects")
    parser.add_argument("--mode", choices=MODES, default=None, help="default: experiment.yaml")
    parser.add_argument("--batch-norm", choices=BATCH_NORMS, default=None,
                        help="default: experiment.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true",
                        help="train without data augmentation (signal_shift and sequence_skip at 0)")
    parser.add_argument("--no-filter", action="store_true",
                        help="load the recordings without the band-pass filter (filter ablation)")
    parser.add_argument("--max-epochs", type=int, default=None, help="override experiment.yaml")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(EXPERIMENT_FILE, encoding="utf-8") as f:
        experiment = yaml.safe_load(f)
    cfg, preprocessing = experiment["training"], experiment["preprocessing"]
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.no_augment:
        cfg = without_augmentation(cfg)
    if args.no_filter:
        preprocessing = without_filter(preprocessing)  # run.json records the filter actually used
    mode = args.mode or cfg["mode"]
    cfg["mode"] = mode  # run.json records the mode actually used
    model_cfg = experiment.get("model", {})
    batch_norm = args.batch_norm or model_cfg.get("batch_norm", "official")
    model_cfg["batch_norm"] = batch_norm
    augmented = is_augmented(cfg, mode)

    name = run_name(args.dataset, mode, args.seed, args.fold, args.source, args.subjects,
                    preprocessing["filter"], batch_norm, augmented)
    run_dir = RUNS_DIR / name
    if (run_dir / "history.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{run_dir} already has a run; use --overwrite to replace it")
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", mode="w")],
    )

    set_seed(args.seed)  # before the model: weight initialization depends on it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(args.dataset)
    train_subjects, valid_subjects = select_subjects(splits, args.fold, args.source, args.subjects)
    train_recordings = load_recordings(args.dataset, splits, train_subjects, preprocessing)
    valid_recordings = load_recordings(args.dataset, splits, valid_subjects, preprocessing)
    model = make_model(mode, batch_norm).to(device)

    info = {
        "run": name,
        "dataset": args.dataset,
        "split": {"fold": args.fold, "source": args.source, "subjects": args.subjects},
        "mode": mode,
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "parameters": count_parameters(model),
        "train_subjects": train_subjects,
        "valid_subjects": valid_subjects,
        "train_recordings": [r["recording_id"] for r in train_recordings],
        "valid_recordings": [r["recording_id"] for r in valid_recordings],
        "model": model_cfg,
        "training": cfg,
        "preprocessing": preprocessing,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    run_json = run_dir / "run.json"
    run_json.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")  # rewritten at the end
    n_train = sum(len(r["y"]) for r in train_recordings)
    n_valid = sum(len(r["y"]) for r in valid_recordings)
    augmentation = (f"shift up to {cfg['signal_shift']:.0%} of an epoch"
                    + (f", skip 0-{cfg['sequence_skip']} epochs" if mode == "sequence" else "")
                    if augmented else "no augmentation")
    low, high = preprocessing["band"]
    filtering = f"band-pass {low:g}-{high:g} Hz" if preprocessing["filter"] else "no band-pass filter"
    log.info(f"run {name} on {device}: {mode} mode, {batch_norm} batch norm, "
             f"{info['parameters']:,} parameters, {augmentation}, {filtering}")
    log.info(f"train subjects {train_subjects} ({len(train_recordings)} recordings, {n_train} epochs)")
    log.info(f"valid subjects {valid_subjects} ({len(valid_recordings)} recordings, {n_valid} epochs)")

    result = fit(model, list(train_recordings), list(valid_recordings), mode, cfg, run_dir,
                 args.seed, device, log=log.info)
    info["result"] = result
    info["finished"] = datetime.now().isoformat(timespec="seconds")
    run_json.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    log.info(f"best validation macro-F1 {result['best_valid_macro_f1']:.1%} "
             f"at epoch {result['best_epoch']} of {result['epochs_run']}; saved in {run_dir}")


if __name__ == "__main__":
    main()