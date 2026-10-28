"""Evaluate one trained run on its test subjects (issue 13). From the repository root:

    python -m src.evaluation.evaluate outputs/runs/sleep_edf_fold0_sequence_seed0
    python -m src.evaluation.evaluate outputs/runs/sleep_edf_source_sequence_seed0 --target isruc

Which subjects are tested comes from the run itself (run.json) and the frozen splits:
    in-domain run (trained with --fold i)   folds[i].test of the same dataset
    source run (trained with --source)      target.test of the --target dataset
target.adapt is never used here, and the test labels are only used to compute the metrics
(D11). The model is the best checkpoint (best.pt), in evaluation mode, with the same
preprocessing as in training.

Writes next to the run (with _<target> in the name for cross-dataset evaluations):
    evaluation.json   metrics, confusion matrices, test subjects and timing
    predictions.csv   one row per test epoch: recording, subject, epoch, true and predicted stage
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.splits import load_splits
from src.evaluation.metrics import compute_metrics
from src.training.loop import make_model, minibatches, prepare
from src.training.train import load_recordings

DATASETS = ["sleep_edf", "isruc", "hmc"]


def evaluation_subjects(info, target=None):
    """(dataset, subjects) on which a run is tested; never adapt or training subjects."""
    split, dataset = info["split"], info["dataset"]
    if split.get("subjects"):
        raise ValueError(f"{info['run']} is an overfit check (--subjects): it has no test subjects")
    if target is None:
        if split.get("fold") is None:
            raise ValueError(f"{info['run']} was trained with --source: pass --target <dataset>")
        return dataset, load_splits(dataset)["in_domain"]["folds"][split["fold"]]["test"]
    if not split.get("source"):
        raise ValueError(f"{info['run']} is an in-domain run: cross-dataset needs a --source run")
    if target == dataset:
        raise ValueError("the target must be a different dataset from the source")
    return target, load_splits(target)["cross_dataset"]["target"]["test"]


def file_suffix(target):
    return "" if target is None else f"_{target}"


def load_model(run_dir, info, device):
    batch_norm = info.get("model", {}).get("batch_norm", "standard")  # runs before D27: standard
    model = make_model(info["mode"], batch_norm).to(device)
    model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location=device))
    return model.eval()


def predict(model, recording, mode, cfg, device):
    """Predicted stage of every epoch of one recording, with the model in evaluation mode.

    Sequence mode runs the recording alone (one row), with the LSTM state carried from one
    minibatch to the next, as in training.
    """
    model.eval()
    data = prepare([recording], mode)
    cfg = {**cfg, "batch_size": 1}  # one recording: no empty rows
    preds, state = [], None
    with torch.no_grad():
        for x, _, lengths, reset in minibatches(data, mode, cfg):
            x = torch.from_numpy(x).to(device)
            if mode == "sequence":
                if reset:
                    state = None
                logits, state = model(x, lengths, state)
                preds.append(logits[0, :lengths[0]].argmax(dim=-1).cpu().numpy())
            else:
                preds.append(model(x.unsqueeze(1)).argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def timed_predict(model, recording, mode, cfg, device):
    """predict() and the time it took, in seconds (the GPU is waited for before each reading)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    y_pred = predict(model, recording, mode, cfg, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return y_pred, time.perf_counter() - start


def training_seconds(run_dir):
    """Mean seconds per training epoch (training pass + validation pass), from history.csv."""
    with open(Path(run_dir) / "history.csv", encoding="utf-8") as f:
        return float(np.mean([float(row["seconds"]) for row in csv.DictReader(f)]))


def evaluate_run(run_dir, target=None, device=None, log=print):
    """Evaluate a run on its test subjects, write evaluation.json and predictions.csv, return the result."""
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if "result" not in info:
        raise ValueError(f"{info['run']} did not finish training (no result in run.json)")
    dataset, subjects = evaluation_subjects(info, target)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(run_dir, info, device)
    recordings = load_recordings(dataset, load_splits(dataset), subjects, info["preprocessing"])

    rows, seconds = [], []
    for recording in recordings:
        y_pred, elapsed = timed_predict(model, recording, info["mode"], info["training"], device)
        seconds.append(elapsed)
        rows += [(recording["recording_id"], recording["subject_id"], int(e), int(t), int(p))
                 for e, t, p in zip(recording["epoch_idx"], recording["y"], y_pred)]
    y_true = np.array([r[3] for r in rows])
    y_pred = np.array([r[4] for r in rows])

    result = {
        "run": info["run"],
        "trained_on": info["dataset"],
        "tested_on": dataset,
        "scenario": "in_domain" if target is None else "cross_dataset",
        "mode": info["mode"],
        "seed": info["seed"],
        "fold": info["split"].get("fold"),
        "batch_norm": info.get("model", {}).get("batch_norm", "standard"),
        "filter": info["preprocessing"]["filter"],
        "test_subjects": subjects,
        "test_recordings": [r["recording_id"] for r in recordings],
        "metrics": compute_metrics(y_true, y_pred),
        "cost": {
            "parameters": info["parameters"],
            "epochs_run": info["result"]["epochs_run"],
            "best_epoch": info["result"]["best_epoch"],
            "seconds_per_training_epoch": training_seconds(run_dir),
            "inference_seconds_per_recording": float(np.mean(seconds)),
            "inference_seconds": seconds,
            "training_device": info["device"],
            "inference_device": str(device),
        },
    }
    suffix = file_suffix(target)
    (run_dir / f"evaluation{suffix}.json").write_text(json.dumps(result, indent=2) + "\n",
                                                       encoding="utf-8")
    with open(run_dir / f"predictions{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["recording_id", "subject_id", "epoch_idx", "y_true", "y_pred"])
        writer.writerows(rows)
    m = result["metrics"]
    log(f"{info['run']} on {dataset} {subjects}: accuracy {m['accuracy']:.1%}, "
        f"macro-F1 {m['macro_f1']:.1%}, kappa {m['kappa']:.3f} ({m['n_epochs']} epochs)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained run on its test subjects.")
    parser.add_argument("run_dir", help="outputs/runs/<run name>")
    parser.add_argument("--target", choices=DATASETS, help="cross-dataset: test on this dataset")
    args = parser.parse_args()
    evaluate_run(args.run_dir, args.target)


if __name__ == "__main__":
    main()