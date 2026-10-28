"""Memorization check (issue 12; batch-norm decision, D27). From the repository root:

    python scripts/diagnose_overfit.py --mode sequence --batch-norm official
    python scripts/diagnose_overfit.py --mode sequence --batch-norm standard
    python scripts/diagnose_overfit.py --mode epoch --batch-norm official

Trains a fresh model on one recording (SC4001) without dropout and without L2, with the
other settings of experiment.yaml, until 99% training accuracy (at most 60 epochs). Then it
evaluates the model on the same recording in evaluation mode.

- A correct loop memorizes the recording: training accuracy close to 100%.
- If evaluation falls far below training on the same data, the model computes something
  different in the two modes. With standard batch norm, the script also evaluates with
  batch statistics, exactly as in training, and with running averages recomputed over the
  whole recording, to show where the difference comes from.

Results so far (standard batch norm): epoch mode 99.3% training, 99.3% evaluation;
sequence mode 99.2% training, 53.0% evaluation (75.3% recomputed, 98.6% with batch statistics).
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from torch import nn  # noqa: E402

from src.data.splits import load_splits  # noqa: E402
from src.models.tinysleepnet import BATCH_NORMS  # noqa: E402
from src.read_data import ROOT  # noqa: E402
from src.training.loop import MODES, make_model, make_optimizer, prepare, run_epoch  # noqa: E402
from src.training.minibatches import PAD_LABEL, sequence_minibatches  # noqa: E402
from src.training.train import load_recordings  # noqa: E402

MAX_EPOCHS = 60


def recompute_bn_statistics(cnn, x, batch_size):
    """Replace the running averages of the CNN's batch norms by a plain average over x."""
    for module in cnn.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None  # None = cumulative average in PyTorch
    cnn.train()
    for module in cnn.modules():
        if isinstance(module, nn.Dropout):
            module.eval()  # batch norm collects statistics, dropout stays off
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            cnn(torch.from_numpy(x[start:start + batch_size]).unsqueeze(1))


def evaluate_with_batch_statistics(model, data, cfg):
    """Sequence mode with batch norm using batch statistics, as in training; weights restored."""
    saved = copy.deepcopy(model.state_dict())
    model.train()  # batch statistics; this model has no dropout
    trues, preds, state = [], [], None
    with torch.no_grad():
        for x, y, lengths, reset in sequence_minibatches(data, cfg["batch_size"], cfg["seq_length"]):
            if reset:
                state = None
            logits, state = model(torch.from_numpy(x), lengths, state)
            real = y != PAD_LABEL
            trues.append(y[real])
            preds.append(logits.argmax(dim=-1).numpy()[real])
    model.load_state_dict(saved)
    y_true, y_pred = np.concatenate(trues), np.concatenate(preds)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def show(label, result):
    print(f"  {label:<44} acc {result['accuracy']:.1%}  mf1 {result['macro_f1']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Memorize one recording and evaluate it.")
    parser.add_argument("--mode", choices=MODES, default="sequence")
    parser.add_argument("--batch-norm", choices=BATCH_NORMS, default="official")
    args = parser.parse_args()

    with open(ROOT / "configs" / "experiment.yaml", encoding="utf-8") as f:
        experiment = yaml.safe_load(f)
    cfg = {**experiment["training"], "l2_weight_decay": 0.0}
    splits = load_splits("sleep_edf")
    recording = load_recordings("sleep_edf", splits, [0], experiment["preprocessing"])[0]  # SC4001
    data = prepare([recording], args.mode)

    print(f"memorization: SC4001, {args.mode} mode, {args.batch_norm} batch norm, no dropout, no L2")
    torch.manual_seed(0)
    model = make_model(args.mode, args.batch_norm)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    optimizer = make_optimizer(model, cfg)
    rng = np.random.default_rng(0)
    for epoch in range(1, MAX_EPOCHS + 1):
        train = run_epoch(model, data, args.mode, cfg, "cpu", optimizer, rng)
        print(f"  [{epoch:2d}] training accuracy {train['accuracy']:.1%}  loss {train['loss']:.3f}")
        if train["accuracy"] >= 0.99:
            break

    show("evaluation mode", run_epoch(model, data, args.mode, cfg, "cpu"))
    if args.batch_norm == "standard":
        if args.mode == "sequence":
            show("batch statistics, as in training", evaluate_with_batch_statistics(model, data, cfg))
        recompute_bn_statistics(model.cnn, recording["x"], cfg["epoch_batch_size"])
        show("evaluation mode, averages recomputed", run_epoch(model, data, args.mode, cfg, "cpu"))


if __name__ == "__main__":
    main()