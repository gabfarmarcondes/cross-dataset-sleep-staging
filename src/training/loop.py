"""Training loop for TinySleepNet (issue 12).

Follows the official code (https://github.com/akaraspt/tinysleepnet: train.py, model.py,
nn.py), with the settings of the `training` block in configs/experiment.yaml:

- Adam (lr 1e-4, betas 0.9 / 0.999, eps 1e-8);
- L2 penalty 1e-3 * sum(w^2) / 2 on the 4 convolution kernels only, added to the loss
  (not Adam's weight_decay, which would also act on batch norm, LSTM and output layer);
- gradients clipped at a global norm of 5.0;
- cross-entropy weighted by class [1, 1.5, 1, 1, 1] (W, N1, N2, N3, REM), averaged over the
  real epochs of the minibatch (PyTorch's weighted mean divides by the sum of the weights);
- after each training epoch, loss, accuracy and macro-F1 on the validation subjects, which
  come from the source only (D11); the weights with the best validation macro-F1 are kept,
  and training stops after `patience` training epochs without improvement.

Two modes:
    "epoch"     EpochClassifier: CNN + linear head; shuffled minibatches of single epochs
    "sequence"  TinySleepNet: CNN + LSTM; stateful minibatches of 15 recordings x 20 epochs
"""
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.models.tinysleepnet import EpochClassifier, TinySleepNet
from src.training.minibatches import PAD_LABEL, sequence_minibatches

MODES = ("epoch", "sequence")
CSV_COLUMNS = [
    "epoch", "train_loss", "train_accuracy", "train_macro_f1",
    "valid_loss", "valid_accuracy", "valid_macro_f1", "seconds", "best",
]


def make_model(mode, batch_norm="official"):
    if mode == "epoch":
        return EpochClassifier(batch_norm=batch_norm)
    if mode == "sequence":
        return TinySleepNet(batch_norm=batch_norm)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def make_optimizer(model, cfg):
    return torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=tuple(float(b) for b in cfg["betas"]),
        eps=float(cfg["adam_eps"]),
    )


def conv_l2(model):
    """sum(w^2) / 2 over the convolution kernels: tf.nn.l2_loss, as in the official code."""
    return sum((m.weight ** 2).sum() for m in model.modules() if isinstance(m, nn.Conv1d)) / 2


def weighted_loss(logits, y, class_weights):
    """Class-weighted cross-entropy averaged over the real epochs.

    logits [..., n_classes] and y [...], with PAD_LABEL on padded positions. Official formula:
    sum(class_weight[y] * cross_entropy) / number of real epochs.
    """
    logits, y = logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
    real = y != PAD_LABEL
    cross_entropy = F.cross_entropy(logits[real], y[real], reduction="none")
    return (class_weights[y[real]] * cross_entropy).sum() / real.sum()


def prepare(recordings, mode):
    """Data in the form each mode iterates over.

    "sequence": the recordings themselves (each with x [n, 3000] and y [n]).
    "epoch": all epochs of all recordings in two arrays, built once.
    """
    if mode == "sequence":
        return list(recordings)
    return {
        "x": np.concatenate([r["x"] for r in recordings]),
        "y": np.concatenate([r["y"] for r in recordings]).astype(np.int64),
    }


def minibatches(data, mode, cfg, rng=None):
    """Yield (x, y, lengths, reset); shuffled when rng is given (training), in order otherwise."""
    if mode == "sequence":
        order = rng.permutation(len(data)) if rng is not None else None
        yield from sequence_minibatches(data, cfg["batch_size"], cfg["seq_length"], order)
        return
    n = len(data["y"])
    order = rng.permutation(n) if rng is not None else np.arange(n)
    size = cfg["epoch_batch_size"]
    for start in range(0, n, size):
        idx = order[start:start + size]
        yield data["x"][idx], data["y"][idx], None, None


def run_epoch(model, data, mode, cfg, device, optimizer=None, rng=None):
    """One pass over the data.

    With an optimizer: training (shuffled order, dropout on, weights updated).
    Without: evaluation (fixed order, dropout off, batch norm with running statistics).
    Returns the weighted loss, accuracy and macro-F1 over the real epochs, and the labels.
    """
    training = optimizer is not None
    model.train(training)
    class_weights = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)
    loss_sum, n_epochs, trues, preds = 0.0, 0, [], []
    state = None
    with torch.set_grad_enabled(training):
        for x, y, lengths, reset in minibatches(data, mode, cfg, rng if training else None):
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            if mode == "sequence":
                if reset:  # new group of recordings: the LSTM starts from zeros
                    state = None
                logits, state = model(x, lengths, state)
            else:
                logits = model(x.unsqueeze(1))  # [batch, 3000] -> [batch, 1, 3000]
            loss = weighted_loss(logits, y, class_weights)

            if training:
                optimizer.zero_grad()
                (loss + float(cfg["l2_weight_decay"]) * conv_l2(model)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip_grad_norm"]))
                optimizer.step()

            real = y != PAD_LABEL
            n = int(real.sum())
            loss_sum += loss.item() * n
            n_epochs += n
            trues.append(y[real].cpu().numpy())
            preds.append(logits.argmax(dim=-1)[real].cpu().numpy())

    y_true, y_pred = np.concatenate(trues), np.concatenate(preds)
    return {
        "loss": loss_sum / n_epochs,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def fit(model, train_data, valid_data, mode, cfg, run_dir, seed, device, log=print):
    """Train with early stopping on validation macro-F1; write history.csv and best.pt.

    Returns the best training epoch and its validation macro-F1; the model ends with the
    best weights loaded.
    """
    rng = np.random.default_rng(seed)  # shuffling; weight init and dropout use torch's seed
    optimizer = make_optimizer(model, cfg)
    best_f1, best_epoch = -1.0, 0
    best_path = run_dir / "best.pt"

    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            start = time.perf_counter()
            train = run_epoch(model, train_data, mode, cfg, device, optimizer, rng)
            valid = run_epoch(model, valid_data, mode, cfg, device)
            seconds = time.perf_counter() - start

            improved = valid["macro_f1"] > best_f1
            if improved:
                best_f1, best_epoch = valid["macro_f1"], epoch
                torch.save(model.state_dict(), best_path)

            writer.writerow([
                epoch,
                f"{train['loss']:.4f}", f"{train['accuracy']:.4f}", f"{train['macro_f1']:.4f}",
                f"{valid['loss']:.4f}", f"{valid['accuracy']:.4f}", f"{valid['macro_f1']:.4f}",
                f"{seconds:.1f}", int(improved),
            ])
            f.flush()  # the file stays usable if the session dies
            log(f"[{epoch:3d}/{cfg['max_epochs']}] "
                f"train loss {train['loss']:.3f} acc {train['accuracy']:.1%} mf1 {train['macro_f1']:.1%} | "
                f"valid loss {valid['loss']:.3f} acc {valid['accuracy']:.1%} mf1 {valid['macro_f1']:.1%} | "
                f"{seconds:.0f} s{' *' if improved else ''}")

            if epoch - best_epoch >= int(cfg["patience"]):
                log(f"early stopping: no better validation macro-F1 in {cfg['patience']} epochs")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return {"best_epoch": best_epoch, "best_valid_macro_f1": best_f1, "epochs_run": epoch}