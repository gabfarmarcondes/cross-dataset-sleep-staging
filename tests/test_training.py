"""Unit tests for the training loop (issue 12). Run from the repository root:

    python -m pytest tests/
"""
import csv
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data.splits import SPLITS_DIR
from src.training import loop
from src.training.loop import conv_l2, fit, make_model, make_optimizer, prepare, run_epoch, weighted_loss
from src.training.minibatches import PAD_LABEL
from src.training.train import run_name, select_subjects

WEIGHTS = torch.tensor([1.0, 1.5, 1.0, 1.0, 1.0])
CFG = {  # experiment.yaml values, except smaller batches for the toy data
    "max_epochs": 200, "patience": 50, "learning_rate": 1.0e-4, "betas": [0.9, 0.999],
    "adam_eps": 1.0e-8, "l2_weight_decay": 1.0e-3, "clip_grad_norm": 5.0,
    "class_weights": [1.0, 1.5, 1.0, 1.0, 1.0], "batch_size": 4, "seq_length": 20,
    "epoch_batch_size": 32,
}


def toy_recordings(n_recordings, n_epochs, seed=0):
    """Learnable toy data: each stage is a sine wave of its own frequency, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(3000) / 100
    freqs = np.array([2.0, 5.0, 9.0, 13.0, 20.0])
    recordings = []
    for _ in range(n_recordings):
        y = rng.integers(0, 5, n_epochs)
        phase = rng.uniform(0, 2 * np.pi, (n_epochs, 1))
        x = np.sin(2 * np.pi * freqs[y][:, None] * t + phase)
        x += 0.5 * rng.standard_normal((n_epochs, 3000))
        recordings.append({"x": x.astype(np.float32), "y": y})
    return recordings


# --- loss and regularization ----------------------------------------------------------------

def test_weighted_loss_divides_by_real_epochs():
    """Official formula: sum(weight * CE) / n epochs, not / sum of weights (PyTorch's mean)."""
    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    y = torch.tensor([0, 1, 1, 2])
    ce = F.cross_entropy(logits, y, reduction="none")
    expected = (ce * torch.tensor([1.0, 1.5, 1.5, 1.0])).sum() / 4
    assert torch.isclose(weighted_loss(logits, y, WEIGHTS), expected)
    assert not torch.isclose(expected, F.cross_entropy(logits, y, weight=WEIGHTS))


def test_padding_does_not_enter_the_loss():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    y = torch.tensor([[0, 2, 4], [3, PAD_LABEL, PAD_LABEL]])
    real = y != PAD_LABEL
    assert torch.isclose(weighted_loss(logits, y, WEIGHTS), weighted_loss(logits[real], y[real], WEIGHTS))


def test_l2_only_on_convolution_kernels():
    model = make_model("sequence")
    convs = [model.cnn.layers[i] for i in (0, 5, 8, 11)]
    assert all(isinstance(c, torch.nn.Conv1d) for c in convs)
    expected = sum((c.weight ** 2).sum() for c in convs) / 2
    assert torch.isclose(conv_l2(model), expected)
    with torch.no_grad():  # LSTM and output layer are not regularized
        model.lstm.weight_ih_l0.mul_(10)
        model.head.weight.mul_(10)
    assert torch.isclose(conv_l2(model), expected)


# --- data selection -------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(SPLITS_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_training_never_sees_test_subjects(path):
    splits = json.loads(path.read_text(encoding="utf-8"))
    for i, fold in enumerate(splits["in_domain"]["folds"]):
        train, valid = select_subjects(splits, fold=i)
        assert (train, valid) == (fold["train"], fold["valid"])
        assert not set(train + valid) & set(fold["test"])
    train, valid = select_subjects(splits, source=True)
    assert sorted(train + valid) == sorted(int(s) for s in splits["recordings"])


# --- training and evaluation ----------------------------------------------------------------

# epochs of training needed to pass 90% with seeds 0-2: 3-4 (epoch), 14-21 (sequence)
@pytest.mark.parametrize("mode, n_epochs", [("epoch", 15), ("sequence", 30)])
def test_training_learns_a_toy_task(mode, n_epochs):
    torch.manual_seed(0)
    model = make_model(mode)
    data = prepare(toy_recordings(4, 50), mode)
    optimizer = make_optimizer(model, CFG)
    rng = np.random.default_rng(0)
    first = run_epoch(model, data, mode, CFG, "cpu", optimizer, rng)
    for _ in range(n_epochs - 1):
        last = run_epoch(model, data, mode, CFG, "cpu", optimizer, rng)
    evaluated = run_epoch(model, data, mode, CFG, "cpu")
    assert last["loss"] < first["loss"]
    assert evaluated["accuracy"] > 0.9


def test_evaluation_changes_nothing():
    """No weight update and no batch-norm statistics update outside training."""
    model = make_model("sequence", "standard")  # the option that has running statistics
    before = {k: v.clone() for k, v in model.state_dict().items()}
    run_epoch(model, prepare(toy_recordings(2, 30), "sequence"), "sequence", CFG, "cpu")
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_fit_stops_early_and_restores_the_best(tmp_path, monkeypatch):
    """Scripted validation macro-F1: best at epoch 2, patience 3 -> stops after epoch 5."""
    scores = iter([0.50, 0.60, 0.55, 0.58, 0.59, 0.90])

    def fake_run_epoch(model, data, mode, cfg, device, optimizer=None, rng=None):
        if optimizer is None:  # validation
            f1 = next(scores)
            with torch.no_grad():
                model.head.bias.fill_(f1)  # marks the weights of this epoch
            return {"loss": 1.0, "accuracy": f1, "macro_f1": f1}
        return {"loss": 1.0, "accuracy": 0.5, "macro_f1": 0.5}

    monkeypatch.setattr(loop, "run_epoch", fake_run_epoch)
    model = make_model("epoch")
    cfg = {**CFG, "max_epochs": 10, "patience": 3}
    result = fit(model, None, None, "epoch", cfg, tmp_path, 0, "cpu", log=lambda message: None)
    assert result == {"best_epoch": 2, "best_valid_macro_f1": 0.60, "epochs_run": 5}
    with open(tmp_path / "history.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["best"] for r in rows] == ["1", "1", "0", "0", "0"]
    assert torch.allclose(model.head.bias, torch.full((5,), 0.60))  # best weights loaded back

def test_run_name_marks_non_default_settings():
    assert run_name("sleep_edf", "sequence", 0, fold=0) == "sleep_edf_fold0_sequence_seed0"
    assert run_name("hmc", "epoch", 2, source=True, filtered=False, batch_norm="standard") == \
        "hmc_source_epoch_seed2_nofilter_bnstandard"