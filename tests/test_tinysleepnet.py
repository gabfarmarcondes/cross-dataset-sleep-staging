"""Unit tests for TinySleepNet: CNN (issue 10) and sequence part (issue 11).
Run from the repository root:

    python -m pytest tests/
"""
import math

import pytest
import torch
from torch import nn

from src.models.tinysleepnet import (
    BN_EPS,
    N_FEATURES,
    EpochClassifier,
    SameMaxPool1d,
    ScaleShift,
    TinySleepNet,
    TinySleepNetCNN,
    count_parameters,
    same_padding,
)

BATCH = 4


@pytest.fixture
def epochs():
    torch.manual_seed(0)
    return torch.randn(BATCH, 1, 3000)


def test_feature_size(epochs):
    """Acceptance criterion: [batch, 1, 3000] -> [batch, 2048] (16 x 128)."""
    cnn = TinySleepNetCNN().eval()
    assert cnn(epochs).shape == (BATCH, N_FEATURES) == (BATCH, 2048)


def test_lengths_after_each_same_layer(epochs):
    """'SAME' keeps ceil(length / stride) steps: 3000 -> 500 -> 63 -> 63 -> 16."""
    lengths, x = [], epochs
    for layer in TinySleepNetCNN().eval().layers:
        x = layer(x)
        if isinstance(layer, (nn.Conv1d, nn.MaxPool1d)):
            lengths.append(x.shape[-1])
    assert lengths == [500, 63, 63, 63, 63, 16]


@pytest.mark.parametrize("length, kernel, stride, expected", [
    (3000, 50, 6, (22, 22)),  # first convolution
    (500, 8, 8, (2, 2)),      # first max pooling
    (63, 8, 1, (3, 4)),       # stride-1 convolutions: odd total, extra sample on the right
    (63, 4, 4, (0, 1)),       # last max pooling
    (64, 4, 4, (0, 0)),       # divisible length: no padding
])
def test_same_padding(length, kernel, stride, expected):
    assert same_padding(length, kernel, stride) == expected


def test_max_pool_ignores_padding():
    """With all-negative input, zero padding would leak a 0 into the last window."""
    x = -torch.ones(1, 1, 63)
    assert torch.equal(SameMaxPool1d(4, 4)(x), -torch.ones(1, 1, 16))


def test_head_gives_one_logit_per_class(epochs):
    assert EpochClassifier().eval()(epochs).shape == (BATCH, 5)


def test_dropout_only_in_training(epochs):
    cnn = TinySleepNetCNN()
    cnn.eval()
    assert torch.equal(cnn(epochs), cnn(epochs))
    cnn.train()
    assert not torch.equal(cnn(epochs), cnn(epochs))


def test_convolutions_have_no_bias():
    convs = [m for m in TinySleepNetCNN().modules() if isinstance(m, nn.Conv1d)]
    assert len(convs) == 4 and all(c.bias is None for c in convs)


def test_parameter_count():
    """4 convolutions (1*50 + 3 * 128*8 inputs per filter) and 4 batch norms (scale, shift)."""
    n = sum(p.numel() for p in TinySleepNetCNN().parameters())
    assert n == 128 * 50 + 3 * 128 * 128 * 8 + 4 * 2 * 128 == 400_640


# --- sequence part (issue 11) --------------------------------------------------------------

def test_sequence_output_shapes():
    model = TinySleepNet().eval()
    logits, (h, c) = model(torch.randn(2, 20, 3000), [20, 20])
    assert logits.shape == (2, 20, 5)
    assert h.shape == c.shape == (1, 2, 128)


def test_carried_state_equals_one_long_sequence():
    """Two minibatches of 20 epochs with the state carried = one pass over the 40 epochs."""
    torch.manual_seed(0)
    model = TinySleepNet().eval()
    x = torch.randn(1, 40, 3000)
    with torch.no_grad():
        whole, _ = model(x, [40])
        first, state = model(x[:, :20], [20])
        second, _ = model(x[:, 20:], [20], state)
    assert torch.allclose(torch.cat([first, second], dim=1), whole, atol=1e-5)


def test_reset_starts_from_zeros():
    model = TinySleepNet().eval()
    x = torch.randn(1, 20, 3000)
    zeros = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
    ones = (torch.ones(1, 1, 128), torch.ones(1, 1, 128))
    with torch.no_grad():
        fresh, _ = model(x, [20])
        assert torch.equal(fresh, model(x, [20], zeros)[0])
        assert not torch.allclose(fresh, model(x, [20], ones)[0])


def test_empty_row_keeps_its_state():
    """A row whose recording has ended (length 0) passes its state through unchanged."""
    model = TinySleepNet().eval()
    state = (torch.randn(1, 2, 128), torch.randn(1, 2, 128))
    with torch.no_grad():
        _, (h, c) = model(torch.randn(2, 20, 3000), [20, 0], state)
    assert torch.equal(h[:, 1], state[0][:, 1]) and torch.equal(c[:, 1], state[1][:, 1])
    assert not torch.equal(h[:, 0], state[0][:, 0])


@pytest.mark.parametrize("batch_norm", ["official", "standard"])
def test_padding_is_ignored_even_by_batch_norm(batch_norm):
    """What comes after a row's length changes nothing, also in training mode."""
    torch.manual_seed(0)
    model = TinySleepNet(dropout=0.0, batch_norm=batch_norm).train()  # no randomness
    x = torch.randn(2, 20, 3000)
    noisy = x.clone()
    noisy[1, 7:] = 100 * torch.randn(13, 3000)
    out_a, (h_a, _) = model(x, [20, 7])
    out_b, (h_b, _) = model(noisy, [20, 7])
    assert torch.allclose(out_a[0], out_b[0], atol=1e-5)
    assert torch.allclose(out_a[1, :7], out_b[1, :7], atol=1e-5)
    assert torch.allclose(h_a, h_b, atol=1e-5)


def test_state_is_detached():
    _, (h, c) = TinySleepNet()(torch.randn(1, 20, 3000), [20])
    assert not h.requires_grad and not c.requires_grad


def test_lstm_init_matches_tensorflow_defaults():
    lstm = TinySleepNet().lstm
    limit = math.sqrt(6 / ((2048 + 128) + 4 * 128))  # Glorot limit of the joint TF kernel
    assert lstm.weight_ih_l0.abs().max() <= limit and lstm.weight_hh_l0.abs().max() <= limit
    forget = slice(128, 256)  # PyTorch gate order: input, forget, cell, output
    assert torch.all(lstm.bias_ih_l0[forget] == 1)
    assert torch.all(lstm.bias_ih_l0[:128] == 0) and torch.all(lstm.bias_ih_l0[256:] == 0)
    assert torch.all(lstm.bias_hh_l0 == 0)


def test_sequence_parameter_count():
    """Official TF config: 1,516,293. Here: no peepholes (-3 x 128), two biases (+4 x 128)."""
    model = TinySleepNet()
    assert count_parameters(model.lstm) == 4 * 128 * (2048 + 128) + 2 * 4 * 128 == 1_115_136
    assert count_parameters(model) == 400_640 + 1_115_136 + (128 * 5 + 5) == 1_516_293 + 128

# --- batch norm: official scale and shift vs. standard (D27) --------------------------------

def test_official_batch_norm_is_a_scale_and_shift():
    """gamma * x / sqrt(1 + eps) + beta, with no running statistics to update."""
    layer = ScaleShift(3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.0, 2.0, -1.0]))
        layer.bias.copy_(torch.tensor([0.0, 0.5, 1.0]))
    x = torch.randn(4, 3, 10)
    expected = x * torch.tensor([1.0, 2.0, -1.0])[:, None] / math.sqrt(1 + BN_EPS)
    expected += torch.tensor([0.0, 0.5, 1.0])[:, None]
    assert torch.allclose(layer(x), expected)
    assert list(layer.buffers()) == []


def test_default_cnn_uses_the_official_batch_norm():
    cnn = TinySleepNetCNN()
    assert sum(isinstance(m, ScaleShift) for m in cnn.modules()) == 4
    assert not any(isinstance(m, nn.BatchNorm1d) for m in cnn.modules())
    standard = TinySleepNetCNN(batch_norm="standard")
    norms = [m for m in standard.modules() if isinstance(m, nn.BatchNorm1d)]
    assert len(norms) == 4 and all(n.eps == BN_EPS and n.momentum == 0.01 for n in norms)
    with pytest.raises(ValueError):
        TinySleepNetCNN(batch_norm="group")


@pytest.mark.parametrize("batch_norm, independent", [("official", True), ("standard", False)])
def test_an_epoch_does_not_depend_on_its_batch(epochs, batch_norm, independent):
    """With the official layer, an epoch gives the same features alone or with others,
    in training mode and in evaluation mode. Standard batch norm does not (D27)."""
    cnn = TinySleepNetCNN(dropout=0.0, batch_norm=batch_norm).train()
    with torch.no_grad():
        in_batch = cnn(epochs)[:1]
        alone = cnn(epochs[:1].repeat(2, 1, 1))[:1]  # 2 copies: BatchNorm1d needs > 1 value
        evaluated = cnn.eval()(epochs)[:1]
    same = torch.allclose(in_batch, alone, atol=1e-5) and torch.allclose(in_batch, evaluated, atol=1e-5)
    assert same == independent