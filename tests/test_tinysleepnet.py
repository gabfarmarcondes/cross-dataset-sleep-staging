"""Unit tests for the TinySleepNet CNN (issue 10). Run from the repository root:

    python -m pytest tests/
"""
import pytest
import torch
from torch import nn

from src.models.tinysleepnet import (
    N_FEATURES,
    EpochClassifier,
    SameMaxPool1d,
    TinySleepNetCNN,
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