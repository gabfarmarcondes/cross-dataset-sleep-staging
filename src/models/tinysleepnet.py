"""TinySleepNet in PyTorch: CNN (issue 10) and sequence part (issue 11).

Reimplementation based on the official TensorFlow 1.x code by Akara Supratak,
https://github.com/akaraspt/tinysleepnet (Apache License 2.0): build_cnn and
append_rnn in model.py, and the layer helpers in nn.py.

    Supratak, A., & Guo, Y. (2020). TinySleepNet: an efficient deep learning model for
    sleep stage scoring based on raw single-channel EEG. EMBC 2020, 641-644.

CNN: one 30-s epoch at 100 Hz, [batch, 1, 3000] -> a feature vector [batch, 2048]
(16 time steps x 128 filters).

    Conv(128, 50, stride 6) -> BN -> ReLU -> MaxPool(8, 8) -> Dropout(0.5)
    3 x [Conv(128, 8, stride 1) -> BN -> ReLU]
    MaxPool(4, 4) -> Flatten -> Dropout(0.5)

BN is chosen with batch_norm (configs/experiment.yaml, model block):
    "official"  what the official layers actually compute: a learned scale and shift per
                channel, with no batch statistics (default)
    "standard"  torch BatchNorm1d: batch statistics in training, running averages in
                evaluation

Sequence part: the CNN features of 20 consecutive epochs -> LSTM(128) -> Dropout(0.5)
-> Linear(5), one prediction per epoch. The LSTM state is carried across consecutive
minibatches of the same recordings (see src/training/minibatches.py).

Kept from the official code: TensorFlow 'SAME' padding, convolutions without bias,
the effective batch-norm behavior (default), the weight initialization (CNN, LSTM,
output layer), the LSTM forget bias of 1, and dropout on the LSTM output.

About batch norm: the official nn.batch_norm receives is_training but never passes it
on, so tf.layers.batch_normalization always runs in inference mode, with its running
mean and variance stuck at their initial values (0 and 1). Each layer is therefore a
learned scale and shift; this is what ScaleShift computes. Real batch normalization
("standard") was the first choice here, but in the sequence mode each minibatch
holds consecutive epochs of a few recordings, so its batch statistics differ from the
running averages used in evaluation: a model that memorized one recording scored 99% in
training mode and 53-75% in evaluation mode on the same data.

Differences from the official code:
- PyTorch's LSTM has no peephole connections (the official LSTMCell uses them) and keeps
  two bias vectors per gate instead of one: 128 more parameters in total.
- Flatten order is channel-major ([128, 16]); TensorFlow flattens time-major
  ([16, 128]). It only matters when loading official weights, which we never do.
"""
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_FILTERS = 128
N_FEATURES = 16 * N_FILTERS  # for a 3000-sample input
N_CLASSES = 5
N_HIDDEN = 128  # LSTM units

BATCH_NORMS = ("official", "standard")

# tf.layers.batch_normalization(momentum=0.99, epsilon=0.001): TensorFlow keeps 99% of the
# old running average, PyTorch's momentum is the weight of the new batch (1 - 0.99)
BN_EPS = 1e-3
BN_MOMENTUM = 0.01

log = logging.getLogger(__name__)


def same_padding(length, kernel_size, stride):
    """(left, right) padding of TensorFlow 'SAME': the output has ceil(length / stride) steps.

    When the total is odd, TensorFlow puts the extra sample on the right.
    """
    out_length = -(-length // stride)  # ceil division
    total = max((out_length - 1) * stride + kernel_size - length, 0)
    return total // 2, total - total // 2


class SameConv1d(nn.Conv1d):
    """Conv1d with TensorFlow 'SAME' padding (zeros), valid for any stride."""

    def forward(self, x):
        left, right = same_padding(x.shape[-1], self.kernel_size[0], self.stride[0])
        return super().forward(F.pad(x, (left, right)))


class SameMaxPool1d(nn.MaxPool1d):
    """MaxPool1d with TensorFlow 'SAME' padding.

    Padded positions are ignored by the max, as in TensorFlow, so they are filled
    with -inf instead of zeros.
    """

    def forward(self, x):
        left, right = same_padding(x.shape[-1], self.kernel_size, self.stride)
        return super().forward(F.pad(x, (left, right), value=float("-inf")))


def lecun_normal_(weight):
    """TF variance_scaling_initializer(scale=1.0, mode="fan_in", distribution="normal").

    TensorFlow 1.13 draws from a normal truncated at 2 standard deviations and widens
    the standard deviation so that the variance after truncation is 1 / fan_in.
    """
    fan_in = weight[0].numel()  # in_channels * kernel_size, or in_features for Linear
    std = math.sqrt(1.0 / fan_in) / 0.87962566103423978  # std of a normal truncated at +-2
    nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std)


def tf_lstm_init_(lstm):
    """Defaults of the official tf.contrib.rnn.LSTMCell(forget_bias=1.0).

    TensorFlow keeps one kernel [input + hidden, 4 * hidden] with Glorot uniform values and
    one zero bias, and adds 1 to the forget gate. PyTorch splits the kernel in two
    (weight_ih, weight_hh), so both get the Glorot limit of the joint kernel; the forget
    gate bias starts at 1 (PyTorch gate order: input, forget, cell, output).
    """
    hidden = lstm.hidden_size
    limit = math.sqrt(6.0 / ((lstm.input_size + hidden) + 4 * hidden))
    with torch.no_grad():
        for name, param in lstm.named_parameters():
            if name.startswith("weight"):
                param.uniform_(-limit, limit)
            else:
                param.zero_()
        lstm.bias_ih_l0[hidden:2 * hidden] = 1.0


class ScaleShift(nn.Module):
    """What the official batch-norm layers compute: a learned scale and shift per channel.

    tf.layers.batch_normalization in inference mode with its initial statistics (mean 0,
    variance 1) gives gamma * x / sqrt(1 + eps) + beta. The output of an epoch depends only
    on that epoch, never on the rest of the batch, so training and evaluation compute the
    same function.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_channels))  # gamma (scale)
        self.bias = nn.Parameter(torch.zeros(n_channels))  # beta (shift)

    def forward(self, x):  # [batch, channels, time]
        return x * (self.weight / math.sqrt(1 + BN_EPS))[:, None] + self.bias[:, None]


def norm_layer(batch_norm):
    if batch_norm == "official":
        return ScaleShift(N_FILTERS)
    if batch_norm == "standard":
        return nn.BatchNorm1d(N_FILTERS, eps=BN_EPS, momentum=BN_MOMENTUM)
    raise ValueError(f"batch_norm must be one of {BATCH_NORMS}, got {batch_norm!r}")


def conv_bn_relu(in_channels, kernel_size, stride, batch_norm):
    return [
        SameConv1d(in_channels, N_FILTERS, kernel_size, stride, bias=False),
        norm_layer(batch_norm),
        nn.ReLU(),
    ]


class TinySleepNetCNN(nn.Module):
    """Feature extractor: [batch, 1, 3000] -> [batch, 2048]."""

    def __init__(self, dropout=0.5, batch_norm="official"):
        super().__init__()
        self.layers = nn.Sequential(
            # kernel 50 and stride 6 are fs / 2 and fs / 16 at 100 Hz, as in the official code
            *conv_bn_relu(1, kernel_size=50, stride=6, batch_norm=batch_norm),
            SameMaxPool1d(8, 8),
            nn.Dropout(dropout),
            *conv_bn_relu(N_FILTERS, kernel_size=8, stride=1, batch_norm=batch_norm),
            *conv_bn_relu(N_FILTERS, kernel_size=8, stride=1, batch_norm=batch_norm),
            *conv_bn_relu(N_FILTERS, kernel_size=8, stride=1, batch_norm=batch_norm),
            SameMaxPool1d(4, 4),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                lecun_normal_(module.weight)

    def forward(self, x):
        return self.layers(x)


class EpochClassifier(nn.Module):
    """CNN + temporary linear head: one prediction per 30-s epoch, without context.

    Used by the epoch-wise mode of the training loop (issue 12); the sequence model
    (TinySleepNet below) puts the LSTM between the CNN and the output layer.
    Returns logits [batch, n_classes]; softmax is left to the loss.
    """

    def __init__(self, n_classes=N_CLASSES, dropout=0.5, batch_norm="official"):
        super().__init__()
        self.cnn = TinySleepNetCNN(dropout, batch_norm)
        self.head = nn.Linear(N_FEATURES, n_classes)
        lecun_normal_(self.head.weight)  # official "softmax_linear": same init, bias 0
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.cnn(x))


class TinySleepNet(nn.Module):
    """CNN per epoch + LSTM over sequences of epochs: one prediction per epoch, with context.

    forward(x, lengths, state) processes one minibatch from src/training/minibatches.py:
        x        [batch, seq_length, 3000]: row i holds consecutive epochs of one recording
        lengths  [batch]: real epochs in each row (0 to seq_length); the positions after
                 them are padding, skipped by the CNN and by the LSTM
        state    (h, c) returned for the previous minibatch of the same recordings, or None
                 to start from zeros (new recordings)
    Returns logits [batch, seq_length, n_classes], where padded positions must be ignored,
    and the final state. A row with length 0 (its recording has ended) keeps its state.
    The returned state is detached: gradients do not flow back into earlier minibatches,
    as in the official code, where the state is fed back as plain numbers.
    """

    def __init__(self, n_classes=N_CLASSES, hidden_size=N_HIDDEN, dropout=0.5,
                 batch_norm="official"):
        super().__init__()
        self.cnn = TinySleepNetCNN(dropout, batch_norm)
        self.lstm = nn.LSTM(N_FEATURES, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # official DropoutWrapper(output_keep_prob=0.5)
        self.head = nn.Linear(hidden_size, n_classes)
        tf_lstm_init_(self.lstm)
        lecun_normal_(self.head.weight)  # official "softmax_linear": same init, bias 0
        nn.init.zeros_(self.head.bias)

    def forward(self, x, lengths, state=None):
        batch, seq_length = x.shape[:2]
        lengths = torch.as_tensor(lengths, dtype=torch.int64).cpu()  # packing wants CPU lengths
        real = (torch.arange(seq_length) < lengths[:, None]).to(x.device)  # [batch, seq_length]

        # Only real epochs go through the CNN: it saves work, and with standard batch norm
        # the padding zeros would enter the batch statistics
        features = x.new_zeros(batch, seq_length, N_FEATURES)
        features[real] = self.cnn(x[real].unsqueeze(1))

        out, state = self.run_lstm(features, lengths, state)
        return self.head(self.dropout(out)), state

    def run_lstm(self, features, lengths, state=None):
        """LSTM over [batch, seq_length, n_features], stopping each row at its length."""
        batch, seq_length = features.shape[:2]
        if state is None:
            zeros = features.new_zeros(1, batch, self.lstm.hidden_size)
            state = (zeros, zeros)
        # packing runs each row only up to its length, so the final state is the one after
        # the last real epoch; it rejects length 0, so empty rows run one padded step here
        # and get their old state back below
        packed = pack_padded_sequence(
            features, lengths.clamp(min=1), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(packed, state)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=seq_length)

        ended = (lengths == 0).to(features.device)[None, :, None]
        h = torch.where(ended, state[0], h).detach()
        c = torch.where(ended, state[1], c).detach()
        return out, (h, c)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Log the parameter count: python -m src.models.tinysleepnet"""
    from src.read_data import ROOT

    log_path = ROOT / "outputs" / "logs" / "tinysleepnet_parameters.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
    )
    model = TinySleepNet()
    for name, part in [("CNN", model.cnn), ("LSTM", model.lstm), ("output layer", model.head)]:
        log.info(f"{name:<14}{count_parameters(part):>11,}")
    log.info(f"{'total':<14}{count_parameters(model):>11,}")
    log.info("official TensorFlow config: 1,516,293 (LSTM with peepholes and one bias per gate)")


if __name__ == "__main__":
    main()