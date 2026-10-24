"""TinySleepNet in PyTorch: the CNN part (issue 10).

Reimplementation based on the official TensorFlow 1.x code by Akara Supratak,
https://github.com/akaraspt/tinysleepnet (Apache License 2.0): build_cnn in model.py
and the layer helpers in nn.py.

    Supratak, A., & Guo, Y. (2020). TinySleepNet: an efficient deep learning model for
    sleep stage scoring based on raw single-channel EEG. EMBC 2020, 641-644.

Input: 30-s epochs at 100 Hz, shape [batch, 1, 3000].
Output: one feature vector per epoch, shape [batch, 2048] (16 time steps x 128 filters).

    Conv(128, 50, stride 6) -> BN -> ReLU -> MaxPool(8, 8) -> Dropout(0.5)
    3 x [Conv(128, 8, stride 1) -> BN -> ReLU]
    MaxPool(4, 4) -> Flatten -> Dropout(0.5)

Kept from the official code: TensorFlow 'SAME' padding, convolutions without bias,
batch-norm epsilon and momentum, and the weight initialization.

Differences from the official code:
- Batch normalization is real batch normalization: batch statistics in training,
  running statistics in evaluation. The official nn.batch_norm receives is_training
  but never passes it on, so its layers always run in inference mode with mean 0 and
  variance 1 (a learned per-channel scale and shift).
- Flatten order is channel-major ([128, 16]); TensorFlow flattens time-major
  ([16, 128]). It only matters when loading official weights, which we never do.
"""
import math

import torch.nn.functional as F
from torch import nn

N_FILTERS = 128
N_FEATURES = 16 * N_FILTERS  # for a 3000-sample input
N_CLASSES = 5

# tf.layers.batch_normalization(momentum=0.99, epsilon=0.001): TensorFlow keeps 99% of the
# old running average, PyTorch's momentum is the weight of the new batch (1 - 0.99)
BN_EPS = 1e-3
BN_MOMENTUM = 0.01


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


def conv_bn_relu(in_channels, kernel_size, stride):
    return [
        SameConv1d(in_channels, N_FILTERS, kernel_size, stride, bias=False),
        nn.BatchNorm1d(N_FILTERS, eps=BN_EPS, momentum=BN_MOMENTUM),
        nn.ReLU(),
    ]


class TinySleepNetCNN(nn.Module):
    """Feature extractor: [batch, 1, 3000] -> [batch, 2048]."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            # kernel 50 and stride 6 are fs / 2 and fs / 16 at 100 Hz, as in the official code
            *conv_bn_relu(1, kernel_size=50, stride=6),
            SameMaxPool1d(8, 8),
            nn.Dropout(dropout),
            *conv_bn_relu(N_FILTERS, kernel_size=8, stride=1),
            *conv_bn_relu(N_FILTERS, kernel_size=8, stride=1),
            *conv_bn_relu(N_FILTERS, kernel_size=8, stride=1),
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
    (issue 11) puts the LSTM between the CNN and the output layer.
    Returns logits [batch, n_classes]; softmax is left to the loss.
    """

    def __init__(self, n_classes=N_CLASSES, dropout=0.5):
        super().__init__()
        self.cnn = TinySleepNetCNN(dropout)
        self.head = nn.Linear(N_FEATURES, n_classes)
        lecun_normal_(self.head.weight)  # official "softmax_linear": same init, bias 0
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.cnn(x))