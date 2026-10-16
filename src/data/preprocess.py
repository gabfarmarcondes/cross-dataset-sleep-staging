# Signal operations shared by the dataset loaders."""
from fractions import Fraction

from scipy.signal import resample_poly

from src.data.format import FS


def resample_to_target(signal, fs_original):
    """Resample a 1-D signal to FS (100 Hz); resample_poly applies an anti-aliasing filter."""
    if fs_original == FS:
        return signal
    ratio = Fraction(FS, int(fs_original))  # 100/200 -> 1/2; 100/256 -> 25/64
    return resample_poly(signal, ratio.numerator, ratio.denominator)