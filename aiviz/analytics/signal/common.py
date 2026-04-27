"""
Shared utilities and base types for advanced signal analysis.

All functions are pure Python/NumPy – no UI dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sci_signal


def validate_signal(x: np.ndarray, min_samples: int = 8) -> None:
    """
    Raise ValueError if signal is too short or empty.
    Korean-friendly error message.
    """
    n = len(x) if x is not None else 0
    if n < min_samples:
        raise ValueError(
            f"신호는 최소 {min_samples}개 샘플이 필요합니다. "
            f"(현재 유효 샘플 수: {n}개)"
        )


def prepare_signal(
    series: pd.Series,
    remove_dc: bool = False,
    detrend: bool = False,
) -> np.ndarray:
    """
    Convert a pandas Series to a clean float64 NumPy array.

    Steps:
    1. Drop NaN values
    2. Convert to float64
    3. Replace remaining NaN / inf with 0
    4. Optionally remove DC offset (mean subtraction)
    5. Optionally linear-detrend the signal
    """
    x = series.dropna().astype(float).values
    # Replace any remaining non-finite values
    x = np.where(np.isfinite(x), x, 0.0)
    if remove_dc:
        x = x - np.mean(x)
    if detrend:
        x = sci_signal.detrend(x)
    return x


def get_window(name: str, n: int) -> np.ndarray:
    """Return a window function array of length *n*."""
    name = name.lower()
    if name in ("none", "rectangular", "rect"):
        return np.ones(n)
    elif name == "hamming":
        return np.hamming(n)
    elif name == "blackman":
        return np.blackman(n)
    else:  # default: hann
        return np.hanning(n)
