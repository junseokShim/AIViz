"""
Power Spectral Density (PSD) via Welch's method.

Welch PSD averages periodograms over overlapping windows, giving a
lower-variance estimate of the true PSD compared to a single FFT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sci_signal

from .common import prepare_signal, validate_signal


@dataclass
class PSDResult:
    freqs: np.ndarray        # frequency axis (Hz)
    psd: np.ndarray          # one-sided PSD (V²/Hz)
    dominant_freq: float     # frequency at peak PSD (Hz)
    dominant_power: float    # PSD at dominant frequency
    total_power: float       # total power (integral of PSD)
    stats: dict


def compute_psd(
    series: pd.Series,
    sample_rate: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    remove_dc: bool = False,
    detrend: bool = False,
) -> PSDResult:
    """
    Compute one-sided PSD using Welch's method.

    Args:
        series:      Input signal.
        sample_rate: Samples per second (Hz).
        window:      Window function name ('hann', 'hamming', 'blackman', 'none').
        nperseg:     Length of each FFT segment.
        noverlap:    Overlap between segments. Defaults to nperseg // 2.
        remove_dc:   Subtract mean before analysis.
        detrend:     Remove linear trend before analysis.

    Returns:
        PSDResult with frequency axis, PSD, and statistics.
    """
    x = prepare_signal(series, remove_dc=remove_dc, detrend=detrend)
    validate_signal(x, min_samples=8)

    n = len(x)
    nperseg = min(nperseg, n)
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = min(noverlap, nperseg - 1)

    freqs, psd = sci_signal.welch(
        x,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        detrend=False,  # detrending already applied above if requested
    )

    # Skip DC (index 0) when finding dominant frequency
    start = 1 if len(psd) > 1 else 0
    dom_idx = int(np.argmax(psd[start:]) + start)
    dominant_freq = float(freqs[dom_idx])
    dominant_power = float(psd[dom_idx])

    # Total power by numerical integration
    total_power = float(np.trapezoid(psd, freqs) if hasattr(np, "trapezoid") else np.trapz(psd, freqs))

    stats = {
        "method": "Welch PSD",
        "sample_rate": sample_rate,
        "n_samples": n,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "window": window,
        "dominant_freq": dominant_freq,
        "dominant_power": dominant_power,
        "total_power": total_power,
        "nyquist": sample_rate / 2.0,
        "freq_resolution": float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0,
    }

    return PSDResult(
        freqs=freqs,
        psd=psd,
        dominant_freq=dominant_freq,
        dominant_power=dominant_power,
        total_power=total_power,
        stats=stats,
    )
