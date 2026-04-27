"""
Short-Time Fourier Transform (STFT) / Spectrogram analysis.

Extends the basic spectrogram with:
- configurable window type and overlap
- dominant frequency tracking over time
- summary statistics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sci_signal

from .common import prepare_signal, validate_signal


@dataclass
class STFTResult:
    times: np.ndarray           # time axis (seconds)
    freqs: np.ndarray           # frequency axis (Hz)
    Sxx: np.ndarray             # power matrix (n_freqs × n_times)
    dominant_freqs: np.ndarray  # dominant frequency at each time step (Hz)
    stats: dict


def compute_stft(
    series: pd.Series,
    sample_rate: float = 1.0,
    window: str = "hann",
    nperseg: int = 128,
    noverlap: Optional[int] = None,
    remove_dc: bool = False,
    detrend: bool = False,
) -> STFTResult:
    """
    Compute STFT spectrogram with dominant-frequency tracking.

    Args:
        series:      Input signal.
        sample_rate: Samples per second (Hz).
        window:      Window function name.
        nperseg:     Samples per STFT segment.
        noverlap:    Overlap between adjacent segments (default: 75% of nperseg).
        remove_dc:   Subtract mean before analysis.
        detrend:     Remove linear trend before analysis.

    Returns:
        STFTResult containing the spectrogram matrix and per-time dominant freq.
    """
    x = prepare_signal(series, remove_dc=remove_dc, detrend=detrend)
    validate_signal(x, min_samples=16)

    n = len(x)
    nperseg = min(nperseg, n)

    if noverlap is None:
        noverlap = nperseg * 3 // 4
    noverlap = min(noverlap, nperseg - 1)

    freqs, times, Sxx = sci_signal.spectrogram(
        x,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )

    # Dominant frequency per time step (skip DC bin 0)
    start_bin = 1 if len(freqs) > 1 else 0
    dom_idx = np.argmax(Sxx[start_bin:, :], axis=0) + start_bin
    dominant_freqs = freqs[dom_idx]

    t_res = float(times[1] - times[0]) if len(times) > 1 else 0.0
    f_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0

    stats = {
        "method": "STFT",
        "sample_rate": sample_rate,
        "n_samples": n,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "window": window,
        "n_time_bins": int(len(times)),
        "n_freq_bins": int(len(freqs)),
        "time_resolution_s": t_res,
        "freq_resolution_hz": f_res,
        "nyquist": sample_rate / 2.0,
        "mean_dominant_freq": float(dominant_freqs.mean()) if len(dominant_freqs) > 0 else 0.0,
    }

    return STFTResult(
        times=times,
        freqs=freqs,
        Sxx=Sxx,
        dominant_freqs=dominant_freqs,
        stats=stats,
    )
