"""
FFT analysis – thin wrapper around the core frequency module.

Adds DC-removal and detrending pre-processing options on top of
the existing compute_fft / compute_band_stats infrastructure.
"""
from __future__ import annotations

import pandas as pd

from aiviz.analytics.frequency import compute_fft, compute_band_stats, FFTResult, BandStats
from .common import prepare_signal


def run_fft(
    series: pd.Series,
    sample_rate: float = 1.0,
    window: str = "hann",
    n_peaks: int = 5,
    remove_dc: bool = False,
    detrend: bool = False,
) -> FFTResult:
    """
    Compute FFT with optional DC removal and detrending.

    Args:
        series:      Input signal (numeric pd.Series).
        sample_rate: Samples per second (Hz).
        window:      Window function: 'hann', 'hamming', 'blackman', 'none'.
        n_peaks:     Number of top peaks to report.
        remove_dc:   If True, subtract mean before FFT.
        detrend:     If True, remove linear trend before FFT.

    Returns:
        FFTResult (same type as aiviz.analytics.frequency.compute_fft).
    """
    x = prepare_signal(series, remove_dc=remove_dc, detrend=detrend)
    return compute_fft(pd.Series(x), sample_rate=sample_rate, window=window, n_peaks=n_peaks)


__all__ = ["run_fft", "FFTResult", "BandStats", "compute_band_stats"]
