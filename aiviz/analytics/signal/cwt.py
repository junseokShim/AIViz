"""
Continuous Wavelet Transform (CWT) analysis.

Uses PyWavelets (pywt) when available.
Falls back gracefully with a clear error message if pywt is not installed.

Supported wavelets (pywt ContinuousWavelet family):
  'morl'   – Morlet (default, good general-purpose)
  'mexh'   – Mexican Hat (Ricker wavelet)
  'gaus1'  – Gaussian derivative (order 1)
  'cgau1'  – Complex Gaussian
  'cmor1-1.5' – Complex Morlet

CWT strengths:
  - Multi-resolution: coarse resolution at low freq, fine at high freq
  - No need to specify window size (automatic)
  - Good for detecting transients and non-stationary features

Limitations vs STFT:
  - Heisenberg uncertainty: can't have both high time AND frequency resolution
  - Magnitude is relative (no physical unit of Hz²/Hz)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .common import prepare_signal, validate_signal

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

# Wavelets supported by the UI selector
SUPPORTED_WAVELETS = ["morl", "mexh", "gaus1", "gaus2", "cgau1", "cmor1-1.5"]
DEFAULT_WAVELET = "morl"


@dataclass
class CWTResult:
    times: np.ndarray                   # time axis (seconds)
    freqs: np.ndarray                   # approximate frequency axis (Hz)
    scales: np.ndarray                  # scale axis
    coefs: np.ndarray                   # CWT power (n_scales × n_samples)
    dominant_freqs: np.ndarray          # dominant freq per time step (Hz)
    high_energy_bands: list             # [(f_low, f_high)] of high-energy regions
    wavelet: str
    stats: dict
    available: bool = True
    error: Optional[str] = None


def compute_cwt(
    series: pd.Series,
    sample_rate: float = 1.0,
    wavelet: str = DEFAULT_WAVELET,
    n_scales: int = 64,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    remove_dc: bool = False,
    detrend: bool = False,
) -> CWTResult:
    """
    Compute CWT scalogram.

    Args:
        series:      Input signal.
        sample_rate: Samples per second.
        wavelet:     ContinuousWavelet name (e.g., 'morl', 'mexh').
        n_scales:    Number of scales (frequency resolution of scalogram).
        freq_min:    Minimum frequency to analyse (Hz). Default: Fs/N.
        freq_max:    Maximum frequency to analyse (Hz). Default: 0.9 * Nyquist.
        remove_dc:   Subtract mean before analysis.
        detrend:     Remove linear trend before analysis.

    Returns:
        CWTResult. If pywt is not installed, returns result with available=False.
    """
    if not HAS_PYWT:
        return CWTResult(
            times=np.array([]),
            freqs=np.array([]),
            scales=np.array([]),
            coefs=np.zeros((1, 1)),
            dominant_freqs=np.array([]),
            high_energy_bands=[],
            wavelet=wavelet,
            stats={"method": "CWT"},
            available=False,
            error=(
                "PyWavelets (pywt) 패키지가 설치되어 있지 않습니다.\n"
                "다음 명령으로 설치하세요:\n  pip install PyWavelets"
            ),
        )

    x = prepare_signal(series, remove_dc=remove_dc, detrend=detrend)
    validate_signal(x, min_samples=16)
    n = len(x)

    # Validate / normalise wavelet name
    try:
        wav = pywt.ContinuousWavelet(wavelet)
    except Exception:
        wavelet = DEFAULT_WAVELET
        wav = pywt.ContinuousWavelet(wavelet)

    nyquist = sample_rate / 2.0
    if freq_max is None or freq_max <= 0:
        freq_max = nyquist * 0.9
    if freq_min is None or freq_min <= 0:
        freq_min = max(sample_rate / n, freq_max / 200.0)
    freq_min = min(freq_min, freq_max * 0.99)

    # Map frequencies to scales
    # pywt: freq = scale2frequency(wav, scale) * sample_rate
    # => scale = central_frequency(wav) * sample_rate / freq
    central_freq = pywt.central_frequency(wav)
    scale_for_fmax = central_freq * sample_rate / freq_max
    scale_for_fmin = central_freq * sample_rate / freq_min

    scale_min = max(scale_for_fmax, 1.0)
    scale_max = min(scale_for_fmin, n / 2.0)

    if scale_min >= scale_max:
        scale_min = 1.0
        scale_max = min(float(n) / 4.0, 128.0)

    scales = np.geomspace(scale_min, scale_max, n_scales)

    # Compute CWT (may take a few seconds for large n)
    coefs_complex, freqs_raw = pywt.cwt(
        x, scales, wavelet, sampling_period=1.0 / sample_rate
    )
    coefs = np.abs(coefs_complex)   # magnitude (n_scales × n_samples)
    freqs = np.abs(freqs_raw)

    times = np.arange(n) / sample_rate

    # Dominant frequency per time step (row with max power for each column)
    dom_scale_idx = np.argmax(coefs, axis=0)
    dominant_freqs = freqs[dom_scale_idx]

    # High-energy bands: rows where total energy > 10% of max row energy
    row_energy = coefs.sum(axis=1)
    threshold = row_energy.max() * 0.1 if row_energy.max() > 0 else 0.0
    he_mask = row_energy >= threshold
    he_freqs = freqs[he_mask]
    high_energy_bands = (
        [(float(he_freqs.min()), float(he_freqs.max()))] if len(he_freqs) > 0 else []
    )

    stats = {
        "method": "CWT",
        "wavelet": wavelet,
        "sample_rate": sample_rate,
        "n_samples": n,
        "n_scales": n_scales,
        "freq_min_hz": float(freqs.min()),
        "freq_max_hz": float(freqs.max()),
        "high_energy_bands": high_energy_bands,
        "mean_dominant_freq": float(dominant_freqs.mean()),
    }

    return CWTResult(
        times=times,
        freqs=freqs,
        scales=scales,
        coefs=coefs,
        dominant_freqs=dominant_freqs,
        high_energy_bands=high_energy_bands,
        wavelet=wavelet,
        stats=stats,
        available=True,
    )
