"""
S-Transform (Stockwell Transform) implementation.

The S-Transform is a time-frequency analysis method that:
  - Generalises the STFT with a frequency-adaptive Gaussian window
  - Maintains absolute phase (lossless invertible)
  - Provides multi-resolution similar to CWT but with absolute frequency reference
  - Reduces to STFT at uniform window width (λ → ∞)

Algorithm (FFT-based, O(N_f × N log N)):
  For each analysis frequency f_k:
    1. Shift the signal's spectrum:  Xₛ[n] = X[n + k]  (circular)
    2. Multiply by Gaussian envelope: G[n] = exp(-2π²n²/k²)
    3. IFFT gives S(τ, f_k) for all time steps τ

Reference: Stockwell, Mansinha & Lowe (1996).

Performance note:
  For signals longer than MAX_SAMPLES, the signal is decimated before
  the S-Transform and the effective sample rate is adjusted accordingly.
  The UI displays a notice when this occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .common import prepare_signal, validate_signal

# Safety limit: S-Transform is O(N_f × N log N); beyond this decimation occurs
MAX_SAMPLES: int = 4096


@dataclass
class STransformResult:
    times: np.ndarray           # time axis (seconds)
    freqs: np.ndarray           # frequency axis (Hz)
    ST: np.ndarray              # |S-Transform| matrix (n_freqs × n_times)
    dominant_freqs: np.ndarray  # dominant freq per time step (Hz)
    stats: dict
    available: bool = True
    error: Optional[str] = None
    decimated: bool = False     # True if signal was decimated for speed
    original_n: int = 0         # original sample count before decimation


def compute_s_transform(
    series: pd.Series,
    sample_rate: float = 1.0,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    n_freqs: int = 64,
    remove_dc: bool = False,
    detrend: bool = False,
) -> STransformResult:
    """
    Compute the S-Transform time-frequency representation.

    Args:
        series:      Input signal.
        sample_rate: Samples per second (Hz).
        freq_min:    Minimum frequency (Hz). Default: Fs/N.
        freq_max:    Maximum frequency (Hz). Default: 0.95 × Nyquist.
        n_freqs:     Number of frequency bins to compute.
        remove_dc:   Subtract mean before analysis.
        detrend:     Remove linear trend before analysis.

    Returns:
        STransformResult with time-frequency power matrix.
    """
    x = prepare_signal(series, remove_dc=remove_dc, detrend=detrend)
    validate_signal(x, min_samples=16)
    original_n = len(x)
    decimated = False

    # Decimate if too long (preserve memory / responsiveness)
    if original_n > MAX_SAMPLES:
        step = original_n // MAX_SAMPLES
        x = x[::step]
        sample_rate = sample_rate / step
        decimated = True

    n = len(x)
    nyquist = sample_rate / 2.0

    if freq_max is None or freq_max <= 0:
        freq_max = nyquist * 0.95
    if freq_min is None or freq_min <= 0:
        freq_min = sample_rate / n
    freq_min = max(freq_min, sample_rate / n)
    freq_min = min(freq_min, freq_max * 0.99)

    # Choose analysis frequencies at DFT bin locations (exact circular shift)
    all_pos_freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mask = (all_pos_freqs >= freq_min) & (all_pos_freqs <= freq_max)
    sel_freqs = all_pos_freqs[mask]

    # Sub-sample to n_freqs if needed
    if len(sel_freqs) > n_freqs:
        idx = np.round(np.linspace(0, len(sel_freqs) - 1, n_freqs)).astype(int)
        sel_freqs = sel_freqs[idx]

    if len(sel_freqs) == 0:
        return STransformResult(
            times=np.arange(n) / sample_rate,
            freqs=np.array([]),
            ST=np.zeros((0, n)),
            dominant_freqs=np.zeros(n),
            stats={"method": "S-Transform", "sample_rate": sample_rate, "n_samples": n},
            available=False,
            error="선택된 주파수 범위에 유효한 주파수 빈이 없습니다.",
            decimated=decimated,
            original_n=original_n,
        )

    X = np.fft.fft(x)                                   # full-length DFT
    alpha = np.fft.fftfreq(n, d=1.0 / sample_rate)      # frequency axis for DFT
    times = np.arange(n) / sample_rate

    ST = np.zeros((len(sel_freqs), n), dtype=np.float64)

    for i_f, f in enumerate(sel_freqs):
        if abs(f) < 1e-10:
            ST[i_f, :] = abs(np.mean(x))
            continue

        # Integer DFT bin index for this frequency
        k = int(round(f * n / sample_rate))
        k = max(0, min(k, n - 1))

        # Gaussian in frequency domain: G(α, f) = exp(−2π²α²/f²)
        G = np.exp(-2.0 * np.pi**2 * alpha**2 / (f**2))

        # Circularly-shifted spectrum X(α + f) → shift indices by k
        X_shifted = np.roll(X, -k)

        # IFFT over α gives S(τ, f) for all τ
        voice = np.fft.ifft(X_shifted * G)
        ST[i_f, :] = np.abs(voice)

    # Dominant frequency per time step
    dom_idx = np.argmax(ST, axis=0) if len(sel_freqs) > 0 else np.zeros(n, dtype=int)
    dominant_freqs = sel_freqs[dom_idx] if len(sel_freqs) > 0 else np.zeros(n)

    stats = {
        "method": "S-Transform",
        "sample_rate": sample_rate,
        "n_samples": n,
        "n_freqs": len(sel_freqs),
        "freq_min_hz": float(sel_freqs[0]) if len(sel_freqs) > 0 else 0.0,
        "freq_max_hz": float(sel_freqs[-1]) if len(sel_freqs) > 0 else 0.0,
        "mean_dominant_freq": float(dominant_freqs.mean()) if len(dominant_freqs) > 0 else 0.0,
        "decimated": decimated,
        "original_n": original_n,
    }

    return STransformResult(
        times=times,
        freqs=sel_freqs,
        ST=ST,
        dominant_freqs=dominant_freqs,
        stats=stats,
        available=True,
        decimated=decimated,
        original_n=original_n,
    )
