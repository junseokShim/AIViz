"""
Envelope spectrum and cepstrum analysis.

Envelope spectrum:
  Uses the Hilbert transform to extract the signal envelope, then computes
  the FFT of the envelope. Widely used for rolling-element bearing fault
  detection (BPFI, BPFO, BSF, FTF detection).

Cepstrum:
  Cepstrum = IFFT(log(|FFT(x)|²))
  The real cepstrum detects periodicity in the log-power spectrum,
  corresponding to harmonic families or echo delays.
  'Dominant quefrency' gives the fundamental period of any harmonic series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sci_signal

from .common import prepare_signal, validate_signal, get_window


@dataclass
class EnvelopeResult:
    times: np.ndarray           # time axis (seconds)
    envelope: np.ndarray        # time-domain envelope |analytic(x)|
    freqs: np.ndarray           # frequency axis for envelope FFT (Hz)
    envelope_spectrum: np.ndarray  # amplitude spectrum of envelope
    dominant_freq: float        # dominant frequency in envelope spectrum (Hz)
    stats: dict


@dataclass
class CepstrumResult:
    quefrency: np.ndarray   # quefrency axis (seconds) – like a "time axis in cepstral domain"
    cepstrum: np.ndarray    # real cepstrum values
    dominant_quefrency: float   # quefrency at peak (seconds)
    dominant_pitch: float       # 1 / dominant_quefrency (Hz) – estimated fundamental
    stats: dict


def compute_envelope_spectrum(
    series: pd.Series,
    sample_rate: float = 1.0,
    window: str = "hann",
    n_peaks: int = 5,
    remove_dc: bool = False,
    detrend: bool = False,
) -> EnvelopeResult:
    """
    Compute envelope spectrum via Hilbert transform.

    Steps:
      1. Compute the analytic signal using scipy.signal.hilbert().
      2. Take the magnitude as the time-domain envelope.
      3. Remove DC offset from envelope (necessary for meaningful FFT).
      4. Apply window and compute FFT of the DC-removed envelope.

    Args:
        series:      Input signal (band-pass filtered recommended for best results).
        sample_rate: Samples per second (Hz).
        window:      Window function for envelope FFT.
        n_peaks:     Unused (kept for API consistency).
        remove_dc:   Subtract mean from raw signal before analysis.
        detrend:     Remove linear trend from raw signal before analysis.

    Returns:
        EnvelopeResult with envelope time signal and its frequency spectrum.
    """
    x = prepare_signal(series, remove_dc=remove_dc, detrend=detrend)
    validate_signal(x, min_samples=16)
    n = len(x)

    # Analytic signal and amplitude envelope
    analytic = sci_signal.hilbert(x)
    envelope = np.abs(analytic)
    times = np.arange(n) / sample_rate

    # Remove DC from envelope before taking FFT
    env_ac = envelope - envelope.mean()

    # Windowed FFT of envelope
    win = get_window(window, n)
    fft_vals = np.fft.rfft(env_ac * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    amp = np.abs(fft_vals) * 2.0 / n
    amp[0] /= 2.0  # DC

    # Dominant frequency (skip DC)
    start = 1 if len(amp) > 1 else 0
    dom_idx = int(np.argmax(amp[start:]) + start)
    dominant_freq = float(freqs[dom_idx])

    stats = {
        "method": "Envelope Spectrum (Hilbert)",
        "sample_rate": sample_rate,
        "n_samples": n,
        "dominant_freq": dominant_freq,
        "envelope_mean": float(envelope.mean()),
        "envelope_rms": float(np.sqrt(np.mean(envelope ** 2))),
        "envelope_peak": float(envelope.max()),
        "nyquist": sample_rate / 2.0,
    }

    return EnvelopeResult(
        times=times,
        envelope=envelope,
        freqs=freqs,
        envelope_spectrum=amp,
        dominant_freq=dominant_freq,
        stats=stats,
    )


def compute_cepstrum(
    series: pd.Series,
    sample_rate: float = 1.0,
    window: str = "hann",
    remove_dc: bool = False,
    detrend: bool = False,
) -> CepstrumResult:
    """
    Compute the real cepstrum.

    Formula: c(τ) = IFFT(log(|FFT(x)|²))

    Args:
        series:      Input signal.
        sample_rate: Samples per second (Hz).
        window:      Window applied before FFT.
        remove_dc:   Subtract mean before analysis.
        detrend:     Remove linear trend before analysis.

    Returns:
        CepstrumResult with quefrency axis and cepstrum values.
    """
    x = prepare_signal(series, remove_dc=remove_dc, detrend=detrend)
    validate_signal(x, min_samples=16)
    n = len(x)

    win = get_window(window, n)
    x_win = x * win

    # Log power spectrum → IFFT = real cepstrum
    X = np.fft.fft(x_win, n=n)
    log_power = np.log(np.abs(X) ** 2 + 1e-12)
    cepstrum_full = np.fft.ifft(log_power).real

    # Positive quefrency half only
    n_half = n // 2 + 1
    quefrency = np.arange(n_half) / sample_rate
    cepstrum = cepstrum_full[:n_half]

    # Find dominant quefrency: skip first few samples (< 1/nyquist period)
    min_skip = max(2, int(1.0 / (sample_rate / 2.0) * sample_rate))
    if len(cepstrum) > min_skip + 1:
        dom_idx = int(np.argmax(np.abs(cepstrum[min_skip:])) + min_skip)
        dominant_quefrency = float(quefrency[dom_idx])
        dominant_pitch = float(1.0 / dominant_quefrency) if dominant_quefrency > 1e-10 else 0.0
    else:
        dominant_quefrency = 0.0
        dominant_pitch = 0.0

    stats = {
        "method": "Real Cepstrum",
        "sample_rate": sample_rate,
        "n_samples": n,
        "dominant_quefrency_s": dominant_quefrency,
        "dominant_pitch_hz": dominant_pitch,
    }

    return CepstrumResult(
        quefrency=quefrency,
        cepstrum=cepstrum,
        dominant_quefrency=dominant_quefrency,
        dominant_pitch=dominant_pitch,
        stats=stats,
    )
