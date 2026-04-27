"""
Band power analysis.

Computes the signal power within user-defined or auto-generated
frequency bands. Supports two estimation methods:

  'welch' (default) – Welch averaged periodogram (lower variance)
  'fft'             – Direct windowed FFT (simpler, faster)

Typical use cases:
  - EEG band power (delta / theta / alpha / beta / gamma)
  - Vibration band monitoring
  - Bearing fault detection (sub-band energy ratios)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sci_signal

from .common import prepare_signal, validate_signal


@dataclass
class BandPowerResult:
    bands: list[dict]    # [{"band", "f_low", "f_high", "power", "relative_power"}]
    total_power: float
    method: str          # 'welch' or 'fft'
    freqs: np.ndarray    # underlying frequency axis (for reference)
    psd: np.ndarray      # underlying PSD (for reference)
    stats: dict


def compute_band_power(
    series: pd.Series,
    sample_rate: float = 1.0,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    method: str = "welch",
    nperseg: int = 256,
    remove_dc: bool = False,
    detrend: bool = False,
) -> BandPowerResult:
    """
    Compute power within frequency bands.

    Args:
        series:      Input signal.
        sample_rate: Samples per second (Hz).
        bands:       Dict of {name: (f_low, f_high)}. If None, auto-creates 4 bands.
        method:      'welch' or 'fft'.
        nperseg:     Segment length for Welch method.
        remove_dc:   Subtract mean before analysis.
        detrend:     Remove linear trend before analysis.

    Returns:
        BandPowerResult with per-band power and relative power (%).
    """
    x = prepare_signal(series, remove_dc=remove_dc, detrend=detrend)
    validate_signal(x, min_samples=8)

    n = len(x)
    nyquist = sample_rate / 2.0

    # Default 4-band auto partition
    if bands is None:
        q = nyquist / 4.0
        bands = {
            "Band 1 (0–25%)": (0.0, q),
            "Band 2 (25–50%)": (q, 2.0 * q),
            "Band 3 (50–75%)": (2.0 * q, 3.0 * q),
            "Band 4 (75–Nyq)": (3.0 * q, nyquist),
        }

    # Estimate PSD
    if method == "welch":
        nperseg_eff = min(nperseg, n)
        freqs, psd = sci_signal.welch(
            x, fs=sample_rate, nperseg=nperseg_eff, scaling="density", detrend=False
        )
    else:
        # FFT-based PSD
        win = np.hanning(n)
        X = np.fft.rfft(x * win)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        psd = (np.abs(X) * 2.0 / n) ** 2
        psd[0] /= 2.0  # DC

    # Integrate PSD over each band
    band_results: list[dict] = []
    for name, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not mask.any():
            continue
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None) or (lambda y, x: float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2)))
        power = float(_trapz(psd[mask], freqs[mask]))
        power = max(power, 0.0)
        band_results.append({
            "band": name,
            "f_low": float(f_lo),
            "f_high": float(f_hi),
            "power": power,
            "relative_power": 0.0,   # filled below
        })

    total_power = sum(b["power"] for b in band_results)
    if total_power <= 0:
        total_power = 1.0

    for b in band_results:
        b["relative_power"] = b["power"] / total_power * 100.0

    stats = {
        "method": f"Band Power ({method})",
        "sample_rate": sample_rate,
        "n_samples": n,
        "n_bands": len(band_results),
        "total_power": total_power,
        "nyquist": nyquist,
    }

    return BandPowerResult(
        bands=band_results,
        total_power=total_power,
        method=method,
        freqs=freqs,
        psd=psd,
        stats=stats,
    )
