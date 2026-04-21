"""
Signal processing utilities – AC/DC separation and AC-only analysis.

Provides:
- remove_dc(series): subtract mean to isolate AC component
- ac_stats(series): RMS, peak, crest factor on AC signal
- compare_ac_dc(series): dict with both component stats
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ACAnalysisResult:
    original: pd.Series
    ac_component: pd.Series
    dc_offset: float
    ac_rms: float
    ac_peak: float
    ac_peak_to_peak: float
    crest_factor: float      # peak / RMS
    total_rms: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary_dict(self) -> dict:
        return {
            "DC 오프셋 (평균)": round(self.dc_offset, 6),
            "AC RMS": round(self.ac_rms, 6),
            "AC 피크": round(self.ac_peak, 6),
            "AC Peak-to-Peak": round(self.ac_peak_to_peak, 6),
            "크레스트 팩터": round(self.crest_factor, 4),
            "전체 RMS": round(self.total_rms, 6),
        }


def analyze_ac(
    series: pd.Series,
    detrend: bool = False,
) -> ACAnalysisResult:
    """
    Separate DC and AC components and compute AC statistics.

    Args:
        series:  Input signal (numeric).
        detrend: If True, also remove linear trend (in addition to mean).

    Returns:
        ACAnalysisResult.
    """
    s = series.dropna().astype(float).reset_index(drop=True)

    if len(s) < 2:
        return ACAnalysisResult(
            original=s, ac_component=s,
            dc_offset=float("nan"), ac_rms=float("nan"),
            ac_peak=float("nan"), ac_peak_to_peak=float("nan"),
            crest_factor=float("nan"), total_rms=float("nan"),
            error="신호 데이터가 너무 짧습니다 (최소 2개 포인트 필요)."
        )

    dc_offset = float(s.mean())
    total_rms = float(np.sqrt(np.mean(s.values ** 2)))

    ac = s - dc_offset

    if detrend:
        # Remove linear trend from AC component
        try:
            from scipy.signal import detrend as scipy_detrend
            ac = pd.Series(scipy_detrend(ac.values), index=ac.index)
        except ImportError:
            # Manual linear detrend
            x = np.arange(len(ac))
            coeffs = np.polyfit(x, ac.values, 1)
            trend_line = np.polyval(coeffs, x)
            ac = pd.Series(ac.values - trend_line, index=ac.index)

    ac_rms = float(np.sqrt(np.mean(ac.values ** 2)))
    ac_peak = float(np.max(np.abs(ac.values)))
    ac_peak_to_peak = float(ac.max() - ac.min())
    crest_factor = float(ac_peak / ac_rms) if ac_rms > 0 else float("nan")

    return ACAnalysisResult(
        original=s,
        ac_component=ac,
        dc_offset=dc_offset,
        ac_rms=ac_rms,
        ac_peak=ac_peak,
        ac_peak_to_peak=ac_peak_to_peak,
        crest_factor=crest_factor,
        total_rms=total_rms,
    )


def remove_dc(series: pd.Series) -> pd.Series:
    """Return AC-only signal (mean removed)."""
    s = series.astype(float)
    return s - s.mean()
