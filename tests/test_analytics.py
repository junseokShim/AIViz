"""
Tests for analytics modules: timeseries, frequency, summary, image.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from aiviz.analytics.timeseries import analyze_series, multi_series_stats
from aiviz.analytics.frequency import compute_fft, compute_band_stats, compute_spectrogram
from aiviz.analytics.summary import compute_summary
from aiviz.analytics.image_analysis import analyze_image


# ---------------------------------------------------------------------------
# Time-series tests
# ---------------------------------------------------------------------------

class TestTimeSeries:
    def _make_series(self, n=100, freq=0.05, noise=0.1):
        t = np.arange(n)
        return pd.Series(np.sin(2 * np.pi * freq * t) + np.random.normal(0, noise, n))

    def test_basic_analysis(self):
        s = self._make_series()
        result = analyze_series(s, window=10)
        assert len(result.original) == len(s)
        assert len(result.rolling_mean) == len(s)
        assert len(result.rolling_std) == len(s)
        assert len(result.anomalies) == len(s)

    def test_trend_slope_sign(self):
        # Clearly increasing signal
        s = pd.Series(np.arange(50, dtype=float))
        result = analyze_series(s, window=5)
        assert result.trend_slope > 0, "Slope must be positive for increasing signal"

    def test_anomaly_detection(self):
        s = pd.Series(np.zeros(100))
        s[50] = 100.0  # obvious spike
        result = analyze_series(s, window=10, anomaly_sigma=2.0)
        assert result.anomalies[50], "Index 50 should be flagged as anomaly"

    def test_smoothed_output(self):
        s = self._make_series(n=100)
        result = analyze_series(s, window=10, smooth=True)
        assert result.smoothed is not None
        assert len(result.smoothed) == len(s)

    def test_multi_series_stats(self):
        df = pd.DataFrame({
            "a": np.arange(50, dtype=float),
            "b": np.arange(50, dtype=float)[::-1],
        })
        stats = multi_series_stats(df, ["a", "b"])
        assert "a" in stats.index
        assert "b" in stats.index
        assert stats.loc["a", "trend_slope"] > 0
        assert stats.loc["b", "trend_slope"] < 0


# ---------------------------------------------------------------------------
# FFT / Frequency tests
# ---------------------------------------------------------------------------

class TestFrequency:
    def _make_signal(self, fs=100, f=10, n=1000):
        """Clean sine wave at frequency f Hz."""
        t = np.arange(n) / fs
        return pd.Series(np.sin(2 * np.pi * f * t))

    def test_fft_basic(self):
        s = self._make_signal()
        result = compute_fft(s, sample_rate=100.0)
        assert len(result.freqs) > 0
        assert len(result.amplitude) == len(result.freqs)
        assert len(result.power) == len(result.freqs)

    def test_dominant_freq_detection(self):
        """FFT should find the dominant frequency within 1 Hz of the true freq."""
        fs = 1000.0
        f_true = 50.0
        s = self._make_signal(fs=int(fs), f=f_true, n=2000)
        result = compute_fft(s, sample_rate=fs)
        assert abs(result.dominant_freq - f_true) < 1.0, (
            f"Dominant freq {result.dominant_freq} not close to {f_true}"
        )

    def test_band_stats(self):
        s = self._make_signal()
        result = compute_fft(s, sample_rate=100.0)
        bands = compute_band_stats(result)
        assert len(bands) > 0
        total_frac = sum(b.energy_fraction for b in bands)
        assert abs(total_frac - 1.0) < 0.05, "Band fractions should sum to ~1"

    def test_spectrogram(self):
        s = self._make_signal(n=512)
        result = compute_spectrogram(s, sample_rate=100.0, nperseg=32)
        assert result.Sxx.ndim == 2
        assert result.Sxx.shape[0] == len(result.freqs)
        assert result.Sxx.shape[1] == len(result.times)

    def test_fft_requires_minimum_samples(self):
        with pytest.raises(ValueError):
            compute_fft(pd.Series([1.0, 2.0]), sample_rate=1.0)

    def test_window_functions(self):
        s = self._make_signal()
        for win in ["hann", "hamming", "blackman", "none"]:
            result = compute_fft(s, sample_rate=100.0, window=win)
            assert result.stats["window"] == win


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestSummary:
    def test_compute_summary(self):
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "cat": np.random.choice(["x", "y", "z"], 100),
        })
        summary = compute_summary(df)
        assert summary.schema.n_rows == 100
        assert not summary.numeric_stats.empty
        assert summary.correlation is not None

    def test_summary_text(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        summary = compute_summary(df)
        text = summary.to_text()
        assert "rows" in text
        assert "Correlation" in text or "Statistics" in text


# ---------------------------------------------------------------------------
# Image analysis tests
# ---------------------------------------------------------------------------

class TestImageAnalysis:
    def _make_image(self, mode="RGB", size=(50, 50)):
        from PIL import Image
        img = Image.new(mode, size, color=(128, 64, 200) if mode == "RGB" else 128)
        return img

    def test_rgb_analysis(self):
        img = self._make_image("RGB")
        result = analyze_image(img, "test.png")
        assert result.width == 50
        assert result.height == 50
        assert result.mode == "RGB"
        assert result.n_channels == 3
        assert not result.is_grayscale
        assert not result.has_transparency

    def test_grayscale_analysis(self):
        img = self._make_image("L")
        result = analyze_image(img, "gray.png")
        assert result.is_grayscale
        assert result.n_channels == 1
        assert "L" in result.histograms

    def test_channel_stats_shape(self):
        img = self._make_image("RGB")
        result = analyze_image(img)
        assert len(result.channel_stats) == 3
        assert "mean" in result.channel_stats.columns

    def test_dominant_colors(self):
        img = self._make_image("RGB")
        result = analyze_image(img, n_colors=4)
        assert result.dominant_colors is not None
        assert len(result.dominant_colors) <= 4
