"""
Frequency-Domain Analysis panel.

FFT, amplitude/power spectrum, peak detection, band energy, spectrogram, AI insight.
"""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QCheckBox, QDoubleSpinBox,
    QSlider, QSplitter, QTabWidget,
)
from PyQt6.QtCore import Qt

from aiviz.app.controller import AppController, WorkerThread
from aiviz.ui.widgets.plot_widget import PlotWidget
from aiviz.ui.widgets.data_table import DataTableView
from aiviz.visualization import mpl_charts
from aiviz.analytics.frequency import compute_fft, compute_band_stats, compute_spectrogram
from aiviz.utils.helpers import numeric_columns


class FrequencyPanel(QWidget):
    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._df: pd.DataFrame | None = None
        self._fft_result = None
        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        hdr = QLabel("Frequency-Domain Analysis (FFT)")
        hdr.setObjectName("heading")
        root.addWidget(hdr)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: controls ──────────────────────────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(230)
        cl = QVBoxLayout(ctrl)
        cl.setContentsMargins(0, 0, 8, 0)

        grp_col = QGroupBox("Signal Column")
        g1 = QVBoxLayout(grp_col)
        self._col_combo = QComboBox()
        g1.addWidget(self._col_combo)
        cl.addWidget(grp_col)

        grp_sr = QGroupBox("Sample Rate (Hz)")
        g2 = QVBoxLayout(grp_sr)
        self._sr_spin = QDoubleSpinBox()
        self._sr_spin.setRange(0.001, 1_000_000)
        self._sr_spin.setValue(1.0)
        self._sr_spin.setDecimals(3)
        g2.addWidget(QLabel("Set actual rate or 1.0 for normalised"))
        g2.addWidget(self._sr_spin)
        cl.addWidget(grp_sr)

        grp_win = QGroupBox("Window Function")
        gw = QVBoxLayout(grp_win)
        self._win_combo = QComboBox()
        self._win_combo.addItems(["hann", "hamming", "blackman", "none"])
        gw.addWidget(self._win_combo)
        cl.addWidget(grp_win)

        self._log_chk = QCheckBox("Log-scale Y axis")
        self._log_chk.setChecked(True)
        cl.addWidget(self._log_chk)

        self._run_btn = QPushButton("Run FFT")
        self._run_btn.setEnabled(False)
        cl.addWidget(self._run_btn)

        self._ai_btn = QPushButton("🤖 AI Insight")
        self._ai_btn.setObjectName("secondary")
        self._ai_btn.setEnabled(False)
        cl.addWidget(self._ai_btn)

        cl.addStretch()

        grp_stft = QGroupBox("Spectrogram Segment Size")
        gstft = QVBoxLayout(grp_stft)
        self._seg_slider = QSlider(Qt.Orientation.Horizontal)
        self._seg_slider.setRange(8, 512)
        self._seg_slider.setValue(64)
        self._seg_lbl = QLabel("64")
        self._seg_slider.valueChanged.connect(lambda v: self._seg_lbl.setText(str(v)))
        gstft.addWidget(self._seg_slider)
        gstft.addWidget(self._seg_lbl)
        cl.addWidget(grp_stft)

        splitter.addWidget(ctrl)

        # ── Right: tabs ─────────────────────────────────────────────────
        tabs = QTabWidget()

        self._amp_plot = PlotWidget(figsize=(10, 5))
        tabs.addTab(self._amp_plot, "Amplitude Spectrum")

        self._pow_plot = PlotWidget(figsize=(10, 5))
        tabs.addTab(self._pow_plot, "Power Spectrum")

        self._band_plot = PlotWidget(figsize=(10, 4))
        tabs.addTab(self._band_plot, "Band Energy")

        self._stft_plot = PlotWidget(figsize=(10, 5))
        tabs.addTab(self._stft_plot, "Spectrogram (STFT)")

        self._peaks_table = DataTableView()
        tabs.addTab(self._peaks_table, "Peaks")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Metrics row above splitter
        metrics_row = QHBoxLayout()
        self._m_labels: dict[str, QLabel] = {}
        for k in ["Dominant Freq", "Amplitude", "Total Power", "RMS"]:
            g = QGroupBox(k)
            gl = QVBoxLayout(g)
            v = QLabel("–")
            v.setObjectName("heading")
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gl.addWidget(v)
            self._m_labels[k] = v
            metrics_row.addWidget(g)
        root.insertLayout(1, metrics_row)

        self._run_btn.clicked.connect(self._run_fft)
        self._ai_btn.clicked.connect(self._run_ai)

    # ------------------------------------------------------------------

    def _on_data_loaded(self, result) -> None:
        if not result.ok or result.df is None:
            self._df = None
            self._run_btn.setEnabled(False)
            return
        self._df = result.df
        cols = numeric_columns(self._df)
        self._col_combo.clear()
        self._col_combo.addItems(cols)
        self._run_btn.setEnabled(bool(cols))

    def _run_fft(self) -> None:
        if self._df is None:
            return
        col = self._col_combo.currentText()
        sr = self._sr_spin.value()
        win = self._win_combo.currentText()
        log = self._log_chk.isChecked()
        series = self._df[col].dropna()

        if len(series) < 4:
            self._ctrl.log_message.emit("[ERROR] Signal must have ≥4 samples for FFT.")
            return

        try:
            self._fft_result = compute_fft(series, sample_rate=sr, window=win)
        except Exception as exc:
            self._ctrl.log_message.emit(f"[ERROR] FFT: {exc}")
            return

        r = self._fft_result
        self._m_labels["Dominant Freq"].setText(f"{r.dominant_freq:.4g} Hz")
        self._m_labels["Amplitude"].setText(f"{r.dominant_amplitude:.4g}")
        self._m_labels["Total Power"].setText(f"{r.stats['total_power']:.4g}")
        self._m_labels["RMS"].setText(f"{r.stats['rms']:.4g}")

        ax = self._amp_plot.get_ax()
        mpl_charts.plot_fft_amplitude(ax, r, log_scale=log)
        self._amp_plot.redraw()

        ax = self._pow_plot.get_ax()
        mpl_charts.plot_fft_power(ax, r, log_scale=log)
        self._pow_plot.redraw()

        # Band energy
        try:
            bands = compute_band_stats(r)
            ax = self._band_plot.get_ax()
            mpl_charts.plot_band_energy(ax, bands)
            self._band_plot.redraw()
        except Exception:
            pass

        # Spectrogram
        try:
            seg = self._seg_slider.value()
            stft = compute_spectrogram(series, sample_rate=sr, nperseg=seg)
            ax = self._stft_plot.get_ax()
            mpl_charts.plot_spectrogram(ax, stft)
            self._stft_plot.redraw()
        except Exception:
            pass

        # Peaks table
        if not r.peaks.empty:
            self._peaks_table.load(r.peaks.round(6))

        self._ai_btn.setEnabled(True)

    def _run_ai(self) -> None:
        if self._fft_result is None:
            return
        from aiviz.ai.agent import AnalysisAgent
        col = self._col_combo.currentText()
        agent = AnalysisAgent()
        self._ai_btn.setEnabled(False)
        self._ai_btn.setText("Asking AI…")

        stats = self._fft_result.stats

        def call():
            return agent.explain_fft(stats, col)

        worker = WorkerThread(call)
        worker.result_ready.connect(self._show_ai)
        worker.error_occurred.connect(lambda _: self._reset_ai_btn())
        worker.start()
        self._worker = worker

    def _show_ai(self, result) -> None:
        self._reset_ai_btn()
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("AI Frequency Insight")
        msg.setText(result.answer)
        msg.exec()

    def _reset_ai_btn(self) -> None:
        self._ai_btn.setEnabled(True)
        self._ai_btn.setText("🤖 AI Insight")
