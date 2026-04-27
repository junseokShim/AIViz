"""
Advanced Frequency / Time-Frequency Analysis Panel.

Supported methods:
  FFT             – single-sided amplitude/power spectrum
  PSD (Welch)     – averaged Welch periodogram
  STFT            – Short-Time Fourier Transform spectrogram
  CWT             – Continuous Wavelet Transform scalogram
  S-Transform     – Stockwell Transform (time-frequency)
  Band Power      – per-band energy analysis
  Envelope Spectrum – Hilbert-based envelope FFT
  Cepstrum        – real cepstrum for harmonic detection

UI layout:
  ┌─────────────────────────────────────────────┐
  │ [Method metrics row]                        │
  ├──────────────┬──────────────────────────────┤
  │ Controls     │ Tabs: Plot | Table | Summary  │
  │ (250px)      │       | AI Insight            │
  └──────────────┴──────────────────────────────┘
"""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QCheckBox, QDoubleSpinBox,
    QSpinBox, QSplitter, QTabWidget, QTextEdit, QStackedWidget,
    QFormLayout, QSlider,
)
from PyQt6.QtCore import Qt

from aiviz.app.controller import AppController, WorkerThread
from aiviz.ui.widgets.plot_widget import PlotWidget
from aiviz.ui.widgets.data_table import DataTableView
from aiviz.ui.widgets.insight_panel import InsightPanel
from aiviz.visualization import mpl_charts
from aiviz.utils.helpers import numeric_columns
from aiviz.utils.schema_utils import safe_col

# ── Method index constants ────────────────────────────────────────────────
_M_FFT      = 0
_M_PSD      = 1
_M_STFT     = 2
_M_CWT      = 3
_M_ST       = 4
_M_BANDPOW  = 5
_M_ENVELOPE = 6
_M_CEPSTRUM = 7

_METHOD_NAMES = [
    "FFT",
    "PSD (Welch)",
    "STFT (스펙트로그램)",
    "CWT (스칼로그램)",
    "S-Transform",
    "Band Power",
    "Envelope Spectrum",
    "Cepstrum",
]


class FrequencyPanel(QWidget):
    """Advanced signal analysis panel."""

    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._df: pd.DataFrame | None = None
        self._result = None          # last analysis result (any type)
        self._current_method: int = _M_FFT
        self._worker = None

        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    # ─────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        hdr = QLabel("고급 주파수 / 시간-주파수 분석")
        hdr.setObjectName("heading")
        root.addWidget(hdr)

        # ── Metrics row ─────────────────────────────────────────────────
        metrics_row = QHBoxLayout()
        self._m_labels: dict[str, QLabel] = {}
        for k in ["지배 주파수", "값", "총 전력", "RMS"]:
            g = QGroupBox(k)
            gl = QVBoxLayout(g)
            v = QLabel("–")
            v.setObjectName("heading")
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gl.addWidget(v)
            self._m_labels[k] = v
            metrics_row.addWidget(g)
        root.addLayout(metrics_row)

        # ── Main splitter ────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: controls ──────────────────────────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(260)
        cl = QVBoxLayout(ctrl)
        cl.setContentsMargins(0, 0, 8, 0)
        cl.setSpacing(6)

        # Column selector
        grp_col = QGroupBox("신호 컬럼")
        g1 = QVBoxLayout(grp_col)
        self._col_combo = QComboBox()
        g1.addWidget(self._col_combo)
        cl.addWidget(grp_col)

        # Sample rate
        grp_sr = QGroupBox("샘플링 레이트 (Hz)")
        g2 = QVBoxLayout(grp_sr)
        self._sr_spin = QDoubleSpinBox()
        self._sr_spin.setRange(0.001, 1_000_000)
        self._sr_spin.setValue(1.0)
        self._sr_spin.setDecimals(3)
        g2.addWidget(QLabel("실제 레이트 또는 1.0 (정규화)"))
        g2.addWidget(self._sr_spin)
        cl.addWidget(grp_sr)

        # Method selector
        grp_method = QGroupBox("분석 방법")
        gm = QVBoxLayout(grp_method)
        self._method_combo = QComboBox()
        self._method_combo.addItems(_METHOD_NAMES)
        gm.addWidget(self._method_combo)
        cl.addWidget(grp_method)

        # Per-method parameters (stacked)
        grp_params = QGroupBox("파라미터")
        gp = QVBoxLayout(grp_params)
        self._params_stack = QStackedWidget()
        gp.addWidget(self._params_stack)
        cl.addWidget(grp_params)

        self._build_param_pages()  # populate stacked widget

        # AC-only checkbox
        self._ac_chk = QCheckBox("AC-only (DC 오프셋 제거)")
        self._ac_chk.setToolTip("분석 전 신호 평균을 빼서 DC 성분을 제거합니다")
        cl.addWidget(self._ac_chk)

        # Run / AI buttons
        self._run_btn = QPushButton("분석 실행")
        self._run_btn.setEnabled(False)
        cl.addWidget(self._run_btn)

        self._ai_btn = QPushButton("🤖 AI 인사이트")
        self._ai_btn.setObjectName("secondary")
        self._ai_btn.setEnabled(False)
        cl.addWidget(self._ai_btn)

        cl.addStretch()
        splitter.addWidget(ctrl)

        # ── Right: tabs ─────────────────────────────────────────────────
        tabs = QTabWidget()

        self._main_plot = PlotWidget(figsize=(10, 5))
        tabs.addTab(self._main_plot, "주 시각화")

        self._sec_plot = PlotWidget(figsize=(10, 3))
        tabs.addTab(self._sec_plot, "보조 시각화")

        self._result_table = DataTableView()
        tabs.addTab(self._result_table, "결과 테이블")

        self._summary_edit = QTextEdit()
        self._summary_edit.setReadOnly(True)
        self._summary_edit.setFont(self._summary_edit.font())
        tabs.addTab(self._summary_edit, "요약")

        self._insight = InsightPanel("🤖 주파수 AI 인사이트")
        tabs.addTab(self._insight, "AI 인사이트")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # ── Connections ──────────────────────────────────────────────────
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._run_btn.clicked.connect(self._run_analysis)
        self._ai_btn.clicked.connect(self._run_ai)

    # ── Per-method parameter pages ──────────────────────────────────────

    def _build_param_pages(self) -> None:
        """Create one QWidget page per method for the QStackedWidget."""

        # ── FFT ──────────────────────────────────────────────────────────
        p_fft = QWidget()
        fl = QFormLayout(p_fft)
        self._fft_window = QComboBox()
        self._fft_window.addItems(["hann", "hamming", "blackman", "none"])
        fl.addRow("Window:", self._fft_window)
        self._fft_peaks = QSpinBox()
        self._fft_peaks.setRange(1, 20)
        self._fft_peaks.setValue(5)
        fl.addRow("피크 수:", self._fft_peaks)
        self._fft_log = QCheckBox("Log Y축")
        self._fft_log.setChecked(True)
        fl.addRow(self._fft_log)
        self._params_stack.addWidget(p_fft)   # index 0

        # ── PSD (Welch) ───────────────────────────────────────────────────
        p_psd = QWidget()
        pl = QFormLayout(p_psd)
        self._psd_window = QComboBox()
        self._psd_window.addItems(["hann", "hamming", "blackman", "none"])
        pl.addRow("Window:", self._psd_window)
        self._psd_nperseg = QSpinBox()
        self._psd_nperseg.setRange(16, 4096)
        self._psd_nperseg.setValue(256)
        pl.addRow("NPerseg:", self._psd_nperseg)
        self._psd_log = QCheckBox("Log Y축")
        self._psd_log.setChecked(True)
        pl.addRow(self._psd_log)
        self._params_stack.addWidget(p_psd)   # index 1

        # ── STFT ─────────────────────────────────────────────────────────
        p_stft = QWidget()
        sl = QFormLayout(p_stft)
        self._stft_window = QComboBox()
        self._stft_window.addItems(["hann", "hamming", "blackman", "none"])
        sl.addRow("Window:", self._stft_window)
        self._stft_nperseg = QSpinBox()
        self._stft_nperseg.setRange(16, 2048)
        self._stft_nperseg.setValue(128)
        sl.addRow("NPerseg:", self._stft_nperseg)
        self._stft_overlap = QSpinBox()
        self._stft_overlap.setRange(0, 99)
        self._stft_overlap.setValue(75)
        self._stft_overlap.setSuffix("%")
        sl.addRow("Overlap %:", self._stft_overlap)
        self._params_stack.addWidget(p_stft)  # index 2

        # ── CWT ──────────────────────────────────────────────────────────
        p_cwt = QWidget()
        cwl = QFormLayout(p_cwt)
        self._cwt_wavelet = QComboBox()
        self._cwt_wavelet.addItems(["morl", "mexh", "gaus1", "gaus2", "cgau1"])
        cwl.addRow("Wavelet:", self._cwt_wavelet)
        self._cwt_scales = QSpinBox()
        self._cwt_scales.setRange(16, 256)
        self._cwt_scales.setValue(64)
        cwl.addRow("스케일 수:", self._cwt_scales)
        self._cwt_fmin = QDoubleSpinBox()
        self._cwt_fmin.setRange(0, 1e6)
        self._cwt_fmin.setValue(0.0)
        self._cwt_fmin.setSpecialValueText("자동")
        cwl.addRow("Freq Min (Hz):", self._cwt_fmin)
        self._cwt_fmax = QDoubleSpinBox()
        self._cwt_fmax.setRange(0, 1e6)
        self._cwt_fmax.setValue(0.0)
        self._cwt_fmax.setSpecialValueText("자동")
        cwl.addRow("Freq Max (Hz):", self._cwt_fmax)
        self._params_stack.addWidget(p_cwt)   # index 3

        # ── S-Transform ──────────────────────────────────────────────────
        p_st = QWidget()
        stl = QFormLayout(p_st)
        self._st_nfreqs = QSpinBox()
        self._st_nfreqs.setRange(16, 256)
        self._st_nfreqs.setValue(64)
        stl.addRow("주파수 빈 수:", self._st_nfreqs)
        self._st_fmin = QDoubleSpinBox()
        self._st_fmin.setRange(0, 1e6)
        self._st_fmin.setValue(0.0)
        self._st_fmin.setSpecialValueText("자동")
        stl.addRow("Freq Min (Hz):", self._st_fmin)
        self._st_fmax = QDoubleSpinBox()
        self._st_fmax.setRange(0, 1e6)
        self._st_fmax.setValue(0.0)
        self._st_fmax.setSpecialValueText("자동")
        stl.addRow("Freq Max (Hz):", self._st_fmax)
        note = QLabel("※ 신호가 길면 자동 다운샘플링됩니다.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        stl.addRow(note)
        self._params_stack.addWidget(p_st)    # index 4

        # ── Band Power ───────────────────────────────────────────────────
        p_bp = QWidget()
        bpl = QFormLayout(p_bp)
        self._bp_method = QComboBox()
        self._bp_method.addItems(["welch", "fft"])
        bpl.addRow("추정 방법:", self._bp_method)
        self._bp_nperseg = QSpinBox()
        self._bp_nperseg.setRange(16, 4096)
        self._bp_nperseg.setValue(256)
        bpl.addRow("NPerseg:", self._bp_nperseg)
        self._params_stack.addWidget(p_bp)    # index 5

        # ── Envelope Spectrum ────────────────────────────────────────────
        p_env = QWidget()
        evl = QFormLayout(p_env)
        self._env_window = QComboBox()
        self._env_window.addItems(["hann", "hamming", "blackman", "none"])
        evl.addRow("Window:", self._env_window)
        self._env_log = QCheckBox("Log Y축")
        evl.addRow(self._env_log)
        self._params_stack.addWidget(p_env)   # index 6

        # ── Cepstrum ─────────────────────────────────────────────────────
        p_cep = QWidget()
        cpl = QFormLayout(p_cep)
        self._cep_window = QComboBox()
        self._cep_window.addItems(["hann", "hamming", "blackman", "none"])
        cpl.addRow("Window:", self._cep_window)
        self._params_stack.addWidget(p_cep)   # index 7

    # ─────────────────────────────────────────────────────────────────────
    # Slots
    # ─────────────────────────────────────────────────────────────────────

    def _on_method_changed(self, index: int) -> None:
        self._current_method = index
        self._params_stack.setCurrentIndex(index)

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
        self._ai_btn.setEnabled(False)
        self._result = None

    # ─────────────────────────────────────────────────────────────────────
    # Analysis dispatch
    # ─────────────────────────────────────────────────────────────────────

    def _run_analysis(self) -> None:
        if self._df is None:
            return
        col = self._col_combo.currentText()
        if not col:
            return

        raw = safe_col(self._df, col)
        if raw is None:
            self._ctrl.log_message.emit(f"[ERROR] 컬럼 '{col}'을 찾을 수 없습니다.")
            return

        series = raw.dropna()
        if len(series) < 8:
            self._ctrl.log_message.emit("[ERROR] 유효 샘플이 너무 적습니다 (최소 8개 필요).")
            return

        sr = self._sr_spin.value()
        ac = self._ac_chk.isChecked()
        m = self._current_method

        self._run_btn.setEnabled(False)
        self._run_btn.setText("분석 중…")

        def do_work():
            if m == _M_FFT:
                return self._compute_fft(series, sr, ac)
            elif m == _M_PSD:
                return self._compute_psd(series, sr, ac)
            elif m == _M_STFT:
                return self._compute_stft(series, sr, ac)
            elif m == _M_CWT:
                return self._compute_cwt(series, sr, ac)
            elif m == _M_ST:
                return self._compute_st(series, sr, ac)
            elif m == _M_BANDPOW:
                return self._compute_band_power(series, sr, ac)
            elif m == _M_ENVELOPE:
                return self._compute_envelope(series, sr, ac)
            elif m == _M_CEPSTRUM:
                return self._compute_cepstrum(series, sr, ac)
            return None

        worker = WorkerThread(do_work)
        worker.result_ready.connect(lambda res: self._on_result(res, col))
        worker.error_occurred.connect(self._on_error)
        worker.start()
        self._worker = worker

    # ── Compute functions ────────────────────────────────────────────────

    def _compute_fft(self, series, sr, ac):
        from aiviz.analytics.signal.fft import run_fft
        return ("fft", run_fft(series, sample_rate=sr,
                               window=self._fft_window.currentText(),
                               n_peaks=self._fft_peaks.value(),
                               remove_dc=ac))

    def _compute_psd(self, series, sr, ac):
        from aiviz.analytics.signal.psd import compute_psd
        return ("psd", compute_psd(series, sample_rate=sr,
                                   window=self._psd_window.currentText(),
                                   nperseg=self._psd_nperseg.value(),
                                   remove_dc=ac))

    def _compute_stft(self, series, sr, ac):
        from aiviz.analytics.signal.stft import compute_stft
        nperseg = self._stft_nperseg.value()
        overlap_pct = self._stft_overlap.value() / 100.0
        noverlap = int(nperseg * overlap_pct)
        return ("stft", compute_stft(series, sample_rate=sr,
                                     window=self._stft_window.currentText(),
                                     nperseg=nperseg, noverlap=noverlap,
                                     remove_dc=ac))

    def _compute_cwt(self, series, sr, ac):
        from aiviz.analytics.signal.cwt import compute_cwt
        fmin = self._cwt_fmin.value() or None
        fmax = self._cwt_fmax.value() or None
        return ("cwt", compute_cwt(series, sample_rate=sr,
                                   wavelet=self._cwt_wavelet.currentText(),
                                   n_scales=self._cwt_scales.value(),
                                   freq_min=fmin, freq_max=fmax,
                                   remove_dc=ac))

    def _compute_st(self, series, sr, ac):
        from aiviz.analytics.signal.s_transform import compute_s_transform
        fmin = self._st_fmin.value() or None
        fmax = self._st_fmax.value() or None
        return ("st", compute_s_transform(series, sample_rate=sr,
                                          freq_min=fmin, freq_max=fmax,
                                          n_freqs=self._st_nfreqs.value(),
                                          remove_dc=ac))

    def _compute_band_power(self, series, sr, ac):
        from aiviz.analytics.signal.band_power import compute_band_power
        return ("bandpower", compute_band_power(series, sample_rate=sr,
                                                method=self._bp_method.currentText(),
                                                nperseg=self._bp_nperseg.value(),
                                                remove_dc=ac))

    def _compute_envelope(self, series, sr, ac):
        from aiviz.analytics.signal.envelope import compute_envelope_spectrum
        return ("envelope", compute_envelope_spectrum(series, sample_rate=sr,
                                                       window=self._env_window.currentText(),
                                                       remove_dc=ac))

    def _compute_cepstrum(self, series, sr, ac):
        from aiviz.analytics.signal.envelope import compute_cepstrum
        return ("cepstrum", compute_cepstrum(series, sample_rate=sr,
                                              window=self._cep_window.currentText(),
                                              remove_dc=ac))

    # ── Result handling ──────────────────────────────────────────────────

    def _on_result(self, tagged_result, col: str) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText("분석 실행")

        if tagged_result is None:
            return

        kind, result = tagged_result
        self._result = (kind, result, col)

        try:
            if kind == "fft":
                self._render_fft(result)
            elif kind == "psd":
                self._render_psd(result)
            elif kind == "stft":
                self._render_stft(result)
            elif kind == "cwt":
                self._render_cwt(result)
            elif kind == "st":
                self._render_st(result)
            elif kind == "bandpower":
                self._render_band_power(result)
            elif kind == "envelope":
                self._render_envelope(result)
            elif kind == "cepstrum":
                self._render_cepstrum(result)
        except Exception as exc:
            self._ctrl.log_message.emit(f"[ERROR] 렌더링 오류: {exc}")
            return

        self._ai_btn.setEnabled(True)
        self._ctrl.log_message.emit(
            f"[OK] {_METHOD_NAMES[self._current_method]} 완료: '{col}'"
        )

    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText("분석 실행")
        self._ctrl.log_message.emit(f"[ERROR] 분석 오류: {msg}")

    # ─────────────────────────────────────────────────────────────────────
    # Per-method rendering
    # ─────────────────────────────────────────────────────────────────────

    def _render_fft(self, r) -> None:
        import numpy as np
        from aiviz.analytics.frequency import compute_band_stats
        from aiviz.analytics.signal_processing_service import analyze_ac

        # Main: amplitude spectrum
        ax = self._main_plot.get_ax()
        mpl_charts.plot_fft_amplitude(ax, r, log_scale=self._fft_log.isChecked())
        self._main_plot.redraw()

        # Secondary: power spectrum
        ax2 = self._sec_plot.get_ax()
        mpl_charts.plot_fft_power(ax2, r, log_scale=self._fft_log.isChecked())
        self._sec_plot.redraw()

        # Table: peaks
        if not r.peaks.empty:
            self._result_table.load(r.peaks.round(6))

        # Metrics
        self._m_labels["지배 주파수"].setText(f"{r.dominant_freq:.4g} Hz")
        self._m_labels["값"].setText(f"{r.dominant_amplitude:.4g}")
        self._m_labels["총 전력"].setText(f"{r.stats['total_power']:.4g}")
        self._m_labels["RMS"].setText(f"{r.stats['rms']:.4g}")

        # Summary
        bands = compute_band_stats(r)
        band_lines = "\n".join(
            f"  {b.band_name}: {b.energy_fraction*100:.1f}% 에너지" for b in bands
        )
        self._summary_edit.setPlainText(
            f"방법: FFT\n"
            f"샘플 수: {r.stats['n_samples']}\n"
            f"샘플링 레이트: {r.stats['sample_rate']} Hz\n"
            f"주파수 해상도: {r.stats['freq_resolution']:.4g} Hz\n"
            f"Nyquist: {r.stats['nyquist']:.4g} Hz\n"
            f"지배 주파수: {r.dominant_freq:.4g} Hz\n"
            f"지배 진폭: {r.dominant_amplitude:.4g}\n"
            f"총 전력: {r.stats['total_power']:.4g}\n"
            f"RMS: {r.stats['rms']:.4g}\n\n"
            f"밴드 에너지:\n{band_lines}"
        )

    def _render_psd(self, r) -> None:
        ax = self._main_plot.get_ax()
        mpl_charts.plot_psd(ax, r, log_scale=self._psd_log.isChecked())
        self._main_plot.redraw()

        # Secondary: band energy via PSD
        from aiviz.analytics.signal.band_power import compute_band_power
        ax2 = self._sec_plot.get_ax()
        ax2.clear()
        ax2.fill_between(r.freqs, r.psd, alpha=0.3, color="#cba6f7")
        ax2.plot(r.freqs, r.psd, color="#cba6f7", linewidth=1.0)
        ax2.set_title("PSD 전체 뷰")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("PSD")
        self._sec_plot.redraw()

        import pandas as pd
        stats_df = pd.DataFrame(list(r.stats.items()), columns=["항목", "값"])
        self._result_table.load(stats_df)

        self._m_labels["지배 주파수"].setText(f"{r.dominant_freq:.4g} Hz")
        self._m_labels["값"].setText(f"{r.dominant_power:.4g} V²/Hz")
        self._m_labels["총 전력"].setText(f"{r.total_power:.4g}")
        self._m_labels["RMS"].setText("–")

        self._summary_edit.setPlainText(
            f"방법: Welch PSD\n"
            + "\n".join(f"  {k}: {v}" for k, v in r.stats.items())
        )

    def _render_stft(self, r) -> None:
        # Main: spectrogram
        ax = self._main_plot.get_ax()
        mpl_charts.plot_spectrogram(ax, r)
        self._main_plot.redraw()

        # Secondary: dominant frequency over time
        ax2 = self._sec_plot.get_ax()
        mpl_charts.plot_dominant_freq_over_time(
            ax2, r.times, r.dominant_freqs, "시간별 지배 주파수 (STFT)"
        )
        self._sec_plot.redraw()

        import pandas as pd
        stats_df = pd.DataFrame(list(r.stats.items()), columns=["항목", "값"])
        self._result_table.load(stats_df)

        self._m_labels["지배 주파수"].setText(
            f"{r.stats.get('mean_dominant_freq', 0):.4g} Hz (평균)"
        )
        self._m_labels["값"].setText(f"{r.stats.get('freq_resolution_hz', 0):.4g} Hz 해상도")
        self._m_labels["총 전력"].setText(f"{r.stats.get('n_time_bins', 0)} 시간 빈")
        self._m_labels["RMS"].setText(f"{r.stats.get('n_freq_bins', 0)} 주파수 빈")

        self._summary_edit.setPlainText(
            "방법: STFT 스펙트로그램\n"
            + "\n".join(f"  {k}: {v}" for k, v in r.stats.items())
        )

    def _render_cwt(self, r) -> None:
        if not r.available:
            self._ctrl.log_message.emit(f"[경고] CWT: {r.error}")

        ax = self._main_plot.get_ax()
        mpl_charts.plot_cwt_scalogram(ax, r)
        self._main_plot.redraw()

        if r.available and len(r.dominant_freqs) > 0:
            ax2 = self._sec_plot.get_ax()
            mpl_charts.plot_dominant_freq_over_time(
                ax2, r.times, r.dominant_freqs, "시간별 지배 주파수 (CWT)"
            )
            self._sec_plot.redraw()

        import pandas as pd
        stats_df = pd.DataFrame(list(r.stats.items()), columns=["항목", "값"])
        self._result_table.load(stats_df)

        if r.available:
            self._m_labels["지배 주파수"].setText(
                f"{r.stats.get('mean_dominant_freq', 0):.4g} Hz (평균)"
            )
            self._m_labels["값"].setText(r.wavelet)
            self._m_labels["총 전력"].setText(f"{r.stats.get('n_scales', 0)} 스케일")
            self._m_labels["RMS"].setText(
                f"{r.stats.get('freq_min_hz', 0):.3g}–{r.stats.get('freq_max_hz', 0):.3g} Hz"
            )
        else:
            for lbl in self._m_labels.values():
                lbl.setText("–")

        self._summary_edit.setPlainText(
            "방법: CWT (연속 웨이블릿 변환)\n"
            + (r.error if not r.available else
               "\n".join(f"  {k}: {v}" for k, v in r.stats.items()))
        )

    def _render_st(self, r) -> None:
        if not r.available:
            self._ctrl.log_message.emit(f"[경고] S-Transform: {r.error}")

        ax = self._main_plot.get_ax()
        mpl_charts.plot_s_transform(ax, r)
        self._main_plot.redraw()

        if r.available and len(r.dominant_freqs) > 0:
            ax2 = self._sec_plot.get_ax()
            mpl_charts.plot_dominant_freq_over_time(
                ax2, r.times, r.dominant_freqs, "시간별 지배 주파수 (S-Transform)"
            )
            self._sec_plot.redraw()

        import pandas as pd
        stats_df = pd.DataFrame(list(r.stats.items()), columns=["항목", "값"])
        self._result_table.load(stats_df)

        if r.available:
            self._m_labels["지배 주파수"].setText(
                f"{r.stats.get('mean_dominant_freq', 0):.4g} Hz (평균)"
            )
            self._m_labels["값"].setText(
                f"{r.stats.get('freq_min_hz', 0):.3g}–{r.stats.get('freq_max_hz', 0):.3g} Hz"
            )
            dec_note = " [다운샘플링]" if r.decimated else ""
            self._m_labels["총 전력"].setText(f"{r.stats.get('n_freqs', 0)} 주파수 빈{dec_note}")
            self._m_labels["RMS"].setText(f"{r.stats.get('n_samples', 0)} 샘플")

        self._summary_edit.setPlainText(
            "방법: S-Transform (Stockwell Transform)\n"
            + (r.error if not r.available else
               "\n".join(f"  {k}: {v}" for k, v in r.stats.items()))
        )

    def _render_band_power(self, r) -> None:
        ax = self._main_plot.get_ax()
        mpl_charts.plot_new_band_power(ax, r)
        self._main_plot.redraw()

        # Secondary: PSD with band overlay
        ax2 = self._sec_plot.get_ax()
        mpl_charts.plot_psd(ax2, r, log_scale=False)
        self._sec_plot.redraw()

        import pandas as pd
        bands_df = pd.DataFrame(r.bands)
        if not bands_df.empty:
            bands_df = bands_df.round(6)
        self._result_table.load(bands_df)

        if r.bands:
            dom = max(r.bands, key=lambda b: b["power"])
            self._m_labels["지배 주파수"].setText(dom["band"])
            self._m_labels["값"].setText(f"{dom['relative_power']:.1f}%")
        self._m_labels["총 전력"].setText(f"{r.total_power:.4g}")
        self._m_labels["RMS"].setText(r.method)

        self._summary_edit.setPlainText(
            f"방법: Band Power ({r.method})\n총 전력: {r.total_power:.4g}\n\n"
            + "\n".join(
                f"  {b['band']}: {b['power']:.4g} ({b['relative_power']:.1f}%)"
                for b in r.bands
            )
        )

    def _render_envelope(self, r) -> None:
        ax = self._main_plot.get_ax()
        mpl_charts.plot_envelope_spectrum(ax, r, log_scale=self._env_log.isChecked())
        self._main_plot.redraw()

        # Secondary: time-domain envelope
        ax2 = self._sec_plot.get_ax()
        import numpy as np
        t = r.times
        env = r.envelope
        t_ds, e_ds, _ = mpl_charts._safe_downsample(t, env)
        ax2.plot(t_ds, e_ds, color="#a6e3a1", linewidth=1.0)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Envelope")
        ax2.set_title("Time-domain Envelope")
        self._sec_plot.redraw()

        import pandas as pd
        stats_df = pd.DataFrame(list(r.stats.items()), columns=["항목", "값"])
        self._result_table.load(stats_df)

        self._m_labels["지배 주파수"].setText(f"{r.dominant_freq:.4g} Hz")
        self._m_labels["값"].setText(f"{r.stats.get('envelope_rms', 0):.4g} (Env RMS)")
        self._m_labels["총 전력"].setText(f"{r.stats.get('envelope_peak', 0):.4g} (Env Peak)")
        self._m_labels["RMS"].setText("Envelope")

        self._summary_edit.setPlainText(
            "방법: Envelope Spectrum (Hilbert)\n"
            + "\n".join(f"  {k}: {v}" for k, v in r.stats.items())
        )

    def _render_cepstrum(self, r) -> None:
        ax = self._main_plot.get_ax()
        mpl_charts.plot_cepstrum(ax, r)
        self._main_plot.redraw()

        # Clear secondary
        ax2 = self._sec_plot.get_ax()
        ax2.text(0.5, 0.5, "케프스트럼 분석 완료\n보조 시각화 없음",
                 ha="center", va="center", transform=ax2.transAxes, color="#cdd6f4")
        self._sec_plot.redraw()

        import pandas as pd
        stats_df = pd.DataFrame(list(r.stats.items()), columns=["항목", "값"])
        self._result_table.load(stats_df)

        self._m_labels["지배 주파수"].setText(f"{r.dominant_pitch:.4g} Hz (추정 기본파)")
        self._m_labels["값"].setText(
            f"{r.dominant_quefrency * 1000:.2f} ms (quefrency)"
        )
        self._m_labels["총 전력"].setText("–")
        self._m_labels["RMS"].setText("Cepstrum")

        self._summary_edit.setPlainText(
            "방법: Real Cepstrum\n"
            + "\n".join(f"  {k}: {v}" for k, v in r.stats.items())
        )

    # ─────────────────────────────────────────────────────────────────────
    # AI insight
    # ─────────────────────────────────────────────────────────────────────

    def _run_ai(self) -> None:
        if self._result is None:
            return
        kind, result, col = self._result

        from aiviz.ai.agent import AnalysisAgent
        agent = AnalysisAgent()

        self._ai_btn.setEnabled(False)
        self._ai_btn.setText("AI 요청 중…")

        def call():
            if kind == "fft":
                return agent.explain_fft(result.stats, col)
            elif kind == "psd":
                return agent.explain_psd(result.stats, col)
            elif kind == "stft":
                return agent.explain_stft(result.stats, col)
            elif kind == "cwt":
                return agent.explain_cwt(result.stats, col)
            elif kind == "st":
                return agent.explain_s_transform(result.stats, col)
            elif kind == "bandpower":
                return agent.explain_band_power(result.bands, col, result.method)
            elif kind == "envelope":
                return agent.explain_envelope(result.stats, col)
            elif kind == "cepstrum":
                return agent.explain_fft(result.stats, col)  # reuse general prompt
            return None

        worker = WorkerThread(call)
        worker.result_ready.connect(self._show_ai)
        worker.error_occurred.connect(lambda _: self._reset_ai_btn())
        worker.start()
        self._ai_worker = worker

    def _show_ai(self, result) -> None:
        self._reset_ai_btn()
        if result is None:
            return
        self._insight.set_text(result.answer)

    def _reset_ai_btn(self) -> None:
        self._ai_btn.setEnabled(True)
        self._ai_btn.setText("🤖 AI 인사이트")
