"""
Time-Series Analysis panel.

Rolling stats, smoothing, anomaly detection, trend, multi-signal comparison.
"""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QCheckBox, QSlider, QDoubleSpinBox,
    QListWidget, QAbstractItemView, QSplitter, QTabWidget,
)
from PyQt6.QtCore import Qt

from aiviz.app.controller import AppController, WorkerThread
from aiviz.ui.widgets.plot_widget import PlotWidget
from aiviz.ui.widgets.data_table import DataTableView
from aiviz.visualization import mpl_charts
from aiviz.analytics.timeseries import analyze_series, multi_series_stats
from aiviz.utils.helpers import numeric_columns, infer_time_column


class TimeSeriesPanel(QWidget):
    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._df: pd.DataFrame | None = None
        self._result = None
        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        hdr = QLabel("Time-Series Analysis")
        hdr.setObjectName("heading")
        root.addWidget(hdr)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: controls ──────────────────────────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(220)
        cl = QVBoxLayout(ctrl)
        cl.setContentsMargins(0, 0, 8, 0)

        grp_sig = QGroupBox("Signal Column")
        g1 = QVBoxLayout(grp_sig)
        self._sig_combo = QComboBox()
        g1.addWidget(self._sig_combo)
        cl.addWidget(grp_sig)

        grp_idx = QGroupBox("Time / Index Column")
        g2 = QVBoxLayout(grp_idx)
        self._idx_combo = QComboBox()
        g2.addWidget(self._idx_combo)
        cl.addWidget(grp_idx)

        grp_win = QGroupBox("Rolling Window")
        gw = QVBoxLayout(grp_win)
        self._window_slider = QSlider(Qt.Orientation.Horizontal)
        self._window_slider.setRange(3, 200)
        self._window_slider.setValue(10)
        self._window_lbl = QLabel("10")
        self._window_slider.valueChanged.connect(
            lambda v: self._window_lbl.setText(str(v))
        )
        gw.addWidget(self._window_slider)
        gw.addWidget(self._window_lbl)
        cl.addWidget(grp_win)

        grp_sig2 = QGroupBox("Anomaly Threshold (σ)")
        gs = QVBoxLayout(grp_sig2)
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(1.0, 6.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setValue(3.0)
        gs.addWidget(self._sigma_spin)
        cl.addWidget(grp_sig2)

        self._smooth_chk = QCheckBox("Apply smoothing")
        self._smooth_chk.setChecked(True)
        cl.addWidget(self._smooth_chk)

        self._run_btn = QPushButton("Analyze")
        self._run_btn.setEnabled(False)
        cl.addWidget(self._run_btn)

        self._ai_btn = QPushButton("🤖 AI Insight")
        self._ai_btn.setObjectName("secondary")
        self._ai_btn.setEnabled(False)
        cl.addWidget(self._ai_btn)

        cl.addStretch()
        splitter.addWidget(ctrl)

        # ── Right: analysis tabs ────────────────────────────────────────
        right = QTabWidget()

        # Main analysis plot
        self._main_plot = PlotWidget(figsize=(10, 6))
        right.addTab(self._main_plot, "Analysis")

        # Multi-signal comparison
        multi_widget = QWidget()
        ml = QVBoxLayout(multi_widget)
        ml.addWidget(QLabel("Select columns to compare (Ctrl+click):"))
        self._multi_list = QListWidget()
        self._multi_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._multi_list.setMaximumHeight(120)
        ml.addWidget(self._multi_list)
        self._compare_btn = QPushButton("Compare Signals")
        self._compare_btn.setEnabled(False)
        ml.addWidget(self._compare_btn)
        self._multi_plot = PlotWidget(figsize=(10, 4))
        ml.addWidget(self._multi_plot)
        self._stats_table = DataTableView()
        ml.addWidget(self._stats_table)
        right.addTab(multi_widget, "Multi-Signal")

        # Anomaly detail table
        self._anom_table = DataTableView()
        right.addTab(self._anom_table, "Anomalies")

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Metrics row
        metrics_row = QHBoxLayout()
        self._m_labels: dict[str, QLabel] = {}
        for k in ["Mean", "Std", "Anomalies", "Trend"]:
            g = QGroupBox(k)
            gl = QVBoxLayout(g)
            v = QLabel("–")
            v.setObjectName("heading")
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gl.addWidget(v)
            self._m_labels[k] = v
            metrics_row.addWidget(g)
        root.insertLayout(1, metrics_row)

        self._run_btn.clicked.connect(self._run_analysis)
        self._compare_btn.clicked.connect(self._run_multi)
        self._ai_btn.clicked.connect(self._run_ai)

    # ------------------------------------------------------------------

    def _on_data_loaded(self, result) -> None:
        if not result.ok or result.df is None:
            self._df = None
            self._run_btn.setEnabled(False)
            return
        self._df = result.df
        num = numeric_columns(self._df)
        time_col = infer_time_column(self._df)

        self._sig_combo.clear()
        self._sig_combo.addItems(num)

        self._idx_combo.clear()
        self._idx_combo.addItem("(default index)")
        self._idx_combo.addItems(self._df.columns.tolist())
        if time_col:
            idx = self._df.columns.tolist().index(time_col) + 1
            self._idx_combo.setCurrentIndex(idx)

        self._multi_list.clear()
        self._multi_list.addItems(num)

        self._run_btn.setEnabled(bool(num))
        self._compare_btn.setEnabled(bool(num))

    def _run_analysis(self) -> None:
        if self._df is None:
            return
        col = self._sig_combo.currentText()
        idx_col = self._idx_combo.currentText()
        window = self._window_slider.value()
        sigma = self._sigma_spin.value()
        smooth = self._smooth_chk.isChecked()

        if idx_col != "(default index)" and idx_col in self._df.columns:
            series = self._df.set_index(idx_col)[col]
        else:
            series = self._df[col]

        self._result = analyze_series(series, window=window, smooth=smooth, anomaly_sigma=sigma)
        r = self._result

        self._m_labels["Mean"].setText(f"{r.stats['mean']:.4g}")
        self._m_labels["Std"].setText(f"{r.stats['std']:.4g}")
        self._m_labels["Anomalies"].setText(str(r.stats["anomaly_count"]))
        direction = "↑" if r.trend_slope > 0 else "↓" if r.trend_slope < 0 else "→"
        self._m_labels["Trend"].setText(f"{direction} {abs(r.trend_slope):.3g}/pt")

        ax = self._main_plot.get_ax()
        mpl_charts.plot_timeseries_analysis(ax, r, col_name=col)
        self._main_plot.redraw()

        # Anomaly table
        import pandas as pd
        anom_idx = r.anomalies[r.anomalies].index
        anom_df = pd.DataFrame({
            "index": anom_idx,
            col: r.original[anom_idx].values,
            "rolling_mean": r.rolling_mean[anom_idx].values,
        })
        self._anom_table.load(anom_df)
        self._ai_btn.setEnabled(True)

    def _run_multi(self) -> None:
        if self._df is None:
            return
        cols = [item.text() for item in self._multi_list.selectedItems()]
        if len(cols) < 2:
            return
        stats_df = multi_series_stats(self._df, cols)

        self._stats_table.load(stats_df.reset_index())

        import numpy as np
        ax = self._multi_plot.get_ax()
        for col in cols:
            s = self._df[col].dropna()
            ax.plot(range(len(s)), s.values, label=col)
        ax.set_title("Multi-Signal Overlay")
        ax.legend(fontsize=8)
        self._multi_plot.redraw()

    def _run_ai(self) -> None:
        if self._result is None:
            return
        from aiviz.ai.agent import AnalysisAgent
        col = self._sig_combo.currentText()
        agent = AnalysisAgent()
        self._ai_btn.setEnabled(False)
        self._ai_btn.setText("Asking AI…")

        def call():
            return agent.explain_timeseries(
                col, self._result.stats, int(self._result.anomalies.sum())
            )

        worker = WorkerThread(call)
        worker.result_ready.connect(self._show_ai)
        worker.error_occurred.connect(lambda e: self._show_ai(None))
        worker.start()
        self._worker = worker  # prevent GC

    def _show_ai(self, result) -> None:
        self._ai_btn.setEnabled(True)
        self._ai_btn.setText("🤖 AI Insight")
        if result is None:
            return
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("AI Time-Series Insight")
        msg.setText(result.answer)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()
