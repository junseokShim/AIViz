"""
Forecast panel.

Holt-Winters / ARIMA / Simple ES forecasting with statsmodels.
Plots historical + forecast + confidence interval.
"""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QSpinBox, QSplitter, QTabWidget,
    QTextEdit,
)
from PyQt6.QtCore import Qt

from aiviz.app.controller import AppController, WorkerThread
from aiviz.ui.widgets.plot_widget import PlotWidget
from aiviz.ui.widgets.data_table import DataTableView
from aiviz.visualization import mpl_charts
from aiviz.analytics.forecast import run_forecast
from aiviz.utils.helpers import numeric_columns
from config import FORECAST


class ForecastPanel(QWidget):
    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._df: pd.DataFrame | None = None
        self._fc_result = None
        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        hdr = QLabel("Forecast Module")
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

        grp_method = QGroupBox("Forecast Method")
        gm = QVBoxLayout(grp_method)
        self._method_combo = QComboBox()
        self._method_combo.addItems(list(FORECAST.available_methods))
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        gm.addWidget(self._method_combo)
        cl.addWidget(grp_method)

        grp_hz = QGroupBox("Forecast Horizon (steps)")
        gh = QVBoxLayout(grp_hz)
        self._horizon_spin = QSpinBox()
        self._horizon_spin.setRange(1, 1000)
        self._horizon_spin.setValue(FORECAST.default_horizon)
        gh.addWidget(self._horizon_spin)
        cl.addWidget(grp_hz)

        # ARIMA-specific
        self._arima_grp = QGroupBox("ARIMA Order (p, d, q)")
        ga = QVBoxLayout(self._arima_grp)
        row = QHBoxLayout()
        self._arima_p = QSpinBox(); self._arima_p.setRange(0, 10); self._arima_p.setValue(1)
        self._arima_d = QSpinBox(); self._arima_d.setRange(0, 3);  self._arima_d.setValue(1)
        self._arima_q = QSpinBox(); self._arima_q.setRange(0, 10); self._arima_q.setValue(1)
        for sp, lbl in [(self._arima_p, "p"), (self._arima_d, "d"), (self._arima_q, "q")]:
            sub = QVBoxLayout()
            sub.addWidget(QLabel(lbl))
            sub.addWidget(sp)
            row.addLayout(sub)
        ga.addLayout(row)
        cl.addWidget(self._arima_grp)
        self._arima_grp.setVisible(False)

        self._run_btn = QPushButton("Run Forecast")
        self._run_btn.setEnabled(False)
        cl.addWidget(self._run_btn)

        self._ai_btn = QPushButton("🤖 AI Insight")
        self._ai_btn.setObjectName("secondary")
        self._ai_btn.setEnabled(False)
        cl.addWidget(self._ai_btn)

        cl.addStretch()

        splitter.addWidget(ctrl)

        # ── Right: tabs ─────────────────────────────────────────────────
        tabs = QTabWidget()

        self._fc_plot = PlotWidget(figsize=(10, 6))
        tabs.addTab(self._fc_plot, "Forecast Chart")

        self._metrics_table = DataTableView()
        tabs.addTab(self._metrics_table, "Metrics")

        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        tabs.addTab(self._summary_text, "Model Summary")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Metrics row
        metrics_row = QHBoxLayout()
        self._m_labels: dict[str, QLabel] = {}
        for k in ["Method", "RMSE", "MAE", "AIC"]:
            g = QGroupBox(k)
            gl = QVBoxLayout(g)
            v = QLabel("–")
            v.setObjectName("heading")
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gl.addWidget(v)
            self._m_labels[k] = v
            metrics_row.addWidget(g)
        root.insertLayout(1, metrics_row)

        self._run_btn.clicked.connect(self._run_forecast)
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

    def _on_method_changed(self, method: str) -> None:
        self._arima_grp.setVisible(method == "ARIMA")

    def _run_forecast(self) -> None:
        if self._df is None:
            return
        col = self._col_combo.currentText()
        method = self._method_combo.currentText()
        horizon = self._horizon_spin.value()
        series = self._df[col].dropna()

        kwargs = {}
        if method == "ARIMA":
            kwargs["order"] = (
                self._arima_p.value(),
                self._arima_d.value(),
                self._arima_q.value(),
            )

        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")

        def call():
            return run_forecast(series, method=method, horizon=horizon, **kwargs)

        worker = WorkerThread(call)
        worker.result_ready.connect(self._on_forecast_done)
        worker.error_occurred.connect(self._on_forecast_error)
        worker.start()
        self._worker = worker

    def _on_forecast_done(self, result) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run Forecast")

        if not result.ok:
            self._ctrl.log_message.emit(f"[ERROR] Forecast: {result.error}")
            return

        self._fc_result = result

        # Plot
        ax = self._fc_plot.get_ax()
        mpl_charts.plot_forecast(
            ax, result.historical, result.forecast,
            conf_int=result.conf_int,
            col_name=self._col_combo.currentText(),
        )
        self._fc_plot.redraw()

        # Metrics
        m = result.metrics
        self._m_labels["Method"].setText(result.method)
        self._m_labels["RMSE"].setText(f"{m.get('rmse', 0):.4g}")
        self._m_labels["MAE"].setText(f"{m.get('mae', 0):.4g}")
        self._m_labels["AIC"].setText(f"{m.get('aic', 'N/A')}")

        metrics_df = pd.DataFrame([
            {"Metric": k, "Value": f"{v:.4g}" if isinstance(v, float) else str(v)}
            for k, v in m.items()
        ])
        self._metrics_table.load(metrics_df)
        self._summary_text.setPlainText(result.model_summary)
        self._ai_btn.setEnabled(True)

    def _on_forecast_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run Forecast")
        self._ctrl.log_message.emit(f"[ERROR] Forecast worker: {msg}")

    def _run_ai(self) -> None:
        if self._fc_result is None:
            return
        from aiviz.ai.agent import AnalysisAgent
        agent = AnalysisAgent()
        self._ai_btn.setEnabled(False)
        self._ai_btn.setText("Asking AI…")
        col = self._col_combo.currentText()
        result_ref = self._fc_result

        def call():
            return agent.explain_forecast(
                col, result_ref.method, result_ref.metrics,
                self._horizon_spin.value()
            )

        worker = WorkerThread(call)
        worker.result_ready.connect(self._show_ai)
        worker.error_occurred.connect(lambda _: self._reset_ai())
        worker.start()
        self._worker_ai = worker

    def _show_ai(self, result) -> None:
        self._reset_ai()
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("AI Forecast Insight")
        msg.setText(result.answer)
        msg.exec()

    def _reset_ai(self) -> None:
        self._ai_btn.setEnabled(True)
        self._ai_btn.setText("🤖 AI Insight")
