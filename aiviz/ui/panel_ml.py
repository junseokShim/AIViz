"""
ML Panel – Clustering and Simple Deep Learning (MLP) for tabular data.

Tabs:
  1. Clustering (KMeans / DBSCAN)
  2. Neural Network (MLP Regressor / Classifier)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QListWidget, QAbstractItemView, QTabWidget, QSplitter,
    QTextEdit, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt

from aiviz.app.controller import AppController, WorkerThread
from aiviz.ui.widgets.plot_widget import PlotWidget
from aiviz.ui.widgets.data_table import DataTableView
from aiviz.ui.widgets.insight_panel import InsightPanel
from aiviz.analytics.clustering_service import run_kmeans, run_dbscan
from aiviz.analytics.dl_service import run_mlp
from aiviz.utils.helpers import numeric_columns
from aiviz.visualization import mpl_charts
from aiviz.app.style import C_GREEN, C_RED, C_BLUE


class MLPanel(QWidget):
    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._df: Optional[pd.DataFrame] = None
        self._cluster_result = None
        self._dl_result = None
        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        hdr = QLabel("ML 분석 – 클러스터링 & 딥러닝")
        hdr.setObjectName("heading")
        root.addWidget(hdr)

        tabs = QTabWidget()
        root.addWidget(tabs)

        tabs.addTab(self._build_clustering_tab(), "🔵 클러스터링")
        tabs.addTab(self._build_dl_tab(), "🧠 신경망 (MLP)")

    # ------------------------------------------------------------------
    # Clustering tab
    # ------------------------------------------------------------------

    def _build_clustering_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        # Controls
        ctrl = QWidget()
        ctrl.setFixedWidth(240)
        cl = QVBoxLayout(ctrl)

        grp_method = QGroupBox("클러스터링 방법")
        gm = QVBoxLayout(grp_method)
        self._cluster_method = QComboBox()
        self._cluster_method.addItems(["KMeans", "DBSCAN"])
        self._cluster_method.currentTextChanged.connect(self._on_cluster_method_changed)
        gm.addWidget(self._cluster_method)
        cl.addWidget(grp_method)

        # KMeans options
        self._kmeans_grp = QGroupBox("KMeans 설정")
        kg = QVBoxLayout(self._kmeans_grp)
        kg.addWidget(QLabel("클러스터 수 (k):"))
        self._k_spin = QSpinBox()
        self._k_spin.setRange(2, 20)
        self._k_spin.setValue(3)
        kg.addWidget(self._k_spin)
        cl.addWidget(self._kmeans_grp)

        # DBSCAN options
        self._dbscan_grp = QGroupBox("DBSCAN 설정")
        dg = QVBoxLayout(self._dbscan_grp)
        dg.addWidget(QLabel("eps:"))
        self._eps_spin = QDoubleSpinBox()
        self._eps_spin.setRange(0.01, 100.0)
        self._eps_spin.setValue(0.5)
        self._eps_spin.setDecimals(2)
        dg.addWidget(self._eps_spin)
        dg.addWidget(QLabel("min_samples:"))
        self._min_samples_spin = QSpinBox()
        self._min_samples_spin.setRange(2, 100)
        self._min_samples_spin.setValue(5)
        dg.addWidget(self._min_samples_spin)
        self._dbscan_grp.hide()
        cl.addWidget(self._dbscan_grp)

        self._scale_chk = QCheckBox("데이터 표준화 (StandardScaler)")
        self._scale_chk.setChecked(True)
        cl.addWidget(self._scale_chk)

        grp_cols = QGroupBox("특성 컬럼 선택 (Ctrl+클릭)")
        gc = QVBoxLayout(grp_cols)
        self._feat_list = QListWidget()
        self._feat_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        gc.addWidget(self._feat_list)
        cl.addWidget(grp_cols)

        self._cluster_run_btn = QPushButton("클러스터링 실행")
        self._cluster_run_btn.setEnabled(False)
        cl.addWidget(self._cluster_run_btn)
        self._cluster_run_btn.clicked.connect(self._run_clustering)

        cl.addStretch()
        layout.addWidget(ctrl)

        # Right: results
        right_tabs = QTabWidget()

        self._cluster_plot = PlotWidget(figsize=(8, 5))
        right_tabs.addTab(self._cluster_plot, "시각화")

        self._cluster_table = DataTableView()
        right_tabs.addTab(self._cluster_table, "클러스터 요약")

        self._cluster_insight = InsightPanel("클러스터링 결과")
        right_tabs.addTab(self._cluster_insight, "결과 요약")

        layout.addWidget(right_tabs)
        return w

    def _on_cluster_method_changed(self, method: str) -> None:
        self._kmeans_grp.setVisible(method == "KMeans")
        self._dbscan_grp.setVisible(method == "DBSCAN")

    def _run_clustering(self) -> None:
        if self._df is None:
            return
        feat_cols = [item.text() for item in self._feat_list.selectedItems()]
        if not feat_cols:
            self._ctrl.log_message.emit("[경고] 특성 컬럼을 하나 이상 선택하세요.")
            return

        method = self._cluster_method.currentText()
        scale = self._scale_chk.isChecked()
        self._cluster_run_btn.setEnabled(False)
        self._cluster_run_btn.setText("실행 중…")

        def call():
            if method == "KMeans":
                return run_kmeans(self._df, feat_cols, self._k_spin.value(), scale)
            else:
                return run_dbscan(self._df, feat_cols, self._eps_spin.value(),
                                  self._min_samples_spin.value(), scale)

        worker = WorkerThread(call)
        worker.result_ready.connect(self._on_cluster_done)
        worker.error_occurred.connect(self._on_cluster_error)
        worker.start()
        self._cluster_worker = worker

    def _on_cluster_done(self, result) -> None:
        self._cluster_run_btn.setEnabled(True)
        self._cluster_run_btn.setText("클러스터링 실행")
        self._cluster_result = result

        if not result.ok:
            self._cluster_insight.set_text(f"오류: {result.error}")
            return

        # Summary text
        lines = [
            f"방법: {result.method}",
            f"클러스터 수: {result.n_clusters}",
            "",
            "클러스터 크기:",
        ]
        for k, v in sorted(result.cluster_sizes.items()):
            label = "노이즈" if k == -1 else f"클러스터 {k}"
            lines.append(f"  {label}: {v}개 ({100*v/sum(result.cluster_sizes.values()):.1f}%)")
        if result.inertia is not None:
            lines.append(f"\n관성 (Inertia): {result.inertia:.4f}")
        if result.silhouette is not None:
            lines.append(f"실루엣 점수: {result.silhouette:.4f}")
        self._cluster_insight.set_text("\n".join(lines))

        # Summary table
        feat_cols = result.feature_cols
        sub = self._df[feat_cols].copy()
        sub["cluster_label"] = result.as_label_series(sub.index[:len(result.labels)])
        summary = sub.groupby("cluster_label")[feat_cols].agg(["mean", "std"]).round(4)
        self._cluster_table.load(summary.reset_index())

        # Scatter plot (first 2 features)
        ax = self._cluster_plot.get_ax()
        ax.clear()
        sub_plot = sub.dropna()
        labels_plot = sub_plot["cluster_label"].values
        if len(feat_cols) >= 2:
            from aiviz.app.style import SERIES_COLORS
            for lbl in sorted(set(labels_plot)):
                mask = labels_plot == lbl
                color = "gray" if lbl == -1 else SERIES_COLORS[lbl % len(SERIES_COLORS)]
                name = "노이즈" if lbl == -1 else f"클러스터 {lbl}"
                ax.scatter(sub_plot[feat_cols[0]].values[mask],
                           sub_plot[feat_cols[1]].values[mask],
                           label=name, color=color, alpha=0.6, s=20)
            ax.set_xlabel(feat_cols[0])
            ax.set_ylabel(feat_cols[1])
            ax.legend(fontsize=8)
        else:
            ax.bar(range(len(result.cluster_sizes)), list(result.cluster_sizes.values()))
            ax.set_xticks(range(len(result.cluster_sizes)))
            ax.set_xticklabels([f"클러스터 {k}" for k in result.cluster_sizes.keys()])

        ax.set_title(f"{result.method} 클러스터링 결과")
        self._cluster_plot.redraw()

        self._ctrl.log_message.emit(
            f"[클러스터링] {result.method} 완료 – {result.n_clusters}개 클러스터"
        )

    def _on_cluster_error(self, msg: str) -> None:
        self._cluster_run_btn.setEnabled(True)
        self._cluster_run_btn.setText("클러스터링 실행")
        self._cluster_insight.set_text(f"오류: {msg}")

    # ------------------------------------------------------------------
    # DL / MLP tab
    # ------------------------------------------------------------------

    def _build_dl_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setFixedWidth(240)
        cl = QVBoxLayout(ctrl)

        grp_task = QGroupBox("작업 유형")
        gt = QVBoxLayout(grp_task)
        self._task_combo = QComboBox()
        self._task_combo.addItems(["자동 감지", "회귀 (Regression)", "분류 (Classification)"])
        gt.addWidget(self._task_combo)
        cl.addWidget(grp_task)

        grp_target = QGroupBox("타겟 컬럼")
        gta = QVBoxLayout(grp_target)
        self._target_combo = QComboBox()
        gta.addWidget(self._target_combo)
        cl.addWidget(grp_target)

        grp_feats = QGroupBox("특성 컬럼 (Ctrl+클릭)")
        gf = QVBoxLayout(grp_feats)
        self._dl_feat_list = QListWidget()
        self._dl_feat_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        gf.addWidget(self._dl_feat_list)
        cl.addWidget(grp_feats)

        grp_arch = QGroupBox("MLP 구조")
        ga = QVBoxLayout(grp_arch)
        ga.addWidget(QLabel("은닉층 (쉼표 구분):"))
        self._hidden_edit = __import__("PyQt6.QtWidgets", fromlist=["QLineEdit"]).QLineEdit("64,32")
        ga.addWidget(self._hidden_edit)
        ga.addWidget(QLabel("최대 반복:"))
        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(50, 2000)
        self._max_iter_spin.setValue(300)
        ga.addWidget(self._max_iter_spin)
        cl.addWidget(grp_arch)

        self._dl_run_btn = QPushButton("학습 실행")
        self._dl_run_btn.setEnabled(False)
        cl.addWidget(self._dl_run_btn)
        self._dl_run_btn.clicked.connect(self._run_dl)

        cl.addStretch()
        layout.addWidget(ctrl)

        # Right
        right_tabs = QTabWidget()

        self._dl_metrics_panel = InsightPanel("학습 결과")
        right_tabs.addTab(self._dl_metrics_panel, "결과 / 지표")

        self._dl_pred_table = DataTableView()
        right_tabs.addTab(self._dl_pred_table, "예측 vs 실제")

        layout.addWidget(right_tabs)
        return w

    def _run_dl(self) -> None:
        if self._df is None:
            return

        feat_cols = [item.text() for item in self._dl_feat_list.selectedItems()]
        target_col = self._target_combo.currentText()

        if not feat_cols:
            self._ctrl.log_message.emit("[경고] 특성 컬럼을 선택하세요.")
            return
        if not target_col:
            self._ctrl.log_message.emit("[경고] 타겟 컬럼을 선택하세요.")
            return

        task_text = self._task_combo.currentText()
        if "회귀" in task_text:
            task = "regression"
        elif "분류" in task_text:
            task = "classification"
        else:
            task = "auto"

        try:
            hidden = tuple(int(x.strip()) for x in self._hidden_edit.text().split(",") if x.strip())
        except ValueError:
            hidden = (64, 32)

        max_iter = self._max_iter_spin.value()

        self._dl_run_btn.setEnabled(False)
        self._dl_run_btn.setText("학습 중…")

        def call():
            return run_mlp(self._df, target_col, feat_cols,
                           task=task, hidden_layer_sizes=hidden, max_iter=max_iter)

        worker = WorkerThread(call)
        worker.result_ready.connect(self._on_dl_done)
        worker.error_occurred.connect(self._on_dl_error)
        worker.start()
        self._dl_worker = worker

    def _on_dl_done(self, result) -> None:
        self._dl_run_btn.setEnabled(True)
        self._dl_run_btn.setText("학습 실행")
        self._dl_result = result

        if not result.ok:
            self._dl_metrics_panel.set_text(f"오류: {result.error}")
            return

        lines = [
            f"작업: {'회귀' if result.task == 'regression' else '분류'}",
            f"타겟: {result.target_col}",
            f"특성: {', '.join(result.feature_cols)}",
            f"학습 샘플: {result.n_train}  테스트 샘플: {result.n_test}",
            "",
            "── 성능 지표 ──",
            result.metrics_text(),
        ]
        self._dl_metrics_panel.set_text("\n".join(lines))

        # Prediction table
        if result.predictions is not None and result.test_targets is not None:
            pred_df = pd.DataFrame({
                "실제값": result.test_targets,
                "예측값": result.predictions,
            })
            if result.task == "regression":
                pred_df["오차"] = (pred_df["예측값"] - pred_df["실제값"]).round(6)
            self._dl_pred_table.load(pred_df.round(6).reset_index(drop=True))

        self._ctrl.log_message.emit(
            f"[MLP] 학습 완료 – {result.task} / 학습:{result.n_train} / 테스트:{result.n_test}"
        )

    def _on_dl_error(self, msg: str) -> None:
        self._dl_run_btn.setEnabled(True)
        self._dl_run_btn.setText("학습 실행")
        self._dl_metrics_panel.set_text(f"오류: {msg}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _on_data_loaded(self, result) -> None:
        if not result.ok or result.df is None:
            self._df = None
            self._cluster_run_btn.setEnabled(False)
            self._dl_run_btn.setEnabled(False)
            return

        self._df = result.df
        num_cols = numeric_columns(self._df)
        all_cols = self._df.columns.tolist()

        # Clustering feature list
        self._feat_list.clear()
        self._feat_list.addItems(num_cols)

        # DL feature list
        self._dl_feat_list.clear()
        self._dl_feat_list.addItems(num_cols)

        # Target combo (all columns)
        self._target_combo.clear()
        self._target_combo.addItems(all_cols)

        self._cluster_run_btn.setEnabled(bool(num_cols))
        self._dl_run_btn.setEnabled(bool(num_cols))
