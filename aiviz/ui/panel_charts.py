"""
Interactive Chart Builder panel.

Lets users select chart type, axes, aggregation, then renders the result.
"""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QListWidget, QAbstractItemView,
    QSpinBox, QSplitter,
)
from PyQt6.QtCore import Qt

from aiviz.app.controller import AppController
from aiviz.ui.widgets.plot_widget import PlotWidget
from aiviz.visualization import mpl_charts
from aiviz.utils.helpers import numeric_columns


CHART_TYPES = ["Line Chart", "Scatter Plot", "Bar Chart", "Histogram", "Box Plot", "Correlation"]


class ChartsPanel(QWidget):
    """Chart Builder tab."""

    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._df: pd.DataFrame | None = None
        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        lbl = QLabel("Chart Builder")
        lbl.setObjectName("heading")
        root.addWidget(lbl)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # --- Left: controls ---
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(0, 0, 8, 0)
        ctrl_widget.setFixedWidth(220)

        # Chart type
        grp_type = QGroupBox("Chart Type")
        type_lay = QVBoxLayout(grp_type)
        self._chart_combo = QComboBox()
        self._chart_combo.addItems(CHART_TYPES)
        self._chart_combo.currentTextChanged.connect(self._on_chart_type_changed)
        type_lay.addWidget(self._chart_combo)
        ctrl_layout.addWidget(grp_type)

        # X axis
        grp_x = QGroupBox("X Axis / Category")
        x_lay = QVBoxLayout(grp_x)
        self._x_combo = QComboBox()
        x_lay.addWidget(self._x_combo)
        ctrl_layout.addWidget(grp_x)

        # Y axis (multi-select for line/box)
        grp_y = QGroupBox("Y Axis / Values")
        y_lay = QVBoxLayout(grp_y)
        self._y_list = QListWidget()
        self._y_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._y_list.setMaximumHeight(140)
        y_lay.addWidget(QLabel("Ctrl+click to multi-select"))
        y_lay.addWidget(self._y_list)
        ctrl_layout.addWidget(grp_y)

        # Colour column
        grp_color = QGroupBox("Color Column (scatter)")
        color_lay = QVBoxLayout(grp_color)
        self._color_combo = QComboBox()
        color_lay.addWidget(self._color_combo)
        ctrl_layout.addWidget(grp_color)

        # Aggregation
        grp_agg = QGroupBox("Aggregation (bar)")
        agg_lay = QVBoxLayout(grp_agg)
        self._agg_combo = QComboBox()
        self._agg_combo.addItems(["sum", "mean", "count", "max", "min"])
        agg_lay.addWidget(self._agg_combo)
        ctrl_layout.addWidget(grp_agg)

        # Bins (histogram)
        grp_bins = QGroupBox("Bins (histogram)")
        bins_lay = QVBoxLayout(grp_bins)
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(5, 500)
        self._bins_spin.setValue(40)
        bins_lay.addWidget(self._bins_spin)
        ctrl_layout.addWidget(grp_bins)

        # Generate button
        self._gen_btn = QPushButton("Generate Chart")
        self._gen_btn.setEnabled(False)
        ctrl_layout.addWidget(self._gen_btn)

        # Export button
        self._export_btn = QPushButton("Save PNG…")
        self._export_btn.setObjectName("secondary")
        self._export_btn.setEnabled(False)
        ctrl_layout.addWidget(self._export_btn)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl_widget)

        # --- Right: plot ---
        self._plot = PlotWidget(figsize=(10, 6))
        splitter.addWidget(self._plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._gen_btn.clicked.connect(self._generate_chart)
        self._export_btn.clicked.connect(self._export_chart)

    # ------------------------------------------------------------------

    def _on_data_loaded(self, result) -> None:
        if not result.ok or result.df is None:
            self._df = None
            self._gen_btn.setEnabled(False)
            return
        self._df = result.df
        self._populate_combos()
        self._gen_btn.setEnabled(True)

    def _populate_combos(self) -> None:
        if self._df is None:
            return
        all_cols = self._df.columns.tolist()
        num_cols = numeric_columns(self._df)

        for combo in (self._x_combo, self._color_combo):
            combo.clear()
            combo.addItem("(none)")
            combo.addItems(all_cols)

        self._y_list.clear()
        self._y_list.addItems(num_cols)
        # Pre-select first item
        if self._y_list.count():
            self._y_list.item(0).setSelected(True)

    def _on_chart_type_changed(self, chart_type: str) -> None:
        pass  # Could show/hide controls based on chart type

    def _generate_chart(self) -> None:
        if self._df is None:
            return

        chart_type = self._chart_combo.currentText()
        x_col = self._x_combo.currentText()
        if x_col == "(none)":
            x_col = self._df.columns[0]

        y_items = [item.text() for item in self._y_list.selectedItems()]
        color_col = self._color_combo.currentText()
        if color_col == "(none)":
            color_col = None

        num_cols = numeric_columns(self._df)
        ax = self._plot.get_ax()

        try:
            if chart_type == "Line Chart":
                if y_items:
                    mpl_charts.plot_line(ax, self._df, x_col, y_items)
            elif chart_type == "Scatter Plot":
                y = y_items[0] if y_items else (num_cols[0] if num_cols else x_col)
                mpl_charts.plot_scatter(ax, self._df, x_col, y, color_col=color_col)
            elif chart_type == "Bar Chart":
                y = y_items[0] if y_items else (num_cols[0] if num_cols else x_col)
                mpl_charts.plot_bar(ax, self._df, x_col, y, agg=self._agg_combo.currentText())
            elif chart_type == "Histogram":
                col = y_items[0] if y_items else (num_cols[0] if num_cols else x_col)
                mpl_charts.plot_histogram(ax, self._df[col], bins=self._bins_spin.value())
            elif chart_type == "Box Plot":
                cols = y_items if y_items else num_cols[:6]
                mpl_charts.plot_box(ax, self._df, cols)
            elif chart_type == "Correlation":
                corr = self._df[num_cols].corr() if len(num_cols) >= 2 else None
                if corr is not None:
                    mpl_charts.plot_heatmap_correlation(ax, corr)
                else:
                    ax.text(0.5, 0.5, "Need ≥2 numeric columns", ha="center", va="center")
        except Exception as exc:
            ax.text(0.5, 0.5, f"Chart error:\n{exc}", ha="center", va="center",
                    color="#f38ba8", fontsize=10)

        self._plot.redraw()
        self._export_btn.setEnabled(True)

    def _export_chart(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Chart", "aiviz_chart.png",
            "PNG Image (*.png);;PDF (*.pdf)"
        )
        if path:
            self._plot.save(path)
