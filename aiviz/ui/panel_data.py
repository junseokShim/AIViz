"""
Data Overview panel.

Shows: table preview, schema, descriptive stats, missing value map,
correlation heatmap, and categorical distributions.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QComboBox, QPushButton, QGroupBox, QSplitter, QScrollArea,
)
from PyQt6.QtCore import Qt

from aiviz.app.controller import AppController
from aiviz.ui.widgets.data_table import DataTableView
from aiviz.ui.widgets.plot_widget import PlotWidget
from aiviz.visualization import mpl_charts


class DataPanel(QWidget):
    """Data Overview tab – preview, schema, stats, correlation."""

    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._summary = None
        self._df = None
        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Heading
        lbl = QLabel("Data Overview")
        lbl.setObjectName("heading")
        layout.addWidget(lbl)

        # Metrics row
        self._metric_labels: dict[str, QLabel] = {}
        metrics_row = QHBoxLayout()
        for key in ["Rows", "Columns", "Missing", "Duplicates", "Memory"]:
            grp = QGroupBox(key)
            grp_lay = QVBoxLayout(grp)
            val = QLabel("–")
            val.setObjectName("heading")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grp_lay.addWidget(val)
            self._metric_labels[key] = val
            metrics_row.addWidget(grp)
        layout.addLayout(metrics_row)

        # Sub-tabs
        self._sub_tabs = QTabWidget()
        layout.addWidget(self._sub_tabs)

        # Tab 1: Table
        self._table_view = DataTableView()
        self._sub_tabs.addTab(self._table_view, "Table Preview")

        # Tab 2: Schema
        self._schema_table = DataTableView()
        self._sub_tabs.addTab(self._schema_table, "Schema")

        # Tab 3: Statistics
        self._stats_table = DataTableView()
        self._sub_tabs.addTab(self._stats_table, "Statistics")

        # Tab 4: Missing values
        self._missing_plot = PlotWidget(figsize=(9, 4))
        self._sub_tabs.addTab(self._missing_plot, "Missing Values")

        # Tab 5: Correlation
        self._corr_plot = PlotWidget(figsize=(9, 6))
        self._sub_tabs.addTab(self._corr_plot, "Correlation")

        # Tab 6: Categorical distributions
        cat_widget = QWidget()
        cat_layout = QVBoxLayout(cat_widget)
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Column:"))
        self._cat_combo = QComboBox()
        self._cat_combo.setMinimumWidth(180)
        ctrl_row.addWidget(self._cat_combo)
        ctrl_row.addStretch()
        cat_layout.addLayout(ctrl_row)
        self._cat_plot = PlotWidget(figsize=(9, 4))
        cat_layout.addWidget(self._cat_plot)
        self._sub_tabs.addTab(cat_widget, "Categoricals")
        self._cat_combo.currentTextChanged.connect(self._refresh_cat_plot)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_data_loaded(self, result) -> None:
        if not result.ok or result.df is None:
            self._clear_all()
            return

        self._df = result.df

        from aiviz.analytics.summary import compute_summary
        self._summary = compute_summary(self._df)
        s = self._summary

        # Metrics
        mem = f"{s.schema.memory_kb:.1f} KB" if s.schema.memory_kb < 1024 else f"{s.schema.memory_kb/1024:.1f} MB"
        self._metric_labels["Rows"].setText(f"{s.schema.n_rows:,}")
        self._metric_labels["Columns"].setText(str(s.schema.n_cols))
        self._metric_labels["Missing"].setText(str(sum(c.null_count for c in s.schema.columns)))
        self._metric_labels["Duplicates"].setText(str(s.schema.duplicate_rows))
        self._metric_labels["Memory"].setText(mem)

        # Table
        self._table_view.load(self._df)

        # Schema
        import pandas as pd
        schema_rows = []
        for c in s.schema.columns:
            schema_rows.append({
                "Column": c.name, "Dtype": c.dtype, "Role": c.inferred_role,
                "Non-null": c.non_null, "Null %": f"{c.null_pct:.1f}",
                "Unique": c.unique, "Sample": str(c.sample_values[:2]),
            })
        self._schema_table.load(pd.DataFrame(schema_rows))

        # Stats
        if not s.numeric_stats.empty:
            self._stats_table.load(s.numeric_stats.reset_index().rename(columns={"index": "stat"}))

        # Missing heatmap
        ax = self._missing_plot.get_ax()
        mpl_charts.plot_missing_heatmap(ax, s.missing_heatmap_data)
        self._missing_plot.redraw()

        # Correlation
        if s.correlation is not None:
            ax = self._corr_plot.get_ax()
            mpl_charts.plot_heatmap_correlation(ax, s.correlation)
            self._corr_plot.redraw()

        # Categoricals combo
        self._cat_combo.blockSignals(True)
        self._cat_combo.clear()
        cat_cols = list(s.top_categoricals.keys())
        self._cat_combo.addItems(cat_cols)
        self._cat_combo.blockSignals(False)
        if cat_cols:
            self._refresh_cat_plot(cat_cols[0])

    def _refresh_cat_plot(self, col: str) -> None:
        if self._summary is None or col not in self._summary.top_categoricals:
            return
        vc = self._summary.top_categoricals[col]
        import pandas as pd
        tmp = pd.DataFrame({"category": vc.index.astype(str), col: vc.values})
        ax = self._cat_plot.get_ax()
        mpl_charts.plot_bar(ax, tmp, "category", col, agg="sum")
        self._cat_plot.redraw()

    def _clear_all(self) -> None:
        for lbl in self._metric_labels.values():
            lbl.setText("–")
        self._table_view.clear()
        self._schema_table.clear()
        self._stats_table.clear()
        self._missing_plot.clear()
        self._corr_plot.clear()
        self._cat_combo.clear()
        self._cat_plot.clear()
