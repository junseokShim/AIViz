"""
PandasTableModel + DataTableView – display a pandas DataFrame in a QTableView.

Supports:
- Sorting by clicking column headers
- Alternating row colours
- Lazy large-DataFrame handling (displays first N rows with a note)
"""

from __future__ import annotations

from typing import Optional, Any

import pandas as pd
from PyQt6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, QSortFilterProxyModel
)
from PyQt6.QtWidgets import QTableView, QWidget, QVBoxLayout, QLabel, QHBoxLayout

MAX_DISPLAY_ROWS = 10_000  # cap to keep UI responsive


class PandasTableModel(QAbstractTableModel):
    """Qt table model backed by a pandas DataFrame."""

    def __init__(self, df: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()
        self._display_df = self._df.head(MAX_DISPLAY_ROWS)

    def update(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self._display_df = df.head(MAX_DISPLAY_ROWS)
        self.endResetModel()

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._display_df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._display_df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            val = self._display_df.iat[index.row(), index.column()]
            if pd.isna(val):
                return "NaN"
            if isinstance(val, float):
                return f"{val:.6g}"
            return str(val)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            val = self._display_df.iat[index.row(), index.column()]
            if pd.api.types.is_numeric_dtype(type(val)):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        if role == Qt.ItemDataRole.ForegroundRole:
            from PyQt6.QtGui import QColor
            val = self._display_df.iat[index.row(), index.column()]
            if pd.isna(val):
                return QColor("#f38ba8")  # red for NaN
            return None

        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._display_df.columns[section])
            return str(section + 1)
        return None

    @property
    def total_rows(self) -> int:
        return len(self._df)


class DataTableView(QWidget):
    """
    Composite widget: QTableView with sortable columns + a row-count label.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model = PandasTableModel()
        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)

        self._view = QTableView()
        self._view.setModel(self._proxy)
        self._view.setSortingEnabled(True)
        self._view.setAlternatingRowColors(True)
        self._view.horizontalHeader().setStretchLastSection(True)
        self._view.verticalHeader().setDefaultSectionSize(24)
        self._view.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )

        self._row_label = QLabel()
        self._row_label.setObjectName("meta")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._view)
        layout.addWidget(self._row_label)

    def load(self, df: pd.DataFrame) -> None:
        self._model.update(df)
        total = self._model.total_rows
        shown = min(total, MAX_DISPLAY_ROWS)
        note = "" if total == shown else f"  (showing {shown:,} of {total:,})"
        self._row_label.setText(
            f"{total:,} rows × {len(df.columns)} columns{note}"
        )

    def clear(self) -> None:
        self._model.update(pd.DataFrame())
        self._row_label.setText("")
