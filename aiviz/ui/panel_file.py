"""
File loading sidebar panel.

Provides: file open button, drag-drop hint, file info display, clear button.
Emits open_requested(bytes, name) when the user picks a file.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QHBoxLayout, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from aiviz.app.controller import AppController
from aiviz.utils.helpers import human_bytes
from config import APP


class FilePanel(QWidget):
    """Left sidebar for loading files."""

    open_requested = pyqtSignal(bytes, str)   # (file_bytes, file_name)

    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._setup_ui()
        self._connect()
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("AIViz")
        title.setObjectName("heading")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Local AI Analytics")
        sub.setObjectName("meta")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)

        layout.addSpacing(8)

        # Open button
        self._open_btn = QPushButton("📂  Open File…")
        self._open_btn.setToolTip(
            "Open CSV, Excel, JSON, Parquet, or image file"
        )
        layout.addWidget(self._open_btn)

        # Drag-drop hint
        hint = QLabel("or drag & drop a file here")
        hint.setObjectName("meta")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        layout.addSpacing(8)

        # File info group
        info_group = QGroupBox("Loaded File")
        info_layout = QVBoxLayout(info_group)

        self._lbl_name = QLabel("–")
        self._lbl_name.setWordWrap(True)
        self._lbl_type = QLabel("–")
        self._lbl_type.setObjectName("meta")
        self._lbl_shape = QLabel("–")
        self._lbl_shape.setObjectName("meta")
        self._lbl_size = QLabel("–")
        self._lbl_size.setObjectName("meta")

        info_layout.addWidget(self._lbl_name)
        info_layout.addWidget(self._lbl_type)
        info_layout.addWidget(self._lbl_shape)
        info_layout.addWidget(self._lbl_size)

        layout.addWidget(info_group)

        # Clear button
        self._clear_btn = QPushButton("✕  Clear")
        self._clear_btn.setObjectName("secondary")
        self._clear_btn.setEnabled(False)
        layout.addWidget(self._clear_btn)

        layout.addStretch()

        # Supported formats note
        fmt_label = QLabel(
            "Supported:\nCSV · Excel · JSON · Parquet\nPNG · JPG · BMP · TIF"
        )
        fmt_label.setObjectName("meta")
        fmt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(fmt_label)

    def _connect(self) -> None:
        self._open_btn.clicked.connect(self._on_open_clicked)
        self._clear_btn.clicked.connect(self._ctrl.clear)
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_open_clicked(self) -> None:
        ext_list = " ".join(
            f"*{e}" for e in APP.supported_tabular + APP.supported_image
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            f"Supported files ({ext_list});;All files (*)"
        )
        if path:
            self._load_path(path)

    def _on_data_loaded(self, result) -> None:
        if not result.ok:
            self._lbl_name.setText("–")
            self._lbl_type.setText("–")
            self._lbl_shape.setText("–")
            self._lbl_size.setText("–")
            self._clear_btn.setEnabled(False)
            return

        self._lbl_name.setText(result.file_name)
        self._lbl_type.setText(result.file_type.capitalize())

        if result.df is not None:
            r, c = result.df.shape
            self._lbl_shape.setText(f"{r:,} rows × {c} cols")
        elif result.image is not None:
            w, h = result.image.size
            self._lbl_shape.setText(f"{w}×{h} px  ({result.image.mode})")
        else:
            self._lbl_shape.setText("–")

        self._lbl_size.setText(human_bytes(len(result.raw_bytes)))
        self._clear_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            data = Path(path).read_bytes()
            self.open_requested.emit(data, Path(path).name)
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "File Error", str(exc))
