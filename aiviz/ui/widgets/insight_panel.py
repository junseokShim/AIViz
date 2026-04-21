"""
InsightPanel – a scrollable, copyable AI insight text widget.

Replaces QMessageBox for AI output; allows selection, copy, and save.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QFileDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from aiviz.app.style import C_MAUVE, C_BASE, C_SUBTEXT


class InsightPanel(QWidget):
    """
    A self-contained panel for displaying AI-generated Korean insights.

    Usage:
        panel = InsightPanel(title="AI 인사이트")
        panel.set_text("분석 결과...")
        panel.append_text("추가 내용...")
    """

    def __init__(self, title: str = "🤖 AI 인사이트", parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui(title)

    def _setup_ui(self, title: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header row
        hdr = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {C_MAUVE}; font-weight: bold; font-size: 13px;")
        hdr.addWidget(lbl)
        hdr.addStretch()

        self._copy_btn = QPushButton("복사")
        self._copy_btn.setFixedWidth(56)
        self._copy_btn.setToolTip("전체 텍스트 복사")
        self._copy_btn.clicked.connect(self._copy_all)
        hdr.addWidget(self._copy_btn)

        self._save_btn = QPushButton("저장")
        self._save_btn.setFixedWidth(56)
        self._save_btn.setToolTip("인사이트를 텍스트 파일로 저장")
        self._save_btn.clicked.connect(self._save_insight)
        hdr.addWidget(self._save_btn)

        self._clear_btn = QPushButton("지우기")
        self._clear_btn.setFixedWidth(56)
        self._clear_btn.setToolTip("패널 내용 지우기")
        self._clear_btn.clicked.connect(self.clear)
        hdr.addWidget(self._clear_btn)

        layout.addLayout(hdr)

        # Text area
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Malgun Gothic, Apple SD Gothic Neo, Arial", 12))
        self._text_edit.setStyleSheet(
            f"background-color: {C_BASE}; "
            f"color: #cdd6f4; "
            f"border-radius: 6px; "
            f"padding: 8px;"
        )
        self._text_edit.setPlaceholderText("AI 인사이트가 여기에 표시됩니다…")
        layout.addWidget(self._text_edit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_text(self, text: str) -> None:
        """Replace all content."""
        self._text_edit.setPlainText(text)
        self._text_edit.moveCursor(self._text_edit.textCursor().MoveOperation.Start)

    def append_text(self, text: str) -> None:
        """Append text to the existing content."""
        self._text_edit.moveCursor(self._text_edit.textCursor().MoveOperation.End)
        self._text_edit.insertPlainText(text)

    def clear(self) -> None:
        """Clear all content."""
        self._text_edit.clear()

    def text(self) -> str:
        return self._text_edit.toPlainText()

    # ------------------------------------------------------------------
    # Button slots
    # ------------------------------------------------------------------

    def _copy_all(self) -> None:
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._text_edit.toPlainText())

    def _save_insight(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "인사이트 저장", "ai_insight.txt",
            "텍스트 파일 (*.txt);;모든 파일 (*)"
        )
        if path:
            try:
                Path(path).write_text(self._text_edit.toPlainText(), encoding="utf-8")
            except Exception as exc:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "저장 실패", str(exc))
