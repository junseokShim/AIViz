"""
AI Assistant panel.

Conversational interface grounded in the loaded dataset.
Supports streaming tokens displayed in a chat-like text area.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QGroupBox, QComboBox, QSplitter,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor, QColor

from aiviz.app.controller import AppController, WorkerThread
from aiviz.ai.agent import AnalysisAgent
from aiviz.ai.ollama_client import get_client
from config import OLLAMA


class StreamWorker(QThread):
    """Stream Ollama tokens and emit them one by one."""

    token_received = pyqtSignal(str)
    finished_streaming = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, prompt: str, model: str):
        super().__init__()
        self._prompt = prompt
        self._model = model

    def run(self) -> None:
        try:
            client = get_client()
            if not client.is_healthy():
                self.error_occurred.emit(
                    "Ollama not available. Start with: `ollama serve`"
                )
                return
            for token in client.stream(self._prompt, model=self._model):
                self.token_received.emit(token)
            self.finished_streaming.emit()
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class AssistantPanel(QWidget):
    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._df: Optional[pd.DataFrame] = None
        self._stream_worker: Optional[StreamWorker] = None
        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        hdr = QLabel("AI Assistant")
        hdr.setObjectName("heading")
        root.addWidget(hdr)

        # Status row
        status_row = QHBoxLayout()
        self._status_lbl = QLabel("Checking Ollama…")
        self._status_lbl.setObjectName("meta")
        status_row.addWidget(self._status_lbl)

        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(160)
        self._model_combo.setToolTip("Select Ollama model")
        status_row.addWidget(QLabel("Model:"))
        status_row.addWidget(self._model_combo)

        self._refresh_btn = QPushButton("↻")
        self._refresh_btn.setFixedWidth(32)
        self._refresh_btn.setObjectName("secondary")
        self._refresh_btn.clicked.connect(self._refresh_models)
        status_row.addWidget(self._refresh_btn)
        status_row.addStretch()
        root.addLayout(status_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: quick actions ─────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(200)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)

        ll.addWidget(QLabel("Quick Actions"))

        self._q_btns = []
        quick_actions = [
            ("📋 Summarize dataset",  "Summarize this dataset. Highlight key patterns, distributions, and data quality issues."),
            ("📈 Suggest charts",     "Suggest the 3 most useful chart types for this dataset with specific column recommendations."),
            ("🔍 Find data issues",   "Identify data quality issues: missing values, outliers, suspicious values, encoding problems."),
            ("🔗 Correlation insight","Which columns are most correlated? Explain what these correlations mean in practical terms."),
            ("📌 Key columns",        "Which columns are most important or interesting? Explain why."),
        ]
        for label, prompt in quick_actions:
            btn = QPushButton(label)
            btn.setObjectName("secondary")
            btn.clicked.connect(lambda _, p=prompt: self._ask_quick(p))
            self._q_btns.append(btn)
            ll.addWidget(btn)

        ll.addStretch()

        data_ctx_grp = QGroupBox("Data Context")
        dc_lay = QVBoxLayout(data_ctx_grp)
        self._ctx_preview = QTextEdit()
        self._ctx_preview.setReadOnly(True)
        self._ctx_preview.setMaximumHeight(120)
        self._ctx_preview.setPlaceholderText("Load a dataset to see context preview…")
        dc_lay.addWidget(self._ctx_preview)
        ll.addWidget(data_ctx_grp)

        splitter.addWidget(left)

        # ── Right: chat area ────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)

        # Chat history
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setObjectName("chatDisplay")
        rl.addWidget(self._chat_display, stretch=3)

        # Input row
        input_row = QHBoxLayout()
        self._input_box = QLineEdit()
        self._input_box.setPlaceholderText("Ask a question about the loaded data…")
        self._input_box.returnPressed.connect(self._send_message)
        input_row.addWidget(self._input_box)

        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self._send_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("secondary")
        self._clear_btn.clicked.connect(self._chat_display.clear)
        input_row.addWidget(self._clear_btn)
        rl.addLayout(input_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Initial model refresh
        self._refresh_models()

    # ------------------------------------------------------------------

    def _on_data_loaded(self, result) -> None:
        if result.ok and result.df is not None:
            self._df = result.df
            from aiviz.utils.helpers import df_to_context_string
            ctx = df_to_context_string(result.df, max_rows=5)
            self._ctx_preview.setPlainText(ctx[:600] + "…" if len(ctx) > 600 else ctx)
            for btn in self._q_btns:
                btn.setEnabled(True)
        else:
            self._df = None
            self._ctx_preview.clear()

    def _refresh_models(self) -> None:
        client = get_client()
        healthy = client.is_healthy()
        if healthy:
            models = client.list_models()
            current = self._model_combo.currentText()
            self._model_combo.clear()
            self._model_combo.addItems(models if models else [OLLAMA.default_model])
            idx = self._model_combo.findText(current)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
            self._status_lbl.setText("🟢 Ollama connected")
        else:
            self._model_combo.clear()
            self._model_combo.addItem(OLLAMA.default_model)
            self._status_lbl.setText("🔴 Ollama offline – run `ollama serve`")

    def _send_message(self) -> None:
        text = self._input_box.text().strip()
        if not text:
            return
        self._input_box.clear()
        self._ask_quick(text)

    def _ask_quick(self, prompt_text: str) -> None:
        if self._stream_worker and self._stream_worker.isRunning():
            self._stream_worker.quit()

        # Add user message to display
        self._append_chat("You", prompt_text, "#89b4fa")

        # Build context-enriched prompt
        if self._df is not None:
            from aiviz.utils.helpers import df_to_context_string, truncate_str
            from aiviz.ai.prompts import general_question_prompt
            ctx = truncate_str(df_to_context_string(self._df), 2500)
            full_prompt = general_question_prompt(ctx, prompt_text)
        else:
            full_prompt = (
                f"You are AIViz, a data analysis assistant.\n"
                f"No dataset is currently loaded.\n\n"
                f"User: {prompt_text}"
            )

        model = self._model_combo.currentText() or OLLAMA.default_model

        # Start streaming
        self._append_chat("AIViz", "", "#a6e3a1")  # start AI turn
        self._send_btn.setEnabled(False)
        self._stream_worker = StreamWorker(full_prompt, model)
        self._stream_worker.token_received.connect(self._append_token)
        self._stream_worker.finished_streaming.connect(self._on_stream_done)
        self._stream_worker.error_occurred.connect(self._on_stream_error)
        self._stream_worker.start()

    def _append_chat(self, speaker: str, text: str, color: str) -> None:
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Speaker header
        self._chat_display.setTextColor(QColor(color))
        self._chat_display.append(f"\n{speaker}:")
        self._chat_display.setTextColor(QColor("#cdd6f4"))
        if text:
            self._chat_display.append(text)
        self._chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def _append_token(self, token: str) -> None:
        """Append a streamed token to the end of the chat display."""
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        self._chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def _on_stream_done(self) -> None:
        self._send_btn.setEnabled(True)
        self._chat_display.append("")  # blank line after response

    def _on_stream_error(self, msg: str) -> None:
        self._send_btn.setEnabled(True)
        self._chat_display.setTextColor(QColor("#f38ba8"))
        self._chat_display.append(f"\n[Error: {msg}]")
        self._chat_display.setTextColor(QColor("#cdd6f4"))
