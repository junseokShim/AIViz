"""
Export / Report Generation panel.

Generates self-contained PDF and HTML reports containing:
- dataset summary
- schema
- statistics
- embedded chart images
- AI-generated summary

Charts are captured from other panels via the PlotWidget.get_png_bytes() method.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
from PIL import Image

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QFileDialog, QTextEdit, QProgressBar,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from aiviz.app.controller import AppController, WorkerThread
from aiviz.export.html_exporter import HTMLExporter
from aiviz.export.pdf_exporter import PDFExporter, is_available as pdf_available


class ExportPanel(QWidget):
    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._df: Optional[pd.DataFrame] = None
        self._img: Optional[Image.Image] = None
        self._file_name: str = ""
        # Chart PNG bytes injected from outside (populated by panel actions)
        self._chart_images: list[tuple[str, bytes]] = []
        self._ai_summary: str = ""
        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        hdr = QLabel("Export / Report Generation")
        hdr.setObjectName("heading")
        root.addWidget(hdr)

        sub = QLabel(
            "Generate a structured report including dataset summary, statistics, "
            "and optional AI-generated insights."
        )
        sub.setObjectName("meta")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # ── Sections to include ─────────────────────────────────────────
        grp_sections = QGroupBox("Sections to Include")
        sec_lay = QVBoxLayout(grp_sections)

        self._chk_overview = QCheckBox("Dataset Overview (rows, cols, missing, duplicates)")
        self._chk_overview.setChecked(True)
        self._chk_schema = QCheckBox("Schema (column types, roles, null %)")
        self._chk_schema.setChecked(True)
        self._chk_stats = QCheckBox("Descriptive Statistics")
        self._chk_stats.setChecked(True)
        self._chk_ai = QCheckBox("AI-Generated Summary (requires Ollama)")
        self._chk_ai.setChecked(True)

        for chk in [self._chk_overview, self._chk_schema, self._chk_stats, self._chk_ai]:
            sec_lay.addWidget(chk)
        root.addWidget(grp_sections)

        # ── AI Summary ──────────────────────────────────────────────────
        grp_ai = QGroupBox("AI Summary Text (editable)")
        ai_lay = QVBoxLayout(grp_ai)
        self._ai_text = QTextEdit()
        self._ai_text.setPlaceholderText(
            "Click 'Generate AI Summary' to automatically create summary text,\n"
            "or type your own notes here."
        )
        self._ai_text.setMaximumHeight(120)
        ai_lay.addWidget(self._ai_text)

        gen_ai_btn = QPushButton("Generate AI Summary")
        gen_ai_btn.setObjectName("secondary")
        gen_ai_btn.clicked.connect(self._generate_ai_summary)
        ai_lay.addWidget(gen_ai_btn)
        root.addWidget(grp_ai)

        # ── Progress ────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.hide()
        root.addWidget(self._progress)

        # ── Export buttons ──────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._html_btn = QPushButton("📄  Export as HTML")
        self._html_btn.setEnabled(False)
        self._html_btn.clicked.connect(self._export_html)
        btn_row.addWidget(self._html_btn)

        self._pdf_btn = QPushButton("📑  Export as PDF")
        self._pdf_btn.setEnabled(pdf_available())
        if not pdf_available():
            self._pdf_btn.setToolTip("Install fpdf2: pip install fpdf2")
        self._pdf_btn.clicked.connect(self._export_pdf)
        btn_row.addWidget(self._pdf_btn)

        root.addLayout(btn_row)

        if not pdf_available():
            hint = QLabel("⚠ PDF export requires fpdf2:  pip install fpdf2")
            hint.setObjectName("meta")
            root.addWidget(hint)

        # ── Log ─────────────────────────────────────────────────────────
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        root.addWidget(self._log)

        root.addStretch()

    # ------------------------------------------------------------------

    def _on_data_loaded(self, result) -> None:
        if result.ok:
            self._df = result.df
            self._img = result.image
            self._file_name = result.file_name
            self._html_btn.setEnabled(True)
            if pdf_available():
                self._pdf_btn.setEnabled(True)
            self._log.append(f"Ready to export: {result.file_name}")
        else:
            self._df = None
            self._html_btn.setEnabled(False)

    def register_chart(self, title: str, png_bytes: bytes) -> None:
        """Called by other panels to register a chart for inclusion in the report."""
        self._chart_images.append((title, png_bytes))
        self._log.append(f"Chart registered: {title}")

    def _generate_ai_summary(self) -> None:
        if self._df is None:
            QMessageBox.information(self, "No Data", "Load a dataset first.")
            return
        from aiviz.ai.agent import AnalysisAgent
        agent = AnalysisAgent()
        self._progress.show()

        def call():
            return agent.summarize_dataset(self._df)

        worker = WorkerThread(call)
        worker.result_ready.connect(self._on_ai_summary)
        worker.error_occurred.connect(lambda e: self._log.append(f"[ERROR] {e}"))
        worker.start()
        self._ai_worker = worker

    def _on_ai_summary(self, result) -> None:
        self._progress.hide()
        self._ai_text.setPlainText(result.answer)

    def _export_html(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export HTML Report", f"aiviz_report_{self._file_name}.html",
            "HTML Report (*.html)"
        )
        if not path:
            return

        self._progress.show()

        def build():
            exp = HTMLExporter(file_name=self._file_name, df=self._df)
            if self._chk_overview.isChecked():
                exp.add_dataset_overview()
            if self._chk_schema.isChecked():
                exp.add_schema_table()
            if self._chk_stats.isChecked():
                exp.add_stats_table()
            for title, png in self._chart_images:
                exp.add_chart_image(png, title=title)
            ai_text = self._ai_text.toPlainText().strip()
            if self._chk_ai.isChecked() and ai_text:
                exp.add_text_section("AI-Generated Summary", ai_text)
            exp.save(path)
            return path

        worker = WorkerThread(build)
        worker.result_ready.connect(self._on_html_done)
        worker.error_occurred.connect(self._on_export_error)
        worker.start()
        self._worker = worker

    def _on_html_done(self, path: str) -> None:
        self._progress.hide()
        self._log.append(f"[OK] HTML report saved: {path}")
        QMessageBox.information(self, "Export Complete",
                                f"HTML report saved to:\n{path}")

    def _export_pdf(self) -> None:
        if not pdf_available():
            QMessageBox.warning(self, "fpdf2 not installed",
                                "Install fpdf2 first:\npip install fpdf2")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF Report", f"aiviz_report_{self._file_name}.pdf",
            "PDF Report (*.pdf)"
        )
        if not path:
            return

        self._progress.show()

        def build():
            exp = PDFExporter(file_name=self._file_name, df=self._df)
            if self._chk_overview.isChecked():
                exp.add_dataset_overview()
            if self._chk_schema.isChecked():
                exp.add_schema_table()
            if self._chk_stats.isChecked():
                exp.add_stats_table()
            for title, png in self._chart_images:
                exp.add_chart_image(png, title=title)
            ai_text = self._ai_text.toPlainText().strip()
            if self._chk_ai.isChecked() and ai_text:
                exp.add_text_section("AI-Generated Summary", ai_text)
            exp.save(path)
            return path

        worker = WorkerThread(build)
        worker.result_ready.connect(self._on_pdf_done)
        worker.error_occurred.connect(self._on_export_error)
        worker.start()
        self._worker = worker

    def _on_pdf_done(self, path: str) -> None:
        self._progress.hide()
        self._log.append(f"[OK] PDF report saved: {path}")
        QMessageBox.information(self, "Export Complete",
                                f"PDF report saved to:\n{path}")

    def _on_export_error(self, msg: str) -> None:
        self._progress.hide()
        self._log.append(f"[ERROR] {msg}")
        QMessageBox.critical(self, "Export Error", msg)
