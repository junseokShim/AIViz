"""
Image Analysis panel.

Preview, channel stats, pixel histogram, dominant colors,
text-based AI stats analysis, and LLaVA multimodal description.
"""

from __future__ import annotations

import io

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSplitter, QTabWidget, QTextEdit, QScrollArea,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

from aiviz.app.controller import AppController, WorkerThread
from aiviz.ui.widgets.plot_widget import PlotWidget
from aiviz.ui.widgets.data_table import DataTableView
from aiviz.visualization import mpl_charts


def _pil_to_pixmap(img) -> QPixmap:
    """Convert PIL Image to QPixmap."""
    rgb = img.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ImagePanel(QWidget):
    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._img = None
        self._img_bytes = None
        self._ana_result = None
        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        hdr = QLabel("Image Analysis")
        hdr.setObjectName("heading")
        root.addWidget(hdr)

        # Metrics row
        metrics_row = QHBoxLayout()
        self._m_labels: dict[str, QLabel] = {}
        for k in ["Size", "Mode", "Channels", "Pixels"]:
            g = QGroupBox(k)
            gl = QVBoxLayout(g)
            v = QLabel("–")
            v.setObjectName("heading")
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gl.addWidget(v)
            self._m_labels[k] = v
            metrics_row.addWidget(g)
        root.addLayout(metrics_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: preview ────────────────────────────────────────────────
        preview_widget = QWidget()
        pv_layout = QVBoxLayout(preview_widget)
        pv_layout.addWidget(QLabel("Image Preview"))

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setMinimumSize(300, 250)
        self._img_label.setStyleSheet("border: 1px solid #313244;")
        self._img_label.setText("No image loaded")

        scroll = QScrollArea()
        scroll.setWidget(self._img_label)
        scroll.setWidgetResizable(True)
        pv_layout.addWidget(scroll)

        # Grayscale toggle
        self._gray_btn = QPushButton("Toggle Grayscale")
        self._gray_btn.setObjectName("secondary")
        self._gray_btn.setEnabled(False)
        self._gray_chk = False
        self._gray_btn.clicked.connect(self._toggle_gray)
        pv_layout.addWidget(self._gray_btn)

        splitter.addWidget(preview_widget)

        # ── Right: analysis tabs ─────────────────────────────────────────
        tabs = QTabWidget()

        # Channel stats
        self._stats_table = DataTableView()
        tabs.addTab(self._stats_table, "Channel Stats")

        # Histograms
        self._hist_plot = PlotWidget(figsize=(7, 4))
        tabs.addTab(self._hist_plot, "Histogram")

        # Dominant colors
        self._color_plot = PlotWidget(figsize=(7, 3))
        tabs.addTab(self._color_plot, "Dominant Colors")

        # Text-based AI (stats)
        ai_stats_widget = QWidget()
        aw = QVBoxLayout(ai_stats_widget)
        self._ai_stats_btn = QPushButton("📊 Analyze Stats with AI")
        self._ai_stats_btn.setEnabled(False)
        aw.addWidget(self._ai_stats_btn)
        self._ai_stats_text = QTextEdit()
        self._ai_stats_text.setReadOnly(True)
        aw.addWidget(self._ai_stats_text)
        tabs.addTab(ai_stats_widget, "AI Stats Analysis")

        # Multimodal (LLaVA)
        mm_widget = QWidget()
        mw = QVBoxLayout(mm_widget)
        mw.addWidget(QLabel("Send image pixels to LLaVA-style vision model:"))
        self._mm_question = QTextEdit()
        self._mm_question.setPlaceholderText(
            "Optional: ask a specific question about the image\n"
            "(e.g. 'Describe defects', 'What patterns do you see?')"
        )
        self._mm_question.setMaximumHeight(80)
        mw.addWidget(self._mm_question)
        self._mm_btn = QPushButton("🔭 Describe Image with LLaVA")
        self._mm_btn.setEnabled(False)
        mw.addWidget(self._mm_btn)
        hint = QLabel(
            "Requires: ollama pull llava\n"
            "Set OLLAMA_VISION_MODEL=llava (or another vision model)"
        )
        hint.setObjectName("meta")
        mw.addWidget(hint)
        self._mm_text = QTextEdit()
        self._mm_text.setReadOnly(True)
        mw.addWidget(self._mm_text)
        tabs.addTab(mm_widget, "Multimodal AI")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self._ai_stats_btn.clicked.connect(self._run_ai_stats)
        self._mm_btn.clicked.connect(self._run_multimodal)

    # ------------------------------------------------------------------

    def _on_data_loaded(self, result) -> None:
        if not result.ok or result.image is None:
            # Clear image panel if tabular data loaded
            self._img = None
            self._img_bytes = None
            self._img_label.setText("No image loaded")
            self._gray_btn.setEnabled(False)
            self._ai_stats_btn.setEnabled(False)
            self._mm_btn.setEnabled(False)
            return

        self._img = result.image
        self._img_bytes = result.raw_bytes

        from aiviz.analytics.image_analysis import analyze_image
        self._ana_result = analyze_image(self._img, file_name=result.file_name)
        r = self._ana_result

        self._m_labels["Size"].setText(f"{r.width}×{r.height}")
        self._m_labels["Mode"].setText(r.mode)
        self._m_labels["Channels"].setText(str(r.n_channels))
        self._m_labels["Pixels"].setText(f"{r.n_pixels:,}")

        # Preview
        self._show_preview(self._img)

        # Channel stats
        self._stats_table.load(r.channel_stats)

        # Histogram
        ax = self._hist_plot.get_ax()
        mpl_charts.plot_pixel_histogram(ax, r.histograms)
        self._hist_plot.redraw()

        # Dominant colors
        if r.dominant_colors is not None:
            ax = self._color_plot.get_ax()
            mpl_charts.plot_dominant_colors(ax, r.dominant_colors)
            self._color_plot.redraw()

        self._gray_btn.setEnabled(True)
        self._ai_stats_btn.setEnabled(True)
        self._mm_btn.setEnabled(True)

    def _show_preview(self, img) -> None:
        pix = _pil_to_pixmap(img)
        scaled = pix.scaled(
            600, 500,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._img_label.setPixmap(scaled)

    def _toggle_gray(self) -> None:
        if self._img is None:
            return
        self._gray_chk = not self._gray_chk
        if self._gray_chk:
            self._show_preview(self._img.convert("L"))
        else:
            self._show_preview(self._img)

    def _run_ai_stats(self) -> None:
        if self._ana_result is None:
            return
        from aiviz.ai.agent import AnalysisAgent
        agent = AnalysisAgent()
        self._ai_stats_btn.setEnabled(False)
        r = self._ana_result

        image_info = {
            "width": r.width,
            "height": r.height,
            "mode": r.mode,
            "n_channels": r.n_channels,
            "has_transparency": r.has_transparency,
            "is_grayscale": r.is_grayscale,
            "aspect_ratio": r.aspect_ratio,
            "channel_stats": r.channel_stats,
        }

        def call():
            return agent.explain_image_stats(image_info)

        worker = WorkerThread(call)
        worker.result_ready.connect(self._show_stats_ai)
        worker.error_occurred.connect(lambda e: self._show_stats_ai(None))
        worker.start()
        self._worker = worker

    def _show_stats_ai(self, result) -> None:
        self._ai_stats_btn.setEnabled(True)
        if result:
            prefix = "⚠️ [Fallback]\n" if result.is_fallback else f"🤖 [{result.model}]\n\n"
            self._ai_stats_text.setPlainText(prefix + result.answer)

    def _run_multimodal(self) -> None:
        if self._img is None:
            return
        from aiviz.ai.agent import AnalysisAgent
        agent = AnalysisAgent()
        question = self._mm_question.toPlainText().strip()
        self._mm_btn.setEnabled(False)
        self._mm_btn.setText("Sending to LLaVA…")
        self._mm_text.setPlainText("Sending image to vision model…")

        img_ref = self._img

        def call():
            return agent.describe_image_visual(img_ref, question=question)

        worker = WorkerThread(call)
        worker.result_ready.connect(self._show_multimodal)
        worker.error_occurred.connect(lambda e: self._show_multimodal(None))
        worker.start()
        self._worker_mm = worker

    def _show_multimodal(self, result) -> None:
        self._mm_btn.setEnabled(True)
        self._mm_btn.setText("🔭 Describe Image with LLaVA")
        if result is None:
            self._mm_text.setPlainText("Error during multimodal request.")
            return
        prefix = "⚠️ [Fallback]\n" if result.is_fallback else f"🔭 [{result.model}]\n\n"
        self._mm_text.setPlainText(prefix + result.answer)
