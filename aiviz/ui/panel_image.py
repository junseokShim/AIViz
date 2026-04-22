"""
Image Analysis panel – v0.2.

New in v0.2:
- Interactive ImageCanvas (click to inspect pixel, drag for ROI)
- Preprocessing controls (brightness / contrast / saturation / clipping / grayscale)
- Edge detection (Canny / Sobel / Laplacian)
- Segmentation (Threshold / Adaptive / K-Means)
- XML save / load for analysis session persistence
- Improved UI layout with Korean labels
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSplitter, QTabWidget, QTextEdit, QScrollArea,
    QComboBox, QSlider, QCheckBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QSizePolicy, QFormLayout,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage

from aiviz.app.controller import AppController, WorkerThread
from aiviz.ui.widgets.plot_widget import PlotWidget
from aiviz.ui.widgets.data_table import DataTableView
from aiviz.ui.widgets.image_canvas import ImageCanvas
from aiviz.visualization import mpl_charts


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _pil_to_pixmap(img) -> QPixmap:
    """Convert PIL Image to QPixmap."""
    rgb = img.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3,
                  QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _make_slider(lo: int, hi: int, val: int, parent: QWidget = None) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal, parent)
    s.setRange(lo, hi)
    s.setValue(val)
    return s


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class ImagePanel(QWidget):
    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._img = None             # original PIL Image (never modified)
        self._img_bytes = None
        self._ana_result = None
        self._proc_img = None        # current preprocessed image
        self._edge_result = None
        self._seg_result = None
        self._file_path = ""         # path of last loaded file (for XML)
        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(6)

        # Header row
        hdr_row = QHBoxLayout()
        hdr = QLabel("이미지 분석 (Image Analysis)")
        hdr.setObjectName("heading")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()
        self._save_xml_btn = QPushButton("💾 XML 저장")
        self._save_xml_btn.setObjectName("secondary")
        self._save_xml_btn.setEnabled(False)
        self._save_xml_btn.setToolTip("분석 결과를 XML 파일로 저장합니다")
        self._load_xml_btn = QPushButton("📂 XML 불러오기")
        self._load_xml_btn.setObjectName("secondary")
        self._load_xml_btn.setToolTip("이전 XML 분석 결과를 불러옵니다")
        hdr_row.addWidget(self._save_xml_btn)
        hdr_row.addWidget(self._load_xml_btn)
        root.addLayout(hdr_row)

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

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: interactive canvas ─────────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 4, 0)
        ll.addWidget(QLabel("이미지 미리보기 (클릭/드래그 지원)"))

        self._canvas = ImageCanvas()
        ll.addWidget(self._canvas, stretch=1)

        btn_row = QHBoxLayout()
        self._gray_btn = QPushButton("흑백 전환")
        self._gray_btn.setObjectName("secondary")
        self._gray_btn.setEnabled(False)
        self._gray_btn.setCheckable(True)
        self._gray_btn.clicked.connect(self._toggle_gray)
        btn_row.addWidget(self._gray_btn)

        self._reset_btn = QPushButton("전처리 초기화")
        self._reset_btn.setObjectName("secondary")
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self._reset_preprocessing)
        btn_row.addWidget(self._reset_btn)
        ll.addLayout(btn_row)

        # ROI info label
        self._roi_label = QLabel("")
        self._roi_label.setObjectName("meta")
        self._roi_label.setWordWrap(True)
        ll.addWidget(self._roi_label)

        splitter.addWidget(left)

        # ── Right: analysis tabs ─────────────────────────────────────
        tabs = QTabWidget()

        # Tab 0: Channel stats
        self._stats_table = DataTableView()
        tabs.addTab(self._stats_table, "채널 통계")

        # Tab 1: Histogram
        self._hist_plot = PlotWidget(figsize=(7, 4))
        tabs.addTab(self._hist_plot, "히스토그램")

        # Tab 2: Dominant colors
        self._color_plot = PlotWidget(figsize=(7, 3))
        tabs.addTab(self._color_plot, "주요 색상")

        # Tab 3: Preprocessing
        tabs.addTab(self._build_preprocess_tab(), "전처리")

        # Tab 4: Edge detection
        tabs.addTab(self._build_edge_tab(), "엣지 검출")

        # Tab 5: Segmentation
        tabs.addTab(self._build_segment_tab(), "분할")

        # Tab 6: AI Stats
        ai_stats_widget = QWidget()
        aw = QVBoxLayout(ai_stats_widget)
        self._ai_stats_btn = QPushButton("📊 통계 AI 분석")
        self._ai_stats_btn.setEnabled(False)
        aw.addWidget(self._ai_stats_btn)
        self._ai_stats_text = QTextEdit()
        self._ai_stats_text.setReadOnly(True)
        aw.addWidget(self._ai_stats_text)
        tabs.addTab(ai_stats_widget, "AI 통계")

        # Tab 7: Multimodal AI
        mm_widget = QWidget()
        mw = QVBoxLayout(mm_widget)
        mw.addWidget(QLabel("이미지를 LLaVA 비전 모델에 전송:"))
        self._mm_question = QTextEdit()
        self._mm_question.setPlaceholderText(
            "선택: 이미지에 대한 질문 입력\n"
            "(예: '결함을 설명해 주세요', '어떤 패턴이 보이나요?')"
        )
        self._mm_question.setMaximumHeight(80)
        mw.addWidget(self._mm_question)
        self._mm_btn = QPushButton("🔭 LLaVA로 이미지 분석")
        self._mm_btn.setEnabled(False)
        mw.addWidget(self._mm_btn)
        hint = QLabel("requires: ollama pull llava")
        hint.setObjectName("meta")
        mw.addWidget(hint)
        self._mm_text = QTextEdit()
        self._mm_text.setReadOnly(True)
        mw.addWidget(self._mm_text)
        tabs.addTab(mm_widget, "멀티모달 AI")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # Connect signals
        self._ai_stats_btn.clicked.connect(self._run_ai_stats)
        self._mm_btn.clicked.connect(self._run_multimodal)
        self._save_xml_btn.clicked.connect(self._save_xml)
        self._load_xml_btn.clicked.connect(self._load_xml)
        self._canvas.pixel_inspected.connect(self._on_pixel_inspected)
        self._canvas.roi_selected.connect(self._on_roi_selected)

    # ── Preprocessing tab ────────────────────────────────────────────

    def _build_preprocess_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        form_grp = QGroupBox("조정 파라미터")
        form = QFormLayout(form_grp)

        # Brightness 0-300 (÷100 → 0.0-3.0, default 100)
        self._br_slider = _make_slider(10, 300, 100)
        self._br_val = QLabel("1.00")
        self._br_slider.valueChanged.connect(
            lambda v: self._br_val.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(self._br_slider)
        row.addWidget(self._br_val)
        form.addRow("밝기 (Brightness):", row)

        # Contrast
        self._ct_slider = _make_slider(10, 300, 100)
        self._ct_val = QLabel("1.00")
        self._ct_slider.valueChanged.connect(
            lambda v: self._ct_val.setText(f"{v / 100:.2f}")
        )
        row2 = QHBoxLayout()
        row2.addWidget(self._ct_slider)
        row2.addWidget(self._ct_val)
        form.addRow("대비 (Contrast):", row2)

        # Saturation
        self._sat_slider = _make_slider(0, 300, 100)
        self._sat_val = QLabel("1.00")
        self._sat_slider.valueChanged.connect(
            lambda v: self._sat_val.setText(f"{v / 100:.2f}")
        )
        row3 = QHBoxLayout()
        row3.addWidget(self._sat_slider)
        row3.addWidget(self._sat_val)
        form.addRow("채도 (Saturation):", row3)

        # Clip min/max
        self._clip_min_slider = _make_slider(0, 254, 0)
        self._clip_min_val = QLabel("0")
        self._clip_min_slider.valueChanged.connect(
            lambda v: self._clip_min_val.setText(str(v))
        )
        row4 = QHBoxLayout()
        row4.addWidget(self._clip_min_slider)
        row4.addWidget(self._clip_min_val)
        form.addRow("클립 최솟값:", row4)

        self._clip_max_slider = _make_slider(1, 255, 255)
        self._clip_max_val = QLabel("255")
        self._clip_max_slider.valueChanged.connect(
            lambda v: self._clip_max_val.setText(str(v))
        )
        row5 = QHBoxLayout()
        row5.addWidget(self._clip_max_slider)
        row5.addWidget(self._clip_max_val)
        form.addRow("클립 최댓값:", row5)

        self._norm_chk = QCheckBox("클립 후 정규화 (0-255 스케일)")
        form.addRow("", self._norm_chk)
        layout.addWidget(form_grp)

        self._apply_preproc_btn = QPushButton("전처리 적용")
        self._apply_preproc_btn.setEnabled(False)
        self._apply_preproc_btn.clicked.connect(self._apply_preprocessing)
        layout.addWidget(self._apply_preproc_btn)

        self._preproc_plot = PlotWidget(figsize=(6, 4))
        layout.addWidget(self._preproc_plot)

        return w

    # ── Edge detection tab ───────────────────────────────────────────

    def _build_edge_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        ctrl_grp = QGroupBox("엣지 검출 설정")
        cl = QFormLayout(ctrl_grp)

        self._edge_method = QComboBox()
        self._edge_method.addItems(["canny", "sobel", "laplacian"])
        cl.addRow("방법:", self._edge_method)

        self._edge_lo = QSpinBox()
        self._edge_lo.setRange(0, 255)
        self._edge_lo.setValue(50)
        cl.addRow("Canny 하한 임계값:", self._edge_lo)

        self._edge_hi = QSpinBox()
        self._edge_hi.setRange(0, 255)
        self._edge_hi.setValue(150)
        cl.addRow("Canny 상한 임계값:", self._edge_hi)

        self._edge_sigma = QDoubleSpinBox()
        self._edge_sigma.setRange(0.0, 10.0)
        self._edge_sigma.setValue(1.5)
        self._edge_sigma.setSingleStep(0.5)
        cl.addRow("Gaussian σ:", self._edge_sigma)

        layout.addWidget(ctrl_grp)

        btn_row = QHBoxLayout()
        self._edge_btn = QPushButton("엣지 검출 실행")
        self._edge_btn.setEnabled(False)
        self._edge_btn.clicked.connect(self._run_edge)
        btn_row.addWidget(self._edge_btn)

        self._edge_overlay_btn = QPushButton("오버레이 표시")
        self._edge_overlay_btn.setObjectName("secondary")
        self._edge_overlay_btn.setEnabled(False)
        self._edge_overlay_btn.clicked.connect(self._show_edge_overlay)
        btn_row.addWidget(self._edge_overlay_btn)
        layout.addLayout(btn_row)

        self._edge_plot = PlotWidget(figsize=(6, 4))
        layout.addWidget(self._edge_plot)

        cv2_note = QLabel("OpenCV 미설치 시 Canny는 scipy 근사 구현 사용")
        cv2_note.setObjectName("meta")
        layout.addWidget(cv2_note)

        return w

    # ── Segmentation tab ─────────────────────────────────────────────

    def _build_segment_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        ctrl_grp = QGroupBox("분할 설정")
        cl = QFormLayout(ctrl_grp)

        self._seg_method = QComboBox()
        self._seg_method.addItems(["threshold", "adaptive", "kmeans"])
        cl.addRow("방법:", self._seg_method)

        self._seg_threshold = QSpinBox()
        self._seg_threshold.setRange(0, 255)
        self._seg_threshold.setValue(128)
        cl.addRow("임계값 (threshold):", self._seg_threshold)

        self._seg_block = QSpinBox()
        self._seg_block.setRange(3, 201)
        self._seg_block.setSingleStep(2)
        self._seg_block.setValue(51)
        cl.addRow("블록 크기 (adaptive):", self._seg_block)

        self._seg_offset = QSpinBox()
        self._seg_offset.setRange(-50, 50)
        self._seg_offset.setValue(10)
        cl.addRow("오프셋 (adaptive):", self._seg_offset)

        self._seg_k = QSpinBox()
        self._seg_k.setRange(2, 20)
        self._seg_k.setValue(3)
        cl.addRow("클러스터 수 (kmeans):", self._seg_k)

        layout.addWidget(ctrl_grp)

        btn_row = QHBoxLayout()
        self._seg_btn = QPushButton("분할 실행")
        self._seg_btn.setEnabled(False)
        self._seg_btn.clicked.connect(self._run_segmentation)
        btn_row.addWidget(self._seg_btn)

        self._seg_overlay_btn = QPushButton("오버레이 표시")
        self._seg_overlay_btn.setObjectName("secondary")
        self._seg_overlay_btn.setEnabled(False)
        self._seg_overlay_btn.clicked.connect(self._show_seg_overlay)
        btn_row.addWidget(self._seg_overlay_btn)
        layout.addLayout(btn_row)

        self._seg_info_label = QLabel("")
        self._seg_info_label.setObjectName("meta")
        layout.addWidget(self._seg_info_label)

        self._seg_plot = PlotWidget(figsize=(6, 4))
        layout.addWidget(self._seg_plot)

        return w

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _on_data_loaded(self, result) -> None:
        if not result.ok or result.image is None:
            self._img = None
            self._img_bytes = None
            self._canvas.clear()
            self._gray_btn.setEnabled(False)
            self._reset_btn.setEnabled(False)
            self._apply_preproc_btn.setEnabled(False)
            self._edge_btn.setEnabled(False)
            self._seg_btn.setEnabled(False)
            self._ai_stats_btn.setEnabled(False)
            self._mm_btn.setEnabled(False)
            self._save_xml_btn.setEnabled(False)
            return

        self._img = result.image
        self._proc_img = self._img
        self._img_bytes = result.raw_bytes
        self._file_path = result.file_name

        from aiviz.analytics.image_analysis import analyze_image
        try:
            self._ana_result = analyze_image(self._img, file_name=result.file_name)
        except Exception as exc:
            self._ctrl.log_message.emit(f"[이미지 분석 오류] {exc}")
            self._ana_result = None
            return

        r = self._ana_result
        self._m_labels["Size"].setText(f"{r.width}×{r.height}")
        self._m_labels["Mode"].setText(r.mode)
        self._m_labels["Channels"].setText(str(r.n_channels))
        self._m_labels["Pixels"].setText(f"{r.n_pixels:,}")

        # Show in interactive canvas
        self._canvas.show_image(self._img)

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

        # Enable buttons
        self._gray_btn.setEnabled(True)
        self._reset_btn.setEnabled(True)
        self._apply_preproc_btn.setEnabled(True)
        self._edge_btn.setEnabled(True)
        self._seg_btn.setEnabled(True)
        self._ai_stats_btn.setEnabled(True)
        self._mm_btn.setEnabled(True)
        self._save_xml_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Preview controls
    # ------------------------------------------------------------------

    def _toggle_gray(self, checked: bool) -> None:
        if self._img is None:
            return
        display_img = self._proc_img if self._proc_img else self._img
        if checked:
            self._canvas.show_image(display_img.convert("L").convert("RGB"))
        else:
            self._canvas.show_image(display_img)

    def _reset_preprocessing(self) -> None:
        if self._img is None:
            return
        self._proc_img = self._img
        self._br_slider.setValue(100)
        self._ct_slider.setValue(100)
        self._sat_slider.setValue(100)
        self._clip_min_slider.setValue(0)
        self._clip_max_slider.setValue(255)
        self._norm_chk.setChecked(False)
        self._gray_btn.setChecked(False)
        self._canvas.show_image(self._img)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _apply_preprocessing(self) -> None:
        if self._img is None:
            return
        from aiviz.analytics.image_preprocess import PreprocessParams, apply_preprocessing

        params = PreprocessParams(
            brightness=self._br_slider.value() / 100.0,
            contrast=self._ct_slider.value() / 100.0,
            saturation=self._sat_slider.value() / 100.0,
            clip_min=self._clip_min_slider.value(),
            clip_max=self._clip_max_slider.value(),
            grayscale=False,
            normalize=self._norm_chk.isChecked(),
        )

        try:
            self._proc_img = apply_preprocessing(self._img, params)
        except Exception as exc:
            self._ctrl.log_message.emit(f"[전처리 오류] {exc}")
            return

        # Update canvas
        self._canvas.show_image(self._proc_img)

        # Show preprocessed histogram
        import numpy as np
        ax = self._preproc_plot.get_ax()
        arr = np.array(self._proc_img)
        ax.set_title("전처리 후 픽셀 히스토그램")
        from aiviz.analytics.image_analysis import analyze_image
        try:
            res = analyze_image(self._proc_img, "preprocessed")
            mpl_charts.plot_pixel_histogram(ax, res.histograms)
        except Exception:
            ax.text(0.5, 0.5, "히스토그램 불가", ha="center", va="center")
        self._preproc_plot.redraw()

    # ------------------------------------------------------------------
    # Edge detection
    # ------------------------------------------------------------------

    def _run_edge(self) -> None:
        if self._proc_img is None:
            return
        from aiviz.analytics.image_edges import detect_edges

        method = self._edge_method.currentText()
        lo = self._edge_lo.value()
        hi = self._edge_hi.value()
        sigma = self._edge_sigma.value()

        try:
            self._edge_result = detect_edges(
                self._proc_img, method=method,
                low_threshold=lo, high_threshold=hi, sigma=sigma
            )
        except Exception as exc:
            self._ctrl.log_message.emit(f"[엣지 검출 오류] {exc}")
            return

        # Display edge map
        ax = self._edge_plot.get_ax()
        mpl_charts.plot_edge_result(ax, self._edge_result.edges, method=method)
        self._edge_plot.redraw()

        self._edge_overlay_btn.setEnabled(True)
        backend = "cv2" if self._edge_result.has_cv2 else "scipy (근사)"
        self._ctrl.status_message.emit(f"엣지 검출 완료 ({method}, {backend})")

    def _show_edge_overlay(self) -> None:
        if self._edge_result is None or self._proc_img is None:
            return
        import numpy as np
        self._canvas.show_overlay(self._proc_img, self._edge_result.edges, alpha=0.5)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _run_segmentation(self) -> None:
        if self._proc_img is None:
            return
        from aiviz.analytics.image_segmentation import segment_image, mask_to_pil

        method = self._seg_method.currentText()
        kwargs = {}
        if method == "threshold":
            kwargs["threshold"] = self._seg_threshold.value()
        elif method == "adaptive":
            kwargs["block_size"] = self._seg_block.value()
            kwargs["offset"] = self._seg_offset.value()
        elif method == "kmeans":
            kwargs["n_clusters"] = self._seg_k.value()

        try:
            self._seg_result = segment_image(self._proc_img, method=method, **kwargs)
        except ImportError as exc:
            self._ctrl.log_message.emit(f"[분할 오류] {exc}")
            return
        except Exception as exc:
            self._ctrl.log_message.emit(f"[분할 오류] {exc}")
            return

        r = self._seg_result
        # Cap display regions to avoid colormap overload
        import numpy as np
        display_n = min(r.n_regions, 50)
        mask_display = np.where(r.mask > display_n, 0, r.mask)

        ax = self._seg_plot.get_ax()
        base = np.array(self._proc_img.convert("RGB"))
        mpl_charts.plot_segmentation_result(
            ax, base, mask_display, display_n, method=method
        )
        self._seg_plot.redraw()

        self._seg_info_label.setText(f"검출된 영역: {r.n_regions:,}개")
        self._seg_overlay_btn.setEnabled(True)
        self._ctrl.status_message.emit(f"분할 완료 ({method}, {r.n_regions:,} 영역)")

    def _show_seg_overlay(self) -> None:
        if self._seg_result is None or self._proc_img is None:
            return
        from aiviz.analytics.image_segmentation import mask_to_pil
        import numpy as np
        overlay_pil = mask_to_pil(self._seg_result, alpha=150)
        overlay_arr = np.array(overlay_pil)
        self._canvas.show_overlay(self._proc_img, overlay_arr)

    # ------------------------------------------------------------------
    # Canvas interactions
    # ------------------------------------------------------------------

    def _on_pixel_inspected(self, x: int, y: int, pixel) -> None:
        import numpy as np
        if hasattr(pixel, "__len__") and len(pixel) >= 3:
            self._roi_label.setText(
                f"픽셀 ({x}, {y}): R={pixel[0]}  G={pixel[1]}  B={pixel[2]}"
            )
        else:
            self._roi_label.setText(f"픽셀 ({x}, {y}): {pixel}")

    def _on_roi_selected(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if self._proc_img is None:
            return
        import numpy as np
        arr = np.array(self._proc_img.convert("RGB"))
        roi = arr[y1:y2, x1:x2]
        if roi.size == 0:
            return
        h, w = roi.shape[:2]
        mean_r = float(np.mean(roi[:, :, 0]))
        mean_g = float(np.mean(roi[:, :, 1]))
        mean_b = float(np.mean(roi[:, :, 2]))
        self._roi_label.setText(
            f"ROI ({x1},{y1})→({x2},{y2})  크기:{w}×{h}px  "
            f"평균 R={mean_r:.1f} G={mean_g:.1f} B={mean_b:.1f}"
        )

    # ------------------------------------------------------------------
    # XML save / load
    # ------------------------------------------------------------------

    def _save_xml(self) -> None:
        if self._ana_result is None:
            return
        from aiviz.analytics.image_preprocess import PreprocessParams
        from aiviz.services.image_xml_service import ImageSessionData, save_xml

        save_path, _ = QFileDialog.getSaveFileName(
            self, "XML 저장", "", "XML Files (*.xml)"
        )
        if not save_path:
            return

        # Collect current preprocessing params
        params = PreprocessParams(
            brightness=self._br_slider.value() / 100.0,
            contrast=self._ct_slider.value() / 100.0,
            saturation=self._sat_slider.value() / 100.0,
            clip_min=self._clip_min_slider.value(),
            clip_max=self._clip_max_slider.value(),
            normalize=self._norm_chk.isChecked(),
        )

        edge_method = self._edge_method.currentText()
        edge_params = {}
        if self._edge_result is not None:
            edge_params = self._edge_result.params

        seg_method = self._seg_method.currentText()
        seg_params = {}
        if self._seg_result is not None:
            seg_params = self._seg_result.params

        r = self._ana_result
        ch_stats = r.channel_stats.to_dict(orient="records") if r.channel_stats is not None else []

        session = ImageSessionData(
            image_path=self._file_path,
            image_size=(r.width, r.height),
            image_mode=r.mode,
            preprocess_params=params.to_dict(),
            edge_method=edge_method,
            edge_params=edge_params,
            segment_method=seg_method,
            segment_params=seg_params,
            channel_stats=ch_stats,
        )

        try:
            save_xml(session, save_path)
            self._ctrl.status_message.emit(f"XML 저장 완료: {save_path}")
        except Exception as exc:
            self._ctrl.log_message.emit(f"[XML 저장 오류] {exc}")

    def _load_xml(self) -> None:
        from aiviz.services.image_xml_service import load_xml
        from aiviz.analytics.image_preprocess import PreprocessParams

        xml_path, _ = QFileDialog.getOpenFileName(
            self, "XML 불러오기", "", "XML Files (*.xml)"
        )
        if not xml_path:
            return

        session = load_xml(xml_path)
        if session is None:
            self._ctrl.log_message.emit("[XML 오류] 유효하지 않은 AIViz XML 파일입니다.")
            return

        # Restore preprocessing sliders
        pp = PreprocessParams.from_dict(session.preprocess_params)
        self._br_slider.setValue(int(pp.brightness * 100))
        self._ct_slider.setValue(int(pp.contrast * 100))
        self._sat_slider.setValue(int(pp.saturation * 100))
        self._clip_min_slider.setValue(pp.clip_min)
        self._clip_max_slider.setValue(pp.clip_max)
        self._norm_chk.setChecked(pp.normalize)

        # Restore edge/seg method selections
        idx = self._edge_method.findText(session.edge_method)
        if idx >= 0:
            self._edge_method.setCurrentIndex(idx)

        idx = self._seg_method.findText(session.segment_method)
        if idx >= 0:
            self._seg_method.setCurrentIndex(idx)

        # Restore edge params
        ep = session.edge_params
        if "low_threshold" in ep:
            self._edge_lo.setValue(int(ep["low_threshold"]))
        if "high_threshold" in ep:
            self._edge_hi.setValue(int(ep["high_threshold"]))
        if "sigma" in ep:
            self._edge_sigma.setValue(float(ep["sigma"]))

        # Restore segment params
        sp = session.segment_params
        if "threshold" in sp:
            self._seg_threshold.setValue(int(sp["threshold"]))
        if "block_size" in sp:
            self._seg_block.setValue(int(sp["block_size"]))
        if "offset" in sp:
            self._seg_offset.setValue(int(sp["offset"]))
        if "n_clusters" in sp:
            self._seg_k.setValue(int(sp["n_clusters"]))

        self._ctrl.status_message.emit(
            f"XML 로드 완료: {Path(xml_path).name}  "
            f"(이미지: {session.image_path or '없음'})"
        )

        # If image is loaded, apply the preprocessing now
        if self._img is not None and not pp.is_identity():
            self._apply_preprocessing()

    # ------------------------------------------------------------------
    # AI analysis
    # ------------------------------------------------------------------

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
        self._mm_btn.setText("LLaVA 전송 중…")
        self._mm_text.setPlainText("비전 모델에 이미지 전송 중…")

        img_ref = self._proc_img or self._img

        def call():
            return agent.describe_image_visual(img_ref, question=question)

        worker = WorkerThread(call)
        worker.result_ready.connect(self._show_multimodal)
        worker.error_occurred.connect(lambda e: self._show_multimodal(None))
        worker.start()
        self._worker_mm = worker

    def _show_multimodal(self, result) -> None:
        self._mm_btn.setEnabled(True)
        self._mm_btn.setText("🔭 LLaVA로 이미지 분석")
        if result is None:
            self._mm_text.setPlainText("멀티모달 요청 중 오류가 발생했습니다.")
            return
        prefix = "⚠️ [Fallback]\n" if result.is_fallback else f"🔭 [{result.model}]\n\n"
        self._mm_text.setPlainText(prefix + result.answer)
