"""
Interactive image canvas widget.

Embeds a matplotlib figure inside a QWidget, providing:
- PIL Image display (scaled to fit)
- Click → inspect pixel value and emit pixel_inspected signal
- Click+drag → ROI rectangle selection and emit roi_selected signal
- Overlay display for edge maps and segmentation masks
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


class ImageCanvas(QWidget):
    """
    PyQt widget wrapping a matplotlib Axes for interactive image display.

    Signals:
        pixel_inspected(x, y, pixel_value): emitted on mouse click.
        roi_selected(x1, y1, x2, y2):       emitted when drag > 5px finishes.
    """

    pixel_inspected = pyqtSignal(int, int, object)   # x, y, pixel array/scalar
    roi_selected    = pyqtSignal(int, int, int, int)  # x1, y1, x2, y2

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._img_arr: Optional[np.ndarray] = None
        self._overlay_arr: Optional[np.ndarray] = None
        self._roi_start: Optional[tuple[int, int]] = None
        self._roi_patch: Optional[Rectangle] = None
        self._img_axes = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._fig = Figure(figsize=(5, 4), dpi=96)
        self._fig.patch.set_facecolor("#1e1e2e")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._canvas, stretch=1)

        self._info_label = QLabel("이미지를 로드하세요")
        self._info_label.setObjectName("meta")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._info_label)

        # Connect mouse events
        self._canvas.mpl_connect("button_press_event",   self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._canvas.mpl_connect("motion_notify_event",  self._on_motion)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def show_image(self, img: Image.Image) -> None:
        """Display a PIL Image and clear any overlay."""
        self._img_arr = np.array(img)
        self._overlay_arr = None
        self._redraw()

    def show_overlay(self, base_img: Image.Image, overlay: np.ndarray, alpha: float = 0.4) -> None:
        """
        Display *base_img* with a colour *overlay* blended on top.

        overlay: 2-D uint8 (grayscale edge map) or 2-D int32 (label mask with colormap).
        For grayscale overlays the edges are shown in cyan.
        """
        self._img_arr = np.array(base_img.convert("RGB"))
        self._overlay_arr = overlay
        self._overlay_alpha = alpha
        self._redraw()

    def clear(self) -> None:
        """Clear image and overlay."""
        self._img_arr = None
        self._overlay_arr = None
        if self._img_axes is not None:
            self._img_axes.cla()
            self._img_axes.axis("off")
        self._canvas.draw_idle()
        self._info_label.setText("이미지를 로드하세요")

    # ------------------------------------------------------------------
    # Private: rendering
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        if self._img_arr is None:
            return

        self._fig.clear()
        ax = self._fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("#1e1e2e")
        ax.axis("off")
        self._img_axes = ax

        ax.imshow(self._img_arr, aspect="auto", interpolation="nearest")

        if self._overlay_arr is not None:
            alpha = getattr(self, "_overlay_alpha", 0.4)
            ov = self._overlay_arr
            if ov.ndim == 2 and ov.dtype == np.uint8:
                # Edge map: show as semi-transparent cyan channel
                h, w = ov.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[ov > 0] = [0, 200, 255, int(alpha * 255)]
                ax.imshow(rgba, aspect="auto", interpolation="nearest")
            elif ov.ndim == 3 and ov.shape[2] == 4:
                # RGBA overlay (segmentation colormap)
                ax.imshow(ov, aspect="auto", interpolation="nearest")

        self._canvas.draw_idle()
        h, w = self._img_arr.shape[:2]
        self._info_label.setText(
            f"클릭: 픽셀 정보 | 드래그: ROI 선택  [{w}×{h}]"
        )

    # ------------------------------------------------------------------
    # Private: mouse interaction
    # ------------------------------------------------------------------

    def _on_press(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        self._roi_start = (x, y)

        # Inspect pixel
        if self._img_arr is not None:
            h, w = self._img_arr.shape[:2]
            cx = max(0, min(w - 1, x))
            cy = max(0, min(h - 1, y))
            pixel = self._img_arr[cy, cx]
            self.pixel_inspected.emit(cx, cy, pixel)
            if self._img_arr.ndim == 3:
                r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
                self._info_label.setText(f"픽셀 ({cx}, {cy}) → R={r}  G={g}  B={b}")
            else:
                self._info_label.setText(f"픽셀 ({cx}, {cy}) → {int(pixel)}")

    def _on_motion(self, event) -> None:
        if self._roi_start is None or event.xdata is None or event.ydata is None:
            return
        if self._img_axes is None:
            return

        x0, y0 = self._roi_start
        x1, y1 = int(round(event.xdata)), int(round(event.ydata))

        if self._roi_patch is not None:
            try:
                self._roi_patch.remove()
            except Exception:
                pass

        rx, ry = min(x0, x1), min(y0, y1)
        rw, rh = abs(x1 - x0), abs(y1 - y0)
        self._roi_patch = Rectangle(
            (rx, ry), rw, rh,
            linewidth=1.5, edgecolor="#89b4fa",
            facecolor="none", linestyle="--",
        )
        self._img_axes.add_patch(self._roi_patch)
        self._canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._roi_start is None:
            return
        if event.xdata is None or event.ydata is None:
            self._roi_start = None
            return

        x0, y0 = self._roi_start
        x1, y1 = int(round(event.xdata)), int(round(event.ydata))
        self._roi_start = None

        if abs(x1 - x0) > 5 and abs(y1 - y0) > 5:
            self.roi_selected.emit(
                min(x0, x1), min(y0, y1),
                max(x0, x1), max(y0, y1),
            )
