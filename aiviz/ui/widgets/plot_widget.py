"""
PlotWidget – embeds a Matplotlib figure inside a QWidget.

Provides:
- get_ax()        → clear figure, return a single Axes
- get_axes(r, c)  → clear figure, return an Axes grid
- redraw()        → refresh the canvas
- save(path)      → save figure to file

The NavigationToolbar gives the user zoom / pan / save for free.
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from aiviz.app.style import C_BG, C_BASE, C_OVERLAY, C_TEXT, C_SUBTEXT, SERIES_COLORS


def _style_ax(ax: Axes) -> None:
    """Apply the AIViz dark theme to an individual Axes object."""
    ax.set_facecolor(C_BASE)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_OVERLAY)
    ax.tick_params(colors=C_SUBTEXT, labelsize=9)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color("#89b4fa")
    ax.grid(True, color=C_OVERLAY, linewidth=0.5, alpha=0.7)
    ax.set_prop_cycle(color=SERIES_COLORS)


class PlotWidget(QWidget):
    """
    A QWidget that wraps a Matplotlib Figure + NavigationToolbar.

    Usage pattern:
        pw = PlotWidget()
        ax = pw.get_ax()
        ax.plot(x, y)
        pw.redraw()
    """

    def __init__(self, parent: Optional[QWidget] = None, figsize: tuple = (9, 5)):
        super().__init__(parent)
        self._setup(figsize)

    def _setup(self, figsize: tuple) -> None:
        self._fig = Figure(figsize=figsize, facecolor=C_BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._toolbar = NavigationToolbar(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def figure(self) -> Figure:
        return self._fig

    def get_ax(self) -> Axes:
        """Clear the figure and return a single styled Axes."""
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        _style_ax(ax)
        return ax

    def get_axes(self, nrows: int = 1, ncols: int = 1):
        """Clear the figure and return an Axes grid (ndarray or single Axes)."""
        self._fig.clear()
        axes = self._fig.subplots(nrows, ncols)
        if isinstance(axes, np.ndarray):
            for a in np.array(axes).flat:
                _style_ax(a)
        else:
            _style_ax(axes)
        return axes

    def redraw(self) -> None:
        """Re-render the canvas (call after modifying axes)."""
        try:
            self._fig.tight_layout(pad=1.0)
        except Exception:
            pass
        self._canvas.draw()

    def clear(self) -> None:
        """Clear all artists from the figure."""
        self._fig.clear()
        self._canvas.draw()

    def save(self, path: str, dpi: int = 150) -> None:
        """Save the current figure to a file."""
        self._fig.savefig(
            path, dpi=dpi, bbox_inches="tight",
            facecolor=self._fig.get_facecolor()
        )

    def get_png_bytes(self) -> bytes:
        """Return the current figure as PNG bytes (for export)."""
        import io
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                          facecolor=self._fig.get_facecolor())
        buf.seek(0)
        return buf.read()
