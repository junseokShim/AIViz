"""
Safe font utilities – ensures PyQt6 never receives invalid font sizes.

QFont::setPointSize crashes / pollutes logs when size <= 0.
Always use safe_font() instead of constructing QFont directly with arbitrary sizes.
"""

from __future__ import annotations

from PyQt6.QtGui import QFont

_MIN_SIZE = 8
_MAX_SIZE = 72


def safe_font(family: str = "", size: int = 11, bold: bool = False) -> QFont:
    """Return a QFont with point size clamped to [8, 72]."""
    clamped = max(_MIN_SIZE, min(_MAX_SIZE, size))
    font = QFont(family, clamped)
    if bold:
        font.setBold(True)
    return font


def clamp_font_size(size: int) -> int:
    """Clamp a raw point size to a valid positive value."""
    return max(_MIN_SIZE, min(_MAX_SIZE, int(size)))
