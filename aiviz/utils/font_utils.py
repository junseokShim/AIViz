"""
Korean-aware font utilities for AIViz.

Provides:
- safe QFont creation with clamped sizes
- Korean font fallback resolution for Qt
- matplotlib Korean font configuration
"""

from __future__ import annotations

import sys
from typing import Optional

_MIN_SIZE = 8
_MAX_SIZE = 72

# Priority-ordered Korean font candidates per platform
_KOREAN_FONTS: dict[str, list[str]] = {
    "darwin": ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "나눔고딕"],
    "win32":  ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Gulim", "굴림", "Dotum", "돋움"],
    "linux":  ["NanumGothic", "나눔고딕", "UnDotum", "Baekmuk Dotum"],
}


def get_korean_qt_font_family() -> str:
    """Return the best available Korean font family name for Qt, or '' if none found."""
    try:
        from PyQt6.QtGui import QFontDatabase
        available = set(QFontDatabase.families())
    except Exception:
        return ""

    platform_key = sys.platform
    candidates: list[str] = list(_KOREAN_FONTS.get(platform_key, []))
    # Also add candidates from other platforms as fallback
    for k, v in _KOREAN_FONTS.items():
        if k != platform_key:
            candidates.extend(v)

    for name in candidates:
        if name in available:
            return name
    return ""


def safe_font(family: str = "", size: int = 11, bold: bool = False):
    """Return a QFont with point size clamped to [8, 72]."""
    from PyQt6.QtGui import QFont
    clamped = max(_MIN_SIZE, min(_MAX_SIZE, size))
    font = QFont(family, clamped)
    if bold:
        font.setBold(True)
    return font


def safe_korean_font(size: int = 11, bold: bool = False):
    """Return a Korean-capable QFont with clamped size."""
    family = get_korean_qt_font_family()
    return safe_font(family, size, bold)


def clamp_font_size(size: int) -> int:
    """Clamp a raw point size to a valid positive value."""
    return max(_MIN_SIZE, min(_MAX_SIZE, int(size)))


def configure_matplotlib_korean() -> None:
    """Configure matplotlib to render Korean text correctly.

    Call this once at startup (after apply_matplotlib_dark_theme).
    """
    import matplotlib
    import matplotlib.font_manager as fm

    platform_key = sys.platform
    candidates: list[str] = list(_KOREAN_FONTS.get(platform_key, []))
    for k, v in _KOREAN_FONTS.items():
        if k != platform_key:
            candidates.extend(v)

    available = {f.name for f in fm.fontManager.ttflist}
    korean_family: Optional[str] = None
    for name in candidates:
        if name in available:
            korean_family = name
            break

    current = list(matplotlib.rcParams.get("font.sans-serif", []))
    if korean_family:
        # Prepend Korean font so it takes priority
        if korean_family not in current:
            matplotlib.rcParams["font.sans-serif"] = [korean_family] + current
    # Always disable unicode minus to prevent □ characters
    matplotlib.rcParams["axes.unicode_minus"] = False
