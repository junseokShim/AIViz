"""
Global styling: Qt stylesheet + matplotlib theme.

All colour constants are defined here so changing the theme only requires
editing this one file.
"""

import matplotlib

# ---------------------------------------------------------------------------
# Colour palette – Catppuccin Mocha
# ---------------------------------------------------------------------------
C_BG = "#1e1e2e"         # main background
C_SURFACE = "#313244"    # panels, cards
C_BASE = "#181825"       # table, text edit backgrounds
C_MANTLE = "#11111b"     # deepest layer
C_OVERLAY = "#45475a"    # borders, inactive
C_TEXT = "#cdd6f4"       # primary text
C_SUBTEXT = "#a6adc8"    # secondary text
C_BLUE = "#89b4fa"       # accent – buttons, highlights
C_LAVENDER = "#b4befe"   # hover state
C_GREEN = "#a6e3a1"      # success
C_RED = "#f38ba8"        # error / anomaly
C_ORANGE = "#fab387"     # warning
C_YELLOW = "#f9e2af"     # caution
C_MAUVE = "#cba6f7"      # AI / special
C_TEAL = "#94e2d5"       # data series
C_SKY = "#89dceb"        # secondary series


DARK_STYLESHEET = f"""
/* ── Base ────────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Apple SD Gothic Neo", "Malgun Gothic", "맑은 고딕",
                 "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}
QSplitter::handle {{ background: {C_OVERLAY}; width: 2px; height: 2px; }}

/* ── Tabs ─────────────────────────────────────────────────────── */
QTabWidget::pane {{ border: 1px solid {C_OVERLAY}; background: {C_BG}; }}
QTabBar::tab {{
    background: {C_SURFACE};
    color: {C_SUBTEXT};
    padding: 8px 18px;
    margin-right: 2px;
    border-radius: 6px 6px 0 0;
}}
QTabBar::tab:selected {{ background: {C_BLUE}; color: {C_BASE}; font-weight: bold; }}
QTabBar::tab:hover:!selected {{ background: {C_OVERLAY}; color: {C_TEXT}; }}

/* ── Buttons ──────────────────────────────────────────────────── */
QPushButton {{
    background-color: {C_BLUE};
    color: {C_BASE};
    border: none;
    padding: 6px 18px;
    border-radius: 5px;
    font-weight: bold;
}}
QPushButton:hover {{ background-color: {C_LAVENDER}; }}
QPushButton:pressed {{ background-color: {C_MAUVE}; }}
QPushButton:disabled {{ background-color: {C_OVERLAY}; color: {C_SUBTEXT}; }}
QPushButton#secondary {{
    background-color: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_OVERLAY};
}}
QPushButton#secondary:hover {{ background-color: {C_OVERLAY}; }}
QPushButton#danger {{ background-color: {C_RED}; color: {C_BASE}; }}
QPushButton#success {{ background-color: {C_GREEN}; color: {C_BASE}; }}

/* ── Inputs ───────────────────────────────────────────────────── */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_OVERLAY};
    border-radius: 5px;
    padding: 5px 8px;
    selection-background-color: {C_BLUE};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {C_BLUE};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_OVERLAY};
    selection-background-color: {C_BLUE};
    selection-color: {C_BASE};
}}

/* ── Table ────────────────────────────────────────────────────── */
QTableView {{
    background-color: {C_BASE};
    gridline-color: {C_SURFACE};
    alternate-background-color: {C_BG};
    selection-background-color: {C_BLUE};
    selection-color: {C_BASE};
    border: 1px solid {C_OVERLAY};
}}
QHeaderView::section {{
    background-color: {C_SURFACE};
    color: {C_BLUE};
    padding: 5px 8px;
    border: 1px solid {C_OVERLAY};
    font-weight: bold;
}}

/* ── Text areas ───────────────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background-color: {C_BASE};
    color: {C_TEXT};
    border: 1px solid {C_OVERLAY};
    border-radius: 4px;
    padding: 4px;
}}

/* ── Sliders ──────────────────────────────────────────────────── */
QSlider::groove:horizontal {{ height: 5px; background: {C_OVERLAY}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {C_BLUE}; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {C_BLUE}; border-radius: 2px; }}

/* ── Scroll bars ──────────────────────────────────────────────── */
QScrollBar:vertical {{ background: {C_BG}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {C_OVERLAY}; border-radius: 5px; min-height: 20px; }}
QScrollBar::handle:vertical:hover {{ background: {C_BLUE}; }}
QScrollBar:horizontal {{ background: {C_BG}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {C_OVERLAY}; border-radius: 5px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

/* ── Group boxes ──────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {C_OVERLAY};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 10px;
}}
QGroupBox::title {{
    color: {C_BLUE};
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    font-weight: bold;
}}

/* ── Dock widgets ─────────────────────────────────────────────── */
QDockWidget::title {{
    background: {C_SURFACE};
    padding: 5px 8px;
    font-weight: bold;
    color: {C_BLUE};
}}
QDockWidget {{ color: {C_TEXT}; }}

/* ── Status bar ───────────────────────────────────────────────── */
QStatusBar {{
    background: {C_MANTLE};
    color: {C_SUBTEXT};
    border-top: 1px solid {C_OVERLAY};
}}
QStatusBar::item {{ border: none; }}

/* ── Menu bar ─────────────────────────────────────────────────── */
QMenuBar {{ background-color: {C_MANTLE}; color: {C_TEXT}; }}
QMenuBar::item:selected {{ background-color: {C_SURFACE}; }}
QMenu {{
    background-color: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_OVERLAY};
}}
QMenu::item:selected {{ background-color: {C_BLUE}; color: {C_BASE}; }}

/* ── Misc ─────────────────────────────────────────────────────── */
QCheckBox {{ color: {C_TEXT}; spacing: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 3px; border: 1px solid {C_OVERLAY}; }}
QCheckBox::indicator:checked {{ background: {C_BLUE}; border-color: {C_BLUE}; }}
QLabel {{ color: {C_TEXT}; }}
QLabel#heading {{ font-size: 16px; font-weight: bold; color: {C_BLUE}; }}
QLabel#subheading {{ font-size: 13px; font-weight: bold; color: {C_MAUVE}; }}
QLabel#meta {{ color: {C_SUBTEXT}; font-size: 11px; }}
QToolBar {{ background: {C_MANTLE}; border: none; spacing: 3px; }}
QToolButton {{ background: transparent; padding: 3px; border-radius: 4px; }}
QToolButton:hover {{ background: {C_SURFACE}; }}
QProgressBar {{
    background: {C_SURFACE};
    border: 1px solid {C_OVERLAY};
    border-radius: 5px;
    text-align: center;
    color: {C_TEXT};
}}
QProgressBar::chunk {{ background: {C_BLUE}; border-radius: 4px; }}
"""


def apply_matplotlib_dark_theme() -> None:
    """Apply the AIViz dark colour scheme to all matplotlib figures."""
    matplotlib.rcParams.update({
        "figure.facecolor": C_BG,
        "axes.facecolor": C_BASE,
        "axes.edgecolor": C_OVERLAY,
        "axes.labelcolor": C_TEXT,
        "axes.titlecolor": C_BLUE,
        "xtick.color": C_SUBTEXT,
        "ytick.color": C_SUBTEXT,
        "text.color": C_TEXT,
        "grid.color": C_SURFACE,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "legend.facecolor": C_SURFACE,
        "legend.edgecolor": C_OVERLAY,
        "legend.labelcolor": C_TEXT,
        "figure.titlesize": 13,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "savefig.facecolor": C_BG,
        "savefig.edgecolor": C_BG,
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.8,
        # Always disable unicode-minus so Korean plots don't show □ for '-'
        "axes.unicode_minus": False,
    })

    # Korean font – try platform fonts; silently skip if none found
    _configure_korean_font()


def _configure_korean_font() -> None:
    """Prepend a Korean-capable font to matplotlib's sans-serif list."""
    import sys
    import matplotlib.font_manager as fm

    _candidates: dict[str, list[str]] = {
        "darwin": ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic"],
        "win32":  ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Gulim"],
        "linux":  ["NanumGothic", "나눔고딕", "UnDotum"],
    }
    platform_key = sys.platform
    priority = list(_candidates.get(platform_key, []))
    # Add other-platform names as last-resort fallback
    for k, v in _candidates.items():
        if k != platform_key:
            priority.extend(v)

    available = {f.name for f in fm.fontManager.ttflist}
    for name in priority:
        if name in available:
            current = list(matplotlib.rcParams.get("font.sans-serif", []))
            if name not in current:
                matplotlib.rcParams["font.sans-serif"] = [name] + current
            break  # first match is enough


# Colour cycle for data series
SERIES_COLORS = [C_BLUE, C_GREEN, C_ORANGE, C_MAUVE, C_TEAL, C_RED, C_YELLOW, C_SKY]
