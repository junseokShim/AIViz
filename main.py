"""
AIViz Desktop Application – main entrypoint.

Run with:
    python main.py
"""

import sys
import os

# Ensure project root on path so aiviz package and config are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set matplotlib backend before any import of pyplot/backends
import matplotlib
matplotlib.use("QtAgg")

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon
except ImportError:
    print("PyQt6 is required. Install with:\n  pip install PyQt6")
    sys.exit(1)

from aiviz.app.main_window import MainWindow
from aiviz.app.style import DARK_STYLESHEET, apply_matplotlib_dark_theme
from config import APP


def main() -> None:
    # Enable high-DPI scaling (Qt6 handles this automatically but explicit is safe)
    app = QApplication(sys.argv)
    app.setApplicationName(APP.name)
    app.setApplicationVersion(APP.version)
    app.setOrganizationName("junseokShim")

    # Apply global dark theme + Korean font configuration
    app.setStyleSheet(DARK_STYLESHEET)
    apply_matplotlib_dark_theme()

    # Configure Qt font fallback for Korean text
    try:
        from aiviz.utils.font_utils import get_korean_qt_font_family
        korean_family = get_korean_qt_font_family()
        if korean_family:
            from PyQt6.QtGui import QFont
            default_font = app.font()
            fallback_fonts = default_font.family()
            # Qt6 does not have setFamilies on QApplication directly;
            # the stylesheet already lists Korean-capable fallback fonts.
            # Just log the result so we know which font was selected.
            import sys as _sys
            print(f"[Font] Korean font resolved: '{korean_family}'", file=_sys.stderr)
    except Exception:
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
