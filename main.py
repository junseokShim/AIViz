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

    # Apply global dark theme
    app.setStyleSheet(DARK_STYLESHEET)
    apply_matplotlib_dark_theme()

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
