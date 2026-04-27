"""
AIViz Main Window.

Layout:
  ┌─────────────────────────────────────────────┐
  │ Menu bar                                    │
  ├──────────┬──────────────────────────────────┤
  │ File     │ Tab widget                        │
  │ Panel    │  Data | Charts | TimeSeries |     │
  │ (dock)   │  Frequency | Image | Forecast |   │
  │          │  AI Assistant | Export            │
  ├──────────┴──────────────────────────────────┤
  │ Status bar                                  │
  ├─────────────────────────────────────────────┤
  │ Log dock (collapsible)                      │
  └─────────────────────────────────────────────┘
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QTabWidget,
    QDockWidget, QTextEdit, QProgressBar, QLabel,
    QMenuBar, QMenu, QStatusBar, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QFont

from aiviz.app.controller import AppController
from aiviz.ui.panel_file import FilePanel
from aiviz.ui.panel_data import DataPanel
from aiviz.ui.panel_charts import ChartsPanel
from aiviz.ui.panel_timeseries import TimeSeriesPanel
from aiviz.ui.panel_frequency import FrequencyPanel
from aiviz.ui.panel_image import ImagePanel
from aiviz.ui.panel_forecast import ForecastPanel
from aiviz.ui.panel_assistant import AssistantPanel
from aiviz.ui.panel_export import ExportPanel
from aiviz.ui.panel_ml import MLPanel
from aiviz.ui.panel_db import DBPanel
from config import APP, OLLAMA


class MainWindow(QMainWindow):
    """Top-level window for AIViz."""

    def __init__(self):
        super().__init__()
        self.controller = AppController(self)

        self._setup_window()
        self._build_menu()
        self._build_central()
        self._build_log_dock()
        self._build_status_bar()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        self.setWindowTitle(f"{APP.name}  –  Local AI Analytics Platform  v{APP.version}")
        self.setMinimumSize(QSize(1280, 800))
        self.resize(QSize(1440, 900))

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu: QMenu = mb.addMenu("File")
        open_act = QAction("Open File…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_act)

        clear_act = QAction("Clear Data", self)
        clear_act.triggered.connect(self.controller.clear)
        file_menu.addAction(clear_act)

        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # View menu
        view_menu: QMenu = mb.addMenu("View")
        self._log_toggle_act = QAction("Show Log", self, checkable=True, checked=True)
        self._log_toggle_act.triggered.connect(self._toggle_log)
        view_menu.addAction(self._log_toggle_act)

        # Help menu
        help_menu: QMenu = mb.addMenu("Help")
        about_act = QAction("About AIViz", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_central(self) -> None:
        # Root splitter: file panel (left) + tab widget (right)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: file loading sidebar
        self._file_panel = FilePanel(self.controller)
        self._file_panel.setFixedWidth(230)
        self._splitter.addWidget(self._file_panel)

        # Right: analysis tabs
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._panel_data = DataPanel(self.controller)
        self._panel_charts = ChartsPanel(self.controller)
        self._panel_ts = TimeSeriesPanel(self.controller)
        self._panel_freq = FrequencyPanel(self.controller)
        self._panel_image = ImagePanel(self.controller)
        self._panel_forecast = ForecastPanel(self.controller)
        self._panel_assistant = AssistantPanel(self.controller)
        self._panel_export = ExportPanel(self.controller)
        self._panel_ml = MLPanel(self.controller)
        self._panel_db = DBPanel(self.controller)

        self._tabs.addTab(self._panel_data,      "📊  Data")
        self._tabs.addTab(self._panel_charts,    "📈  Charts")
        self._tabs.addTab(self._panel_ts,        "📉  Time-Series")
        self._tabs.addTab(self._panel_freq,      "🔊  Frequency")
        self._tabs.addTab(self._panel_image,     "🖼  Image")
        self._tabs.addTab(self._panel_forecast,  "🔮  Forecast")
        self._tabs.addTab(self._panel_assistant, "🤖  AI Assistant")
        self._tabs.addTab(self._panel_ml,        "🧠  ML")
        self._tabs.addTab(self._panel_db,        "🗄  Database")
        self._tabs.addTab(self._panel_export,    "📄  Export")

        self._splitter.addWidget(self._tabs)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)

        self.setCentralWidget(self._splitter)

    def _build_log_dock(self) -> None:
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setMaximumHeight(160)
        font = QFont("Courier New", 10)
        self._log_edit.setFont(font)
        self._log_edit.setObjectName("log")

        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setWidget(self._log_edit)
        self._log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self._log_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)

    def _build_status_bar(self) -> None:
        sb: QStatusBar = self.statusBar()

        self._status_label = QLabel("Ready")
        sb.addWidget(self._status_label, 1)

        self._ollama_label = QLabel()
        sb.addPermanentWidget(self._ollama_label)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(120)
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.hide()
        sb.addPermanentWidget(self._progress)

        self._update_ollama_status()

    def _connect_signals(self) -> None:
        ctrl = self.controller
        ctrl.status_message.connect(self._status_label.setText)
        ctrl.log_message.connect(self._append_log)
        ctrl.loading_started.connect(self._progress.show)
        ctrl.loading_finished.connect(self._progress.hide)

        # Pipe FilePanel open-file requests to controller
        self._file_panel.open_requested.connect(self._load_file_bytes)
        self._file_panel.folder_requested.connect(self._load_folder)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _open_file_dialog(self) -> None:
        from config import APP
        ext_str = " ".join(
            f"*{e}" for e in APP.supported_tabular + APP.supported_image
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            f"Supported files ({ext_str});;All files (*)"
        )
        if path:
            self._load_file_from_path(path)

    def _load_file_from_path(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                data = f.read()
            import os
            self.controller.load_file_async(data, os.path.basename(path))
        except Exception as exc:
            QMessageBox.critical(self, "File Error", str(exc))

    def _load_file_bytes(self, file_bytes: bytes, file_name: str) -> None:
        self.controller.load_file_async(file_bytes, file_name)

    def _load_folder(self, folder_path: str) -> None:
        from aiviz.ingestion.folder_loader import load_folder
        from aiviz.app.controller import WorkerThread

        self.controller.loading_started.emit()
        self.controller.status_message.emit(f"폴더 로드 중: {folder_path}")

        def do_load():
            return load_folder(folder_path, recursive=False, combine=True)

        def on_done(result):
            self.controller.loading_finished.emit()
            if not result.ok:
                QMessageBox.warning(self, "폴더 로드 실패", result.error or "알 수 없는 오류")
                return

            self.controller.status_message.emit(
                f"폴더 로드 완료: {result.loaded}/{result.total}개 파일"
            )
            self.controller.log_message.emit(result.summary_text())

            # Show dialog to choose which DataFrame to use
            self._show_folder_dialog(result)

        worker = WorkerThread(do_load)
        worker.result_ready.connect(on_done)
        worker.error_occurred.connect(lambda e: (
            self.controller.loading_finished.emit(),
            QMessageBox.critical(self, "폴더 로드 오류", e)
        ))
        worker.start()
        self._folder_worker = worker

    def _show_folder_dialog(self, folder_result) -> None:
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QDialogButtonBox, QLabel, QCheckBox
        from aiviz.ingestion.loader import DataLoadResult

        dlg = QDialog(self)
        dlg.setWindowTitle("폴더 로드 결과")
        dlg.setMinimumSize(500, 400)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel(folder_result.summary_text()))

        chk_combined = QCheckBox("결합 데이터셋 사용 (모든 파일 병합)")
        chk_combined.setChecked(folder_result.combined_df is not None)
        chk_combined.setEnabled(folder_result.combined_df is not None)
        layout.addWidget(chk_combined)

        layout.addWidget(QLabel("또는 단일 파일 선택:"))
        file_list = QListWidget()
        for name in folder_result.per_file_dfs.keys():
            file_list.addItem(name)
        layout.addWidget(file_list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if chk_combined.isChecked() and folder_result.combined_df is not None:
            df = folder_result.combined_df
            name = f"폴더: {folder_result.folder.name} ({folder_result.loaded}개 파일)"
        elif file_list.currentItem():
            selected = file_list.currentItem().text()
            df = folder_result.per_file_dfs[selected]
            name = selected
        else:
            return

        # Inject into controller as a synthetic DataLoadResult
        import io
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        self.controller.load_file_async(csv_bytes, name if name.endswith(".csv") else name + ".csv")

    def _append_log(self, msg: str) -> None:
        self._log_edit.append(msg)

    def _toggle_log(self, checked: bool) -> None:
        self._log_dock.setVisible(checked)

    def _update_ollama_status(self) -> None:
        from aiviz.ai.ollama_client import get_client
        try:
            ok = get_client().is_healthy()
        except Exception:
            ok = False
        icon = "🟢" if ok else "🔴"
        self._ollama_label.setText(
            f"  {icon} Ollama  |  model: {OLLAMA.default_model}  "
        )

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP.name}",
            f"<b>AIViz v{APP.version}</b><br>"
            "Local AI Analytics Platform<br><br>"
            "Powered by Ollama · Built with PyQt6 · Open Source<br>"
            "github.com/junseokShim/AIViz"
        )
