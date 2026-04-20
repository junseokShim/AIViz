"""
AppController – central application state manager.

Holds the currently loaded data/image and broadcasts state changes to all
panels via Qt signals. Panels never talk to each other directly; they only
talk to the controller.

Threading:
- File loading runs in a background QThread (WorkerThread)
- AI calls are also run in background threads managed by individual panels
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from PIL import Image

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from aiviz.ingestion.loader import load_file, DataLoadResult

logger = logging.getLogger("aiviz.controller")


# ---------------------------------------------------------------------------
# Generic background worker
# ---------------------------------------------------------------------------

class WorkerThread(QThread):
    """
    Generic QThread that runs a callable in the background.

    Usage:
        worker = WorkerThread(fn, arg1, kwarg=val)
        worker.result_ready.connect(handler)
        worker.error_occurred.connect(err_handler)
        worker.start()
    """

    result_ready = pyqtSignal(object)   # emits whatever fn() returns
    error_occurred = pyqtSignal(str)    # emits error message string

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.result_ready.emit(result)
        except Exception as exc:
            logger.exception("WorkerThread error")
            self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# Application controller
# ---------------------------------------------------------------------------

class AppController(QObject):
    """
    Single source of truth for the loaded dataset / image.

    Signals
    -------
    data_loaded(DataLoadResult)
        Emitted once a file has been successfully parsed.
    status_message(str)
        Short status string for the status bar.
    log_message(str)
        Longer message appended to the log dock.
    loading_started()
        Emitted just before a file is loaded (shows progress).
    loading_finished()
        Emitted after loading completes (hides progress).
    """

    data_loaded = pyqtSignal(object)      # DataLoadResult
    status_message = pyqtSignal(str)
    log_message = pyqtSignal(str)
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._result: Optional[DataLoadResult] = None
        self._worker: Optional[WorkerThread] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def result(self) -> Optional[DataLoadResult]:
        return self._result

    @property
    def df(self) -> Optional[pd.DataFrame]:
        return self._result.df if self._result else None

    @property
    def image(self) -> Optional[Image.Image]:
        return self._result.image if self._result else None

    @property
    def file_name(self) -> str:
        return self._result.file_name if self._result else ""

    @property
    def file_type(self) -> str:
        return self._result.file_type if self._result else ""

    @property
    def has_data(self) -> bool:
        return self._result is not None and self._result.ok

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def load_file_async(self, file_bytes: bytes, file_name: str) -> None:
        """Load a file in a background thread; emits signals on completion."""
        if self._worker and self._worker.isRunning():
            self._worker.quit()

        self.loading_started.emit()
        self.status_message.emit(f"Loading {file_name}…")

        self._worker = WorkerThread(load_file, file_bytes, file_name)
        self._worker.result_ready.connect(self._on_load_done)
        self._worker.error_occurred.connect(self._on_load_error)
        self._worker.start()

    def _on_load_done(self, result: DataLoadResult) -> None:
        self.loading_finished.emit()
        if result.ok:
            self._result = result
            rows = len(result.df) if result.df is not None else "–"
            cols = len(result.df.columns) if result.df is not None else "–"
            mode = f"{result.image.size[0]}×{result.image.size[1]}" if result.image else ""
            info = f"{rows} rows × {cols} cols" if result.file_type == "tabular" else mode
            self.status_message.emit(f"Loaded: {result.file_name}  ({info})")
            self.log_message.emit(f"[OK] Loaded: {result.file_name}")
            self.data_loaded.emit(result)
        else:
            self.status_message.emit(f"Load failed: {result.error}")
            self.log_message.emit(f"[ERROR] {result.error}")

    def _on_load_error(self, msg: str) -> None:
        self.loading_finished.emit()
        self.status_message.emit(f"Error: {msg}")
        self.log_message.emit(f"[ERROR] {msg}")

    def clear(self) -> None:
        """Clear loaded data and notify all panels."""
        self._result = None
        self.status_message.emit("Ready")
        self.log_message.emit("[INFO] Data cleared.")
        # Emit an empty result so panels reset themselves
        self.data_loaded.emit(DataLoadResult())
