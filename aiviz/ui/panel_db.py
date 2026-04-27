"""
Database Integration Panel.

Provides:
  A. Connection Management – SQLite file or external DB (MySQL/PostgreSQL)
  B. Table browser         – list, preview, metadata
  C. SQL query editor      – execute arbitrary SELECT queries
  D. Import / Export       – load table → active dataset, save DataFrame → table

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │ [Connection GroupBox]                                    │
  ├──────────────────┬───────────────────────────────────────┤
  │ Table List       │ Tabs: Preview | Schema | Query | Export│
  │ (sidebar)        │                                       │
  └──────────────────┴───────────────────────────────────────┘
"""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QComboBox, QSplitter, QTabWidget,
    QTextEdit, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QFormLayout, QSpinBox, QCheckBox, QSizePolicy,
)
from PyQt6.QtCore import Qt

from aiviz.app.controller import AppController, WorkerThread
from aiviz.ui.widgets.data_table import DataTableView
from aiviz.db.db_service import DBService
from aiviz.db.models import DBType


class DBPanel(QWidget):
    """Database integration panel for AIViz."""

    def __init__(self, controller: AppController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._service = DBService()
        self._current_table: str | None = None
        self._df: pd.DataFrame | None = None   # currently loaded dataset
        self._worker = None

        self._setup_ui()
        self._ctrl.data_loaded.connect(self._on_data_loaded)

    # ─────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        hdr = QLabel("데이터베이스 연동")
        hdr.setObjectName("heading")
        root.addWidget(hdr)

        # ── Connection panel ─────────────────────────────────────────────
        root.addWidget(self._build_connection_group())

        # ── Main splitter: table list | content ──────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Left: table list
        left = QWidget()
        left.setFixedWidth(200)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)

        ll.addWidget(QLabel("테이블 목록"))
        self._table_list = QListWidget()
        self._table_list.itemClicked.connect(self._on_table_selected)
        ll.addWidget(self._table_list)

        self._refresh_btn = QPushButton("목록 새로고침")
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.clicked.connect(self._refresh_tables)
        ll.addWidget(self._refresh_btn)

        splitter.addWidget(left)

        # Right: content tabs
        tabs = QTabWidget()

        self._preview_table = DataTableView()
        tabs.addTab(self._preview_table, "미리보기")

        self._schema_edit = QTextEdit()
        self._schema_edit.setReadOnly(True)
        tabs.addTab(self._schema_edit, "스키마")

        tabs.addTab(self._build_query_tab(), "SQL 쿼리")
        tabs.addTab(self._build_export_tab(), "가져오기 / 내보내기")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _build_connection_group(self) -> QGroupBox:
        grp = QGroupBox("연결 설정")
        gl = QVBoxLayout(grp)

        # DB type selector
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("DB 유형:"))
        self._dbtype_combo = QComboBox()
        self._dbtype_combo.addItems(["SQLite", "MySQL", "PostgreSQL"])
        self._dbtype_combo.currentIndexChanged.connect(self._on_dbtype_changed)
        type_row.addWidget(self._dbtype_combo)
        type_row.addStretch()
        gl.addLayout(type_row)

        # SQLite form
        self._sqlite_group = QWidget()
        sq = QFormLayout(self._sqlite_group)
        self._sqlite_path = QLineEdit()
        self._sqlite_path.setPlaceholderText("database.db 파일 경로")
        browse_btn = QPushButton("찾기…")
        browse_btn.setMaximumWidth(60)
        browse_btn.clicked.connect(self._browse_sqlite)
        path_row = QHBoxLayout()
        path_row.addWidget(self._sqlite_path)
        path_row.addWidget(browse_btn)
        path_row.setContentsMargins(0, 0, 0, 0)
        pw = QWidget()
        pw.setLayout(path_row)
        sq.addRow("파일:", pw)
        gl.addWidget(self._sqlite_group)

        # External DB form
        self._ext_group = QWidget()
        self._ext_group.setVisible(False)
        ex = QFormLayout(self._ext_group)
        self._ext_host = QLineEdit("localhost")
        ex.addRow("호스트:", self._ext_host)
        self._ext_port = QSpinBox()
        self._ext_port.setRange(1, 65535)
        self._ext_port.setValue(3306)
        ex.addRow("포트:", self._ext_port)
        self._ext_db = QLineEdit()
        self._ext_db.setPlaceholderText("데이터베이스 이름")
        ex.addRow("DB 이름:", self._ext_db)
        self._ext_user = QLineEdit()
        ex.addRow("사용자:", self._ext_user)
        self._ext_pass = QLineEdit()
        self._ext_pass.setEchoMode(QLineEdit.EchoMode.Password)
        ex.addRow("비밀번호:", self._ext_pass)
        gl.addWidget(self._ext_group)

        # Buttons + status
        btn_row = QHBoxLayout()
        self._connect_btn = QPushButton("연결")
        self._connect_btn.clicked.connect(self._connect)
        self._disconnect_btn = QPushButton("연결 해제")
        self._disconnect_btn.setEnabled(False)
        self._disconnect_btn.clicked.connect(self._disconnect)
        self._test_btn = QPushButton("연결 테스트")
        self._test_btn.clicked.connect(self._test_connection)
        btn_row.addWidget(self._connect_btn)
        btn_row.addWidget(self._disconnect_btn)
        btn_row.addWidget(self._test_btn)
        btn_row.addStretch()
        gl.addLayout(btn_row)

        self._conn_status = QLabel("⚪ 연결 없음")
        gl.addWidget(self._conn_status)

        return grp

    def _build_query_tab(self) -> QWidget:
        w = QWidget()
        ql = QVBoxLayout(w)

        ql.addWidget(QLabel("SQL 쿼리 (SELECT만 권장):"))
        self._query_edit = QTextEdit()
        self._query_edit.setPlaceholderText("SELECT * FROM table_name LIMIT 100")
        self._query_edit.setMaximumHeight(120)
        ql.addWidget(self._query_edit)

        qbtn_row = QHBoxLayout()
        self._run_query_btn = QPushButton("쿼리 실행")
        self._run_query_btn.setEnabled(False)
        self._run_query_btn.clicked.connect(self._run_query)
        self._load_query_btn = QPushButton("결과를 AIViz에 로드")
        self._load_query_btn.setEnabled(False)
        self._load_query_btn.clicked.connect(self._load_query_result)
        qbtn_row.addWidget(self._run_query_btn)
        qbtn_row.addWidget(self._load_query_btn)
        qbtn_row.addStretch()
        ql.addLayout(qbtn_row)

        self._query_result_table = DataTableView()
        ql.addWidget(self._query_result_table)
        self._query_result: pd.DataFrame | None = None

        return w

    def _build_export_tab(self) -> QWidget:
        w = QWidget()
        el = QVBoxLayout(w)

        # Import section
        import_grp = QGroupBox("DB 테이블 → AIViz 로드")
        il = QVBoxLayout(import_grp)
        self._import_table_combo = QComboBox()
        self._import_table_combo.setEditable(True)
        self._import_table_combo.setPlaceholderText("테이블 이름 선택 또는 입력")
        il.addWidget(QLabel("테이블:"))
        il.addWidget(self._import_table_combo)
        self._import_limit = QSpinBox()
        self._import_limit.setRange(0, 10_000_000)
        self._import_limit.setValue(0)
        self._import_limit.setSpecialValueText("전체 로드")
        il.addWidget(QLabel("최대 행 수:"))
        il.addWidget(self._import_limit)
        self._import_btn = QPushButton("AIViz에 로드")
        self._import_btn.setEnabled(False)
        self._import_btn.clicked.connect(self._import_table)
        il.addWidget(self._import_btn)
        el.addWidget(import_grp)

        # Export section
        export_grp = QGroupBox("AIViz 데이터 → DB 저장")
        exl = QVBoxLayout(export_grp)
        self._export_table_name = QLineEdit()
        self._export_table_name.setPlaceholderText("저장할 테이블 이름")
        exl.addWidget(QLabel("테이블 이름:"))
        exl.addWidget(self._export_table_name)
        self._export_mode = QComboBox()
        self._export_mode.addItems(["replace (교체)", "append (추가)"])
        exl.addWidget(QLabel("저장 모드:"))
        exl.addWidget(self._export_mode)
        self._export_btn = QPushButton("DB에 저장")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_dataframe)
        exl.addWidget(self._export_btn)
        el.addWidget(export_grp)

        el.addStretch()
        return w

    # ─────────────────────────────────────────────────────────────────────
    # Connection management
    # ─────────────────────────────────────────────────────────────────────

    def _on_dbtype_changed(self, idx: int) -> None:
        is_sqlite = (idx == 0)
        self._sqlite_group.setVisible(is_sqlite)
        self._ext_group.setVisible(not is_sqlite)
        if idx == 1:
            self._ext_port.setValue(3306)
        elif idx == 2:
            self._ext_port.setValue(5432)

    def _browse_sqlite(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "SQLite 파일 선택", "", "SQLite DB (*.db *.sqlite *.sqlite3);;모든 파일 (*)"
        )
        if path:
            self._sqlite_path.setText(path)

    def _test_connection(self) -> None:
        """Test connection without persisting it."""
        from aiviz.db.sqlite_backend import SQLiteBackend
        from aiviz.db.sqlalchemy_backend import SQLAlchemyBackend, HAS_SQLALCHEMY

        idx = self._dbtype_combo.currentIndex()

        def do_test():
            if idx == 0:
                backend = SQLiteBackend(self._sqlite_path.text().strip() or ":memory:")
                return backend.test_connection()
            elif not HAS_SQLALCHEMY:
                return (
                    False,
                    "SQLAlchemy가 설치되어 있지 않습니다.\n"
                    "pip install sqlalchemy pymysql psycopg2-binary",
                )
            else:
                if idx == 1:
                    backend = SQLAlchemyBackend.for_mysql(
                        self._ext_host.text(), self._ext_port.value(),
                        self._ext_db.text(), self._ext_user.text(), self._ext_pass.text(),
                    )
                else:
                    backend = SQLAlchemyBackend.for_postgresql(
                        self._ext_host.text(), self._ext_port.value(),
                        self._ext_db.text(), self._ext_user.text(), self._ext_pass.text(),
                    )
                return backend.test_connection()

        worker = WorkerThread(do_test)
        worker.result_ready.connect(
            lambda res: QMessageBox.information(
                self, "연결 테스트", f"{'✅ ' if res[0] else '❌ '}{res[1]}"
            )
        )
        worker.error_occurred.connect(
            lambda msg: QMessageBox.warning(self, "오류", msg)
        )
        worker.start()
        self._test_worker = worker

    def _connect(self) -> None:
        idx = self._dbtype_combo.currentIndex()

        def do_connect():
            if idx == 0:
                path = self._sqlite_path.text().strip()
                if not path:
                    return (False, "SQLite 파일 경로를 입력하세요.")
                return self._service.connect_sqlite(path)
            else:
                from aiviz.db.connection import make_mysql_config, make_postgresql_config
                if idx == 1:
                    cfg = make_mysql_config(
                        self._ext_host.text(), self._ext_port.value(),
                        self._ext_db.text(), self._ext_user.text(), self._ext_pass.text(),
                    )
                else:
                    cfg = make_postgresql_config(
                        self._ext_host.text(), self._ext_port.value(),
                        self._ext_db.text(), self._ext_user.text(), self._ext_pass.text(),
                    )
                return self._service.connect_external(cfg)

        self._connect_btn.setEnabled(False)
        worker = WorkerThread(do_connect)
        worker.result_ready.connect(self._on_connect_done)
        worker.error_occurred.connect(self._on_connect_error)
        worker.start()
        self._worker = worker

    def _on_connect_done(self, result: tuple) -> None:
        ok, msg = result
        self._connect_btn.setEnabled(True)
        if ok:
            self._conn_status.setText(f"🟢 연결됨: {self._service.db_label}")
            self._disconnect_btn.setEnabled(True)
            self._refresh_btn.setEnabled(True)
            self._run_query_btn.setEnabled(True)
            self._import_btn.setEnabled(True)
            self._export_btn.setEnabled(True)
            self._ctrl.log_message.emit(f"[DB] {msg}")
            self._refresh_tables()
        else:
            self._conn_status.setText("🔴 연결 실패")
            self._ctrl.log_message.emit(f"[DB ERROR] {msg}")
            QMessageBox.warning(self, "DB 연결 오류", msg)

    def _on_connect_error(self, msg: str) -> None:
        self._connect_btn.setEnabled(True)
        self._conn_status.setText("🔴 연결 실패")
        self._ctrl.log_message.emit(f"[DB ERROR] {msg}")

    def _disconnect(self) -> None:
        self._service.disconnect()
        self._conn_status.setText("⚪ 연결 없음")
        self._disconnect_btn.setEnabled(False)
        self._refresh_btn.setEnabled(False)
        self._run_query_btn.setEnabled(False)
        self._load_query_btn.setEnabled(False)
        self._import_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._table_list.clear()
        self._import_table_combo.clear()
        self._ctrl.log_message.emit("[DB] 연결 해제됨")

    # ─────────────────────────────────────────────────────────────────────
    # Table management
    # ─────────────────────────────────────────────────────────────────────

    def _refresh_tables(self) -> None:
        tables, err = self._service.list_tables()
        if err:
            self._ctrl.log_message.emit(f"[DB ERROR] {err}")
            return
        self._table_list.clear()
        self._import_table_combo.clear()
        for t in tables:
            self._table_list.addItem(QListWidgetItem(t))
            self._import_table_combo.addItem(t)
        self._ctrl.log_message.emit(f"[DB] {len(tables)}개 테이블 발견")

    def _on_table_selected(self, item: QListWidgetItem) -> None:
        table_name = item.text()
        self._current_table = table_name
        self._preview_table_data(table_name)
        self._show_schema(table_name)

    def _preview_table_data(self, table_name: str) -> None:
        result = self._service.preview_table(table_name, n_rows=200)
        if result.ok and result.df is not None:
            self._preview_table.load(result.df)
        else:
            self._ctrl.log_message.emit(f"[DB ERROR] {result.error}")

    def _show_schema(self, table_name: str) -> None:
        info, err = self._service.get_table_info(table_name)
        if err:
            self._schema_edit.setPlainText(f"스키마 조회 오류: {err}")
            return
        lines = [
            f"테이블: {info['name']}",
            f"행 수: {info['row_count']:,}",
            f"컬럼 수: {len(info['columns'])}",
            "",
            "컬럼 목록:",
        ]
        for col in info["columns"]:
            lines.append(f"  {col['name']}  ({col['type']})")
        self._schema_edit.setPlainText("\n".join(lines))

    # ─────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────

    def _run_query(self) -> None:
        sql = self._query_edit.toPlainText().strip()
        if not sql:
            QMessageBox.warning(self, "오류", "SQL 쿼리를 입력하세요.")
            return

        self._run_query_btn.setEnabled(False)
        self._load_query_btn.setEnabled(False)

        def do_query():
            return self._service.execute_query(sql)

        worker = WorkerThread(do_query)
        worker.result_ready.connect(self._on_query_done)
        worker.error_occurred.connect(self._on_query_error)
        worker.start()
        self._query_worker = worker

    def _on_query_done(self, result) -> None:
        self._run_query_btn.setEnabled(True)
        if result.ok and result.df is not None:
            self._query_result = result.df
            self._query_result_table.load(result.df)
            self._load_query_btn.setEnabled(True)
            self._ctrl.log_message.emit(f"[DB] 쿼리 결과: {result.row_count:,}행")
        else:
            self._ctrl.log_message.emit(f"[DB ERROR] {result.error}")
            QMessageBox.warning(self, "쿼리 오류", result.error or "알 수 없는 오류")

    def _on_query_error(self, msg: str) -> None:
        self._run_query_btn.setEnabled(True)
        self._ctrl.log_message.emit(f"[DB ERROR] {msg}")

    def _load_query_result(self) -> None:
        if self._query_result is None:
            return
        self._inject_df(self._query_result, "db_query_result.csv")

    # ─────────────────────────────────────────────────────────────────────
    # Import / Export
    # ─────────────────────────────────────────────────────────────────────

    def _import_table(self) -> None:
        table_name = self._import_table_combo.currentText().strip()
        if not table_name:
            QMessageBox.warning(self, "오류", "가져올 테이블 이름을 입력하세요.")
            return

        limit = self._import_limit.value() or None

        def do_import():
            return self._service.read_table(table_name, limit=limit)

        self._import_btn.setEnabled(False)
        worker = WorkerThread(do_import)
        worker.result_ready.connect(
            lambda r: self._on_import_done(r, table_name)
        )
        worker.error_occurred.connect(
            lambda msg: (
                self._import_btn.setEnabled(True),
                self._ctrl.log_message.emit(f"[DB ERROR] {msg}"),
            )
        )
        worker.start()
        self._import_worker = worker

    def _on_import_done(self, result, table_name: str) -> None:
        self._import_btn.setEnabled(True)
        if result.ok and result.df is not None:
            self._inject_df(result.df, f"{table_name}.csv")
            self._ctrl.log_message.emit(
                f"[DB] '{table_name}' 로드 완료: {result.row_count:,}행"
            )
        else:
            self._ctrl.log_message.emit(f"[DB ERROR] {result.error}")
            QMessageBox.warning(self, "가져오기 오류", result.error or "알 수 없는 오류")

    def _export_dataframe(self) -> None:
        if self._df is None or self._df.empty:
            QMessageBox.warning(self, "오류", "AIViz에 로드된 데이터가 없습니다.")
            return

        table_name = self._export_table_name.text().strip()
        if not table_name:
            QMessageBox.warning(self, "오류", "저장할 테이블 이름을 입력하세요.")
            return

        mode_text = self._export_mode.currentText()
        if_exists = "replace" if "replace" in mode_text else "append"

        self._export_btn.setEnabled(False)

        def do_export():
            return self._service.save_dataframe(self._df, table_name, if_exists=if_exists)

        worker = WorkerThread(do_export)
        worker.result_ready.connect(self._on_export_done)
        worker.error_occurred.connect(
            lambda msg: (
                self._export_btn.setEnabled(True),
                self._ctrl.log_message.emit(f"[DB ERROR] {msg}"),
            )
        )
        worker.start()
        self._export_worker = worker

    def _on_export_done(self, result: tuple) -> None:
        self._export_btn.setEnabled(True)
        ok, msg = result
        if ok:
            self._ctrl.log_message.emit(f"[DB] {msg}")
            QMessageBox.information(self, "저장 완료", msg)
            self._refresh_tables()
        else:
            self._ctrl.log_message.emit(f"[DB ERROR] {msg}")
            QMessageBox.warning(self, "저장 오류", msg)

    # ─────────────────────────────────────────────────────────────────────
    # Helper: inject DataFrame into AppController
    # ─────────────────────────────────────────────────────────────────────

    def _inject_df(self, df: pd.DataFrame, name: str) -> None:
        """Load a DataFrame into the main AIViz controller as the active dataset."""
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        self._ctrl.load_file_async(csv_bytes, name)

    def _on_data_loaded(self, result) -> None:
        """Track current DataFrame for export."""
        if result.ok and result.df is not None:
            self._df = result.df
            self._export_btn.setEnabled(self._service.is_connected)
        else:
            self._df = None
