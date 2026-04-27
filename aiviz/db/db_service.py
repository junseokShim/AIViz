"""
DBService – high-level database service for the AIViz UI.

Provides a unified interface over SQLite and external DBs.
The UI layer (panel_db.py) talks only to this class.

Design principles:
  - All error paths return (False, "Korean error message") tuples
  - Never raises – all exceptions are caught and returned as error strings
  - State is encapsulated; UI only reads .is_connected and .db_label
  - DataFrames are the universal data exchange type
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .models import ConnectionConfig, DBType, QueryResult

logger = logging.getLogger("aiviz.db")


class DBService:
    """
    Unified DB service for AIViz.

    Supports:
      - SQLite  (always available)
      - MySQL   (requires sqlalchemy + pymysql)
      - PostgreSQL (requires sqlalchemy + psycopg2-binary)
    """

    def __init__(self):
        self._backend = None
        self._db_label: str = ""
        self._db_type: Optional[DBType] = None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._backend is not None and self._backend.is_connected()

    @property
    def db_label(self) -> str:
        """Short display string for the current connection."""
        return self._db_label if self.is_connected else "연결 없음"

    @property
    def db_type(self) -> Optional[DBType]:
        return self._db_type

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect_sqlite(self, db_path: str) -> tuple[bool, str]:
        """Open a SQLite database file. Creates it if it does not exist."""
        self.disconnect()
        try:
            from .sqlite_backend import SQLiteBackend
            backend = SQLiteBackend(db_path)
            ok, msg = backend.test_connection()
            if not ok:
                return False, msg
            backend.connect()
            self._backend = backend
            self._db_label = f"SQLite: {db_path}"
            self._db_type = DBType.SQLITE
            logger.info("SQLite connected: %s", db_path)
            return True, f"SQLite 연결 완료: {db_path}"
        except Exception as exc:
            logger.exception("SQLite connect error")
            return False, f"SQLite 연결 오류: {exc}"

    def connect_external(self, config: ConnectionConfig) -> tuple[bool, str]:
        """Connect to MySQL or PostgreSQL via SQLAlchemy."""
        self.disconnect()
        try:
            from .sqlalchemy_backend import SQLAlchemyBackend, HAS_SQLALCHEMY
        except ImportError:
            return False, "sqlalchemy_backend 모듈을 불러올 수 없습니다."

        if not HAS_SQLALCHEMY:
            return False, (
                "SQLAlchemy가 설치되어 있지 않습니다.\n"
                "pip install sqlalchemy pymysql psycopg2-binary"
            )

        try:
            if config.db_type == DBType.MYSQL:
                backend = SQLAlchemyBackend.for_mysql(
                    config.host, config.port, config.database,
                    config.username, config.password,
                )
            elif config.db_type == DBType.POSTGRESQL:
                backend = SQLAlchemyBackend.for_postgresql(
                    config.host, config.port, config.database,
                    config.username, config.password,
                )
            else:
                return False, f"지원하지 않는 DB 타입: {config.db_type.value}"

            ok, msg = backend.test_connection()
            if not ok:
                return False, msg

            backend.connect()
            self._backend = backend
            self._db_label = config.display_name
            self._db_type = config.db_type
            return True, f"{config.db_type.value.upper()} 연결 완료"
        except Exception as exc:
            logger.exception("External DB connect error")
            return False, f"DB 연결 오류: {exc}"

    def disconnect(self) -> None:
        """Disconnect from current DB."""
        if self._backend:
            try:
                self._backend.disconnect()
            except Exception:
                pass
            self._backend = None
        self._db_label = ""
        self._db_type = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def list_tables(self) -> tuple[list[str], Optional[str]]:
        if not self.is_connected:
            return [], "연결되어 있지 않습니다."
        try:
            return self._backend.list_tables(), None
        except Exception as exc:
            return [], f"테이블 목록 조회 오류: {exc}"

    def get_table_info(self, table_name: str) -> tuple[Optional[dict], Optional[str]]:
        if not self.is_connected:
            return None, "연결되어 있지 않습니다."
        try:
            return self._backend.get_table_info(table_name), None
        except Exception as exc:
            return None, f"테이블 정보 조회 오류: {exc}"

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------

    def read_table(
        self,
        table_name: str,
        limit: Optional[int] = None,
    ) -> QueryResult:
        if not self.is_connected:
            return QueryResult(error="연결되어 있지 않습니다.")
        try:
            df = self._backend.read_table(table_name, limit=limit)
            return QueryResult(df=df, columns=list(df.columns), row_count=len(df))
        except Exception as exc:
            return QueryResult(error=f"테이블 로드 오류: {exc}")

    def execute_query(self, sql: str) -> QueryResult:
        if not self.is_connected:
            return QueryResult(error="연결되어 있지 않습니다.")
        sql = sql.strip()
        if not sql:
            return QueryResult(error="SQL 쿼리가 비어 있습니다.")
        try:
            df = self._backend.execute_query(sql)
            return QueryResult(df=df, columns=list(df.columns), row_count=len(df))
        except Exception as exc:
            return QueryResult(error=f"쿼리 실행 오류: {exc}")

    def save_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> tuple[bool, str]:
        if df is None or df.empty:
            return False, "저장할 데이터가 없습니다."
        if not self.is_connected:
            return False, "연결되어 있지 않습니다."
        try:
            n = self._backend.save_dataframe(df, table_name, if_exists=if_exists)
            action = "교체" if if_exists == "replace" else "추가"
            return True, f"'{table_name}' 테이블에 {n:,}행 저장 완료 ({action})"
        except Exception as exc:
            return False, f"저장 오류: {exc}"

    def preview_table(self, table_name: str, n_rows: int = 100) -> QueryResult:
        return self.read_table(table_name, limit=n_rows)
