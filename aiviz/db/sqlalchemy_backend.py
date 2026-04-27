"""
SQLAlchemy-based multi-DB backend.

Supports MySQL (via pymysql) and PostgreSQL (via psycopg2).
Also works with SQLite via SQLAlchemy for consistency.

Optional dependencies:
  pip install sqlalchemy pymysql psycopg2-binary

Graceful fallback:
  If SQLAlchemy is not installed, all operations return errors without raising.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    from sqlalchemy import create_engine, text, inspect as sa_inspect
    from sqlalchemy.engine import Engine
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False


_SA_MISSING = (
    "SQLAlchemy가 설치되어 있지 않습니다.\n"
    "다음 명령으로 설치하세요:\n"
    "  pip install sqlalchemy pymysql psycopg2-binary"
)


class SQLAlchemyBackend:
    """
    Multi-DB backend using SQLAlchemy.

    Create instances via the class methods:
        SQLAlchemyBackend.for_mysql(...)
        SQLAlchemyBackend.for_postgresql(...)
    """

    def __init__(self, connection_url: str):
        if not HAS_SQLALCHEMY:
            raise ImportError(_SA_MISSING)
        self._url = connection_url
        self._engine: Optional[Engine] = None
        self._conn = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def for_mysql(
        cls,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
    ) -> "SQLAlchemyBackend":
        url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        return cls(url)

    @classmethod
    def for_postgresql(
        cls,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
    ) -> "SQLAlchemyBackend":
        url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
        return cls(url)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._engine = create_engine(self._url, pool_pre_ping=True)
        self._conn = self._engine.connect()

    def disconnect(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._engine:
            try:
                self._engine.dispose()
            except Exception:
                pass
            self._engine = None

    def is_connected(self) -> bool:
        return self._conn is not None

    def test_connection(self) -> tuple[bool, str]:
        if not HAS_SQLALCHEMY:
            return False, _SA_MISSING
        try:
            eng = create_engine(self._url, pool_pre_ping=True)
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            eng.dispose()
            return True, "외부 DB 연결 성공"
        except Exception as exc:
            return False, f"DB 연결 오류: {exc}"

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def list_tables(self) -> list[str]:
        self._require_connected()
        inspector = sa_inspect(self._engine)
        return inspector.get_table_names()

    def get_table_info(self, table_name: str) -> dict:
        self._require_connected()
        inspector = sa_inspect(self._engine)
        cols = inspector.get_columns(table_name)
        columns = [{"name": c["name"], "type": str(c["type"])} for c in cols]
        count = self._conn.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        ).scalar()
        return {"name": table_name, "row_count": int(count or 0), "columns": columns}

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def read_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        self._require_connected()
        query = f"SELECT * FROM {table_name}"
        if limit is not None and limit > 0:
            query += f" LIMIT {int(limit)}"
        return pd.read_sql(query, self._engine)

    def execute_query(self, sql: str) -> pd.DataFrame:
        self._require_connected()
        return pd.read_sql(sql, self._engine)

    def save_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> int:
        self._require_connected()
        df.to_sql(table_name, self._engine, if_exists=if_exists, index=False)
        return len(df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._conn:
            raise RuntimeError("DB에 연결되어 있지 않습니다.")
