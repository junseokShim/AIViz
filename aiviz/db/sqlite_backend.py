"""
SQLite backend using Python's built-in sqlite3 module.

No external dependencies required.

Thread safety:
  sqlite3 connections created with check_same_thread=False so they can be
  used from background threads (used inside WorkerThread).
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd


class SQLiteBackend:
    """
    Direct SQLite backend.

    Uses only Python's built-in sqlite3 module – no SQLAlchemy required.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open connection. Raises on failure."""
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

    def disconnect(self) -> None:
        """Close connection if open."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def is_connected(self) -> bool:
        return self._conn is not None

    def test_connection(self) -> tuple[bool, str]:
        """Test that the file can be opened and queried. Does not persist connection."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("SELECT 1")
            conn.close()
            return True, "SQLite 연결 성공"
        except Exception as exc:
            return False, f"SQLite 연결 오류: {exc}"

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def list_tables(self) -> list[str]:
        self._require_connected()
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_table_info(self, table_name: str) -> dict:
        """Return dict with name, row_count, columns list."""
        self._require_connected()
        safe_name = table_name.replace('"', '""')
        row_count = self._conn.execute(
            f'SELECT COUNT(*) FROM "{safe_name}"'
        ).fetchone()[0]
        pragma = self._conn.execute(
            f'PRAGMA table_info("{safe_name}")'
        ).fetchall()
        columns = [{"name": row[1], "type": row[2]} for row in pragma]
        return {"name": table_name, "row_count": row_count, "columns": columns}

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def read_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        self._require_connected()
        safe_name = table_name.replace('"', '""')
        query = f'SELECT * FROM "{safe_name}"'
        if limit is not None and limit > 0:
            query += f" LIMIT {int(limit)}"
        return pd.read_sql_query(query, self._conn)

    def execute_query(self, sql: str) -> pd.DataFrame:
        self._require_connected()
        return pd.read_sql_query(sql, self._conn)

    def save_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> int:
        """
        Save a DataFrame to a table.

        Args:
            df:         DataFrame to save.
            table_name: Destination table name.
            if_exists:  'replace', 'append', or 'fail'.

        Returns:
            Number of rows saved.
        """
        self._require_connected()
        df.to_sql(table_name, self._conn, if_exists=if_exists, index=False)
        self._conn.commit()
        return len(df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._conn:
            raise RuntimeError("DB에 연결되어 있지 않습니다. 먼저 연결하세요.")
