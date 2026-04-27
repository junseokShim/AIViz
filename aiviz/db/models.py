"""
Data models for the AIViz DB layer.

These dataclasses are shared between all backends and the service layer.
No PyQt or analytics dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DBType(Enum):
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


@dataclass
class ConnectionConfig:
    """Configuration for a single DB connection."""
    db_type: DBType
    database: str               # SQLite: file path; others: database name
    host: str = "localhost"
    port: int = 3306
    username: str = ""
    password: str = ""

    @property
    def display_name(self) -> str:
        if self.db_type == DBType.SQLITE:
            return f"SQLite: {self.database}"
        return f"{self.db_type.value}://{self.username}@{self.host}:{self.port}/{self.database}"

    @property
    def default_port(self) -> int:
        return {
            DBType.SQLITE: 0,
            DBType.MYSQL: 3306,
            DBType.POSTGRESQL: 5432,
        }.get(self.db_type, 3306)


@dataclass
class ColumnInfo:
    name: str
    dtype: str


@dataclass
class TableInfo:
    """Metadata for a single DB table."""
    name: str
    row_count: int
    columns: list[ColumnInfo] = field(default_factory=list)

    def to_display_dict(self) -> dict:
        return {
            "테이블": self.name,
            "행 수": self.row_count,
            "컬럼 수": len(self.columns),
        }


@dataclass
class QueryResult:
    """Result of a DB query or table read operation."""
    df: Any = None                       # pandas DataFrame or None
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None
