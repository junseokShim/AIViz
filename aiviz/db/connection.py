"""
Convenience factories for ConnectionConfig objects.
"""
from __future__ import annotations

from .models import ConnectionConfig, DBType


def make_sqlite_config(db_path: str) -> ConnectionConfig:
    """Create a SQLite connection config from a file path."""
    return ConnectionConfig(db_type=DBType.SQLITE, database=db_path)


def make_mysql_config(
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
) -> ConnectionConfig:
    """Create a MySQL connection config."""
    return ConnectionConfig(
        db_type=DBType.MYSQL,
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )


def make_postgresql_config(
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
) -> ConnectionConfig:
    """Create a PostgreSQL connection config."""
    return ConnectionConfig(
        db_type=DBType.POSTGRESQL,
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )
