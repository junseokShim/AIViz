"""
AIViz Database integration package.

Provides unified access to:
  - SQLite  (always available via Python's built-in sqlite3)
  - MySQL   (via SQLAlchemy + pymysql, optional)
  - PostgreSQL (via SQLAlchemy + psycopg2, optional)

Architecture:
  db_service.DBService   – high-level UI-facing service
  sqlite_backend         – direct sqlite3 implementation
  sqlalchemy_backend     – SQLAlchemy multi-DB implementation
  connection             – ConnectionConfig helpers
  models                 – shared dataclasses
"""
from .models import ConnectionConfig, DBType, QueryResult, TableInfo
from .db_service import DBService
from .connection import make_sqlite_config, make_mysql_config, make_postgresql_config

__all__ = [
    "DBService",
    "ConnectionConfig", "DBType", "QueryResult", "TableInfo",
    "make_sqlite_config", "make_mysql_config", "make_postgresql_config",
]
