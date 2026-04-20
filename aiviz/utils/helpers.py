"""
Shared utility functions used across AIViz modules.
"""

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def file_hash(data: bytes) -> str:
    """Return a short SHA-256 hex digest for cache-keying uploaded files."""
    return hashlib.sha256(data).hexdigest()[:16]


def human_bytes(n: int) -> str:
    """Convert byte count to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric column names in a DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return list of datetime-like column names."""
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


def infer_time_column(df: pd.DataFrame) -> str | None:
    """
    Best-effort heuristic to find a time/index column.
    Checks for columns named 'time', 'date', 'timestamp', etc.
    Returns the column name or None.
    """
    candidates = [c for c in df.columns if any(
        k in c.lower() for k in ("time", "date", "timestamp", "ts", "datetime")
    )]
    if candidates:
        return candidates[0]
    dt_cols = datetime_columns(df)
    if dt_cols:
        return dt_cols[0]
    return None


def safe_sample(df: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """Return at most n rows without shuffling (preserves order)."""
    if len(df) <= n:
        return df
    step = max(1, len(df) // n)
    return df.iloc[::step].head(n)


def df_to_context_string(df: pd.DataFrame, max_rows: int = 10) -> str:
    """
    Serialise a DataFrame into a compact text block suitable for
    inserting into an LLM prompt.
    """
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    buf.write(f"Columns: {', '.join(df.columns.tolist())}\n\n")
    buf.write("Sample (first rows):\n")
    buf.write(df.head(max_rows).to_string(index=False))
    buf.write("\n\nDescriptive statistics:\n")
    buf.write(df.describe(include="all").to_string())
    return buf.getvalue()


def truncate_str(s: str, max_len: int = 2000) -> str:
    """Truncate a string to avoid exceeding LLM context limits."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"\n... [truncated at {max_len} chars]"
