"""
Schema utilities – safe, robust DataFrame column access.

Prevents KeyError crashes caused by:
- Columns that don't exist in the loaded file
- Leading/trailing whitespace in column names
- Case mismatches
- Schema differences across multi-file loads
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("aiviz.schema_utils")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with all column names stripped of whitespace.
    Column values are NOT modified.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def resolve_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """
    Try to find `name` in df.columns, with fallbacks:
    1. Exact match (after stripping whitespace from stored columns)
    2. Case-insensitive match
    3. Returns None if not found

    Does NOT raise; callers should check for None and warn the user.
    """
    if name is None:
        return None

    name_stripped = str(name).strip()

    # 1. Exact match
    cols_stripped = {str(c).strip(): c for c in df.columns}
    if name_stripped in cols_stripped:
        return cols_stripped[name_stripped]

    # 2. Case-insensitive
    name_lower = name_stripped.lower()
    for orig, stripped in ((c, str(c).strip()) for c in df.columns):
        if stripped.lower() == name_lower:
            return orig

    logger.warning("Column %r not found in DataFrame (columns: %s)", name, list(df.columns))
    return None


def safe_col(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """
    Safely retrieve a Series by column name.
    Returns None (instead of raising KeyError) if the column doesn't exist.
    Logs a warning so the user can investigate.
    """
    resolved = resolve_column(df, name)
    if resolved is None:
        return None
    return df[resolved]


def assert_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """
    Return a list of required column names that are MISSING from df.
    Empty list means all columns are present.
    """
    missing = []
    for col in required:
        if resolve_column(df, col) is None:
            missing.append(col)
    return missing


def available_numeric(df: pd.DataFrame) -> list[str]:
    """Return numeric column names (whitespace-stripped)."""
    import numpy as np
    return df.select_dtypes(include=[np.number]).columns.tolist()
