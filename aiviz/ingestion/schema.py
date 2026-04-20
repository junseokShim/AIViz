"""
Schema inspection utilities.

Provides structured summaries of DataFrame schemas, missing values,
and type breakdowns – used in the data preview UI and AI context building.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    non_null: int
    null_count: int
    null_pct: float
    unique: int
    sample_values: list[Any]
    inferred_role: str  # "numeric" | "datetime" | "categorical" | "text" | "boolean"


@dataclass
class SchemaReport:
    n_rows: int
    n_cols: int
    memory_kb: float
    columns: list[ColumnInfo]
    duplicate_rows: int

    def numeric_cols(self) -> list[str]:
        return [c.name for c in self.columns if c.inferred_role == "numeric"]

    def datetime_cols(self) -> list[str]:
        return [c.name for c in self.columns if c.inferred_role == "datetime"]

    def categorical_cols(self) -> list[str]:
        return [c.name for c in self.columns if c.inferred_role in ("categorical", "boolean")]

    def to_markdown_table(self) -> str:
        lines = ["| Column | Type | Non-null | Null% | Unique | Role |",
                 "|--------|------|----------|-------|--------|------|"]
        for c in self.columns:
            lines.append(
                f"| {c.name} | {c.dtype} | {c.non_null} "
                f"| {c.null_pct:.1f}% | {c.unique} | {c.inferred_role} |"
            )
        return "\n".join(lines)


def inspect_schema(df: pd.DataFrame) -> SchemaReport:
    """
    Build a full SchemaReport for the given DataFrame.
    """
    columns = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        non_null = len(series) - null_count
        null_pct = 100.0 * null_count / max(len(series), 1)
        unique = int(series.nunique())
        sample = series.dropna().head(3).tolist()
        role = _infer_role(series)
        columns.append(ColumnInfo(
            name=col,
            dtype=str(series.dtype),
            non_null=non_null,
            null_count=null_count,
            null_pct=null_pct,
            unique=unique,
            sample_values=sample,
            inferred_role=role,
        ))

    return SchemaReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        memory_kb=df.memory_usage(deep=True).sum() / 1024,
        columns=columns,
        duplicate_rows=int(df.duplicated().sum()),
    )


def _infer_role(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # Categorical heuristic: object with low cardinality relative to length
    if pd.api.types.is_object_dtype(series):
        if series.nunique() / max(len(series), 1) < 0.3:
            return "categorical"
        return "text"
    return "categorical"
