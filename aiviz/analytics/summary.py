"""
Descriptive statistics and summary analytics.

Provides rich summary objects that are used both for display and
as context injected into AI prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from aiviz.ingestion.schema import SchemaReport, inspect_schema


@dataclass
class DataSummary:
    schema: SchemaReport
    numeric_stats: pd.DataFrame   # describe() on numeric columns
    correlation: Optional[pd.DataFrame]
    top_categoricals: dict[str, pd.Series]   # col -> value_counts
    missing_heatmap_data: pd.DataFrame        # binary null mask (sampled)

    def to_text(self) -> str:
        """Compact text representation for LLM context."""
        lines = [
            f"Dataset: {self.schema.n_rows} rows, {self.schema.n_cols} columns.",
            f"Duplicate rows: {self.schema.duplicate_rows}",
            f"Memory: {self.schema.memory_kb:.1f} KB",
            "",
            "Schema:",
            self.schema.to_markdown_table(),
        ]
        if not self.numeric_stats.empty:
            lines += ["", "Numeric Statistics:", self.numeric_stats.to_string()]
        if self.correlation is not None and not self.correlation.empty:
            lines += ["", "Top Correlations (|r| > 0.5):"]
            corr = self.correlation
            for col in corr.columns:
                hi = corr[col][corr[col].abs() > 0.5].drop(col, errors="ignore")
                for other, val in hi.items():
                    lines.append(f"  {col} ↔ {other}: {val:.2f}")
        return "\n".join(lines)


def compute_summary(df: pd.DataFrame) -> DataSummary:
    """
    Compute full summary analytics for a loaded DataFrame.
    """
    schema = inspect_schema(df)

    numeric_cols = schema.numeric_cols()
    numeric_stats = df[numeric_cols].describe() if numeric_cols else pd.DataFrame()

    # Correlation (only when ≥2 numeric columns)
    correlation: Optional[pd.DataFrame] = None
    if len(numeric_cols) >= 2:
        try:
            correlation = df[numeric_cols].corr()
        except Exception:
            correlation = None

    # Top categoricals (value counts for low-cardinality columns)
    top_cats: dict[str, pd.Series] = {}
    for col in schema.categorical_cols():
        try:
            top_cats[col] = df[col].value_counts().head(10)
        except Exception:
            pass

    # Missing value mask (sampled to at most 200 rows for heatmap display)
    sample_size = min(200, len(df))
    missing_heatmap = df.sample(sample_size, random_state=42).isna().astype(int)

    return DataSummary(
        schema=schema,
        numeric_stats=numeric_stats,
        correlation=correlation,
        top_categoricals=top_cats,
        missing_heatmap_data=missing_heatmap,
    )
