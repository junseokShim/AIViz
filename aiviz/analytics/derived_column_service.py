"""
Derived column creation via a safe expression engine.

Uses pandas.eval() with a restricted namespace.
Supports: arithmetic, ratio, diff, rolling mean, abs, log, normalization.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("aiviz.derived_column")


@dataclass
class DerivedColumnResult:
    column_name: str
    expression: str
    series: Optional[pd.Series]
    preview: Optional[pd.DataFrame]  # first few rows with result
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.series is not None


def create_derived_column(
    df: pd.DataFrame,
    column_name: str,
    expression: str,
) -> DerivedColumnResult:
    """
    Evaluate `expression` against `df` and return a DerivedColumnResult.

    Supported operations:
    - Standard arithmetic: +, -, *, /, **, %
    - Column references by name: actLoad, signal_a, etc.
    - Functions: abs(), log(), sqrt(), diff(), rolling_mean(col, window)
    - Normalization: normalize(col)
    - np functions: np.log, np.sqrt, np.abs, etc.

    NOT supported (for safety):
    - import statements
    - exec/eval calls
    - file I/O

    Returns DerivedColumnResult; never raises.
    """
    column_name = column_name.strip()
    expression = expression.strip()

    if not column_name:
        return DerivedColumnResult(
            column_name=column_name, expression=expression,
            series=None, preview=None, error="컬럼 이름을 입력하세요."
        )
    if not expression:
        return DerivedColumnResult(
            column_name=column_name, expression=expression,
            series=None, preview=None, error="수식을 입력하세요."
        )

    # Safety check – block dangerous patterns
    _FORBIDDEN = ["import ", "exec(", "eval(", "open(", "__", "subprocess", "os."]
    for bad in _FORBIDDEN:
        if bad in expression:
            return DerivedColumnResult(
                column_name=column_name, expression=expression,
                series=None, preview=None,
                error=f"허용되지 않는 키워드: '{bad}'"
            )

    try:
        result_series = _eval_expression(df, expression)
    except Exception as exc:
        return DerivedColumnResult(
            column_name=column_name, expression=expression,
            series=None, preview=None,
            error=f"수식 오류: {exc}"
        )

    if result_series is None:
        return DerivedColumnResult(
            column_name=column_name, expression=expression,
            series=None, preview=None,
            error="수식이 유효한 결과를 반환하지 않았습니다."
        )

    # Build preview
    preview_df = df.head(5).copy()
    preview_df[column_name] = result_series.head(5)

    return DerivedColumnResult(
        column_name=column_name,
        expression=expression,
        series=result_series,
        preview=preview_df,
    )


def apply_derived_column(df: pd.DataFrame, result: DerivedColumnResult) -> pd.DataFrame:
    """Add the derived column to df and return the modified DataFrame."""
    if not result.ok:
        raise ValueError(result.error)
    df = df.copy()
    df[result.column_name] = result.series.values[:len(df)]
    return df


# ---------------------------------------------------------------------------
# Internal evaluation
# ---------------------------------------------------------------------------

def _eval_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    """
    Try multiple evaluation strategies in order:
    1. Custom helper functions (diff, rolling_mean, normalize)
    2. pd.eval() with local namespace
    3. Python eval with safe namespace
    """
    # Pre-process helper functions
    expression, local_ns = _preprocess_helpers(df, expression)

    # Try pd.eval first (fastest, safest)
    try:
        result = df.eval(expression, local_dict=local_ns)
        if isinstance(result, pd.Series):
            return result
        if isinstance(result, (int, float)):
            return pd.Series([result] * len(df), index=df.index)
    except Exception:
        pass

    # Fallback: Python eval with controlled namespace
    safe_ns = _build_safe_namespace(df)
    safe_ns.update(local_ns)

    result = eval(expression, {"__builtins__": {}}, safe_ns)  # noqa: S307

    if isinstance(result, pd.Series):
        return result
    if isinstance(result, np.ndarray):
        return pd.Series(result, index=df.index[:len(result)])
    if isinstance(result, (int, float)):
        return pd.Series([result] * len(df), index=df.index)

    raise ValueError(f"수식이 Series를 반환하지 않았습니다: {type(result)}")


def _preprocess_helpers(
    df: pd.DataFrame, expression: str
) -> tuple[str, dict]:
    """
    Replace helper calls with pre-computed Series references.
    Returns (modified_expression, local_namespace).
    """
    import re
    local_ns: dict = {}

    # rolling_mean(col, window) → __rm_{col}_{window}
    for m in re.finditer(r'rolling_mean\((\w+),\s*(\d+)\)', expression):
        col, window = m.group(1), int(m.group(2))
        key = f"__rm_{col}_{window}"
        if col in df.columns:
            local_ns[key] = df[col].rolling(window=window, min_periods=1).mean()
        expression = expression.replace(m.group(0), key)

    # normalize(col) → __norm_{col}
    for m in re.finditer(r'normalize\((\w+)\)', expression):
        col = m.group(1)
        key = f"__norm_{col}"
        if col in df.columns:
            s = df[col].astype(float)
            local_ns[key] = (s - s.mean()) / (s.std() or 1)
        expression = expression.replace(m.group(0), key)

    # diff(col) → __diff_{col}
    for m in re.finditer(r'diff\((\w+)\)', expression):
        col = m.group(1)
        key = f"__diff_{col}"
        if col in df.columns:
            local_ns[key] = df[col].diff().fillna(0)
        expression = expression.replace(m.group(0), key)

    return expression, local_ns


def _build_safe_namespace(df: pd.DataFrame) -> dict:
    """Build a safe evaluation namespace with DataFrame columns and math helpers."""
    ns: dict = {}

    # Add all columns
    for col in df.columns:
        safe_name = str(col).replace(" ", "_").replace("-", "_")
        ns[safe_name] = df[col]
        ns[col] = df[col]   # also as-is

    # Safe math
    ns.update({
        "abs": np.abs,
        "log": np.log,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "sin": np.sin,
        "cos": np.cos,
        "np": np,
        "pd": pd,
        "math": math,
        "round": round,
        "min": np.minimum,
        "max": np.maximum,
    })

    return ns
