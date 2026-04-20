"""
Data ingestion layer.

Responsible for reading files into pandas DataFrames or PIL Images.
All format-specific logic is isolated here – callers receive a clean
DataLoadResult regardless of the source format.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from config import APP


@dataclass
class DataLoadResult:
    """Container returned by every load function."""
    df: Optional[pd.DataFrame] = None
    image: Optional[Image.Image] = None
    file_name: str = ""
    file_type: str = ""      # "tabular" | "image"
    raw_bytes: bytes = field(default_factory=bytes, repr=False)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def load_file(file_bytes: bytes, file_name: str) -> DataLoadResult:
    """
    Main entry point. Detect file type and dispatch to the right parser.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        file_name:  Original file name (used for extension detection).

    Returns:
        DataLoadResult with either df or image populated.
    """
    suffix = Path(file_name).suffix.lower()

    if suffix in APP.supported_image:
        return _load_image(file_bytes, file_name)
    elif suffix in APP.supported_tabular:
        return _load_tabular(file_bytes, file_name, suffix)
    else:
        return DataLoadResult(
            file_name=file_name,
            error=f"Unsupported file type: '{suffix}'. "
                  f"Supported: {APP.supported_tabular + APP.supported_image}"
        )


# ---------------------------------------------------------------------------
# Private parsers
# ---------------------------------------------------------------------------

def _load_tabular(file_bytes: bytes, file_name: str, suffix: str) -> DataLoadResult:
    buf = io.BytesIO(file_bytes)
    try:
        if suffix == ".csv":
            df = _read_csv(buf)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(buf)
        elif suffix == ".json":
            df = _read_json(buf)
        elif suffix == ".parquet":
            df = pd.read_parquet(buf)
        else:
            return DataLoadResult(file_name=file_name, error=f"Unhandled tabular suffix {suffix}")
    except Exception as exc:
        return DataLoadResult(file_name=file_name, error=f"Parse error: {exc}")

    df = _clean_dataframe(df)
    return DataLoadResult(
        df=df,
        file_name=file_name,
        file_type="tabular",
        raw_bytes=file_bytes,
    )


def _read_csv(buf: io.BytesIO) -> pd.DataFrame:
    """Try common CSV encodings."""
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            buf.seek(0)
            return pd.read_csv(buf, encoding=enc)
        except UnicodeDecodeError:
            continue
    buf.seek(0)
    return pd.read_csv(buf, encoding="utf-8", errors="replace")


def _read_json(buf: io.BytesIO) -> pd.DataFrame:
    """Handle both JSON arrays and JSON objects."""
    raw = json.loads(buf.read())
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    elif isinstance(raw, dict):
        # Try orient='records', 'index', or wrap in list
        return pd.DataFrame.from_dict(raw, orient="index").reset_index()
    raise ValueError("JSON root must be an array or object.")


def _load_image(file_bytes: bytes, file_name: str) -> DataLoadResult:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()  # force decode
    except Exception as exc:
        return DataLoadResult(file_name=file_name, error=f"Image decode error: {exc}")

    return DataLoadResult(
        image=img,
        file_name=file_name,
        file_type="image",
        raw_bytes=file_bytes,
    )


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-load cleanup:
    - strip whitespace from string column names
    - attempt datetime parsing for likely time columns
    - convert object columns that look numeric
    """
    df.columns = [str(c).strip() for c in df.columns]

    # Attempt auto datetime parsing
    time_hints = [c for c in df.columns if any(
        k in c.lower() for k in ("date", "time", "timestamp", "ts", "datetime")
    )]
    for col in time_hints:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass

    # Convert apparent numeric object/string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            converted = pd.to_numeric(df[col], errors="raise")
            df[col] = converted
        except Exception:
            pass

    return df
