"""
Data ingestion layer.

Responsible for reading files into pandas DataFrames or PIL Images.
All format-specific logic is isolated here – callers receive a clean
DataLoadResult regardless of the source format.

Big-Data Strategy
-----------------
Files above LARGE_FILE_BYTES (50 MB) or rows above PREVIEW_ROW_LIMIT are
loaded in preview mode by default:
  - CSV:     nrows = PREVIEW_ROW_LIMIT
  - Parquet: all columns, but pyarrow row-group streaming (first N rows)
  - Excel:   full load (openpyxl does not support streaming; warn user)
  - JSON:    full load (in-memory JSON); warn if large

When preview mode is active, DataLoadResult.is_preview = True and
DataLoadResult.total_rows_estimate is set to the best available estimate.

Use load_file(..., preview=False) to force full loading regardless of size.
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


# ---------------------------------------------------------------------------
# Big-data thresholds
# ---------------------------------------------------------------------------

PREVIEW_ROW_LIMIT   = 100_000        # rows to load in preview mode
LARGE_FILE_BYTES    = 50 * 1024 * 1024   # 50 MB
WARN_ROW_THRESHOLD  = 500_000        # display warning even if fully loaded


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DataLoadResult:
    """Container returned by every load function."""
    df: Optional[pd.DataFrame] = None
    image: Optional[Image.Image] = None
    file_name: str = ""
    file_type: str = ""       # "tabular" | "image"
    raw_bytes: bytes = field(default_factory=bytes, repr=False)
    error: Optional[str] = None
    # Big-data metadata
    is_preview: bool = False           # True when only a subset was loaded
    total_rows_estimate: Optional[int] = None  # best row estimate for full file
    sampling_note: str = ""            # human-readable note about what was loaded

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(
    file_bytes: bytes,
    file_name: str,
    preview: bool = True,
) -> DataLoadResult:
    """
    Main entry point. Detect file type and dispatch to the right parser.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        file_name:  Original file name (used for extension detection).
        preview:    If True (default), apply row limits for large files.
                    Pass False to force full loading.

    Returns:
        DataLoadResult with either df or image populated.
    """
    suffix = Path(file_name).suffix.lower()

    if suffix in APP.supported_image:
        return _load_image(file_bytes, file_name)
    elif suffix in APP.supported_tabular:
        return _load_tabular(file_bytes, file_name, suffix, preview=preview)
    else:
        return DataLoadResult(
            file_name=file_name,
            error=(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: {APP.supported_tabular + APP.supported_image}"
            ),
        )


# ---------------------------------------------------------------------------
# Private parsers
# ---------------------------------------------------------------------------

def _load_tabular(
    file_bytes: bytes,
    file_name: str,
    suffix: str,
    preview: bool = True,
) -> DataLoadResult:
    file_size = len(file_bytes)
    is_large  = file_size > LARGE_FILE_BYTES
    buf = io.BytesIO(file_bytes)

    is_preview = False
    total_rows_estimate: Optional[int] = None
    sampling_note = ""

    try:
        if suffix == ".csv":
            df, is_preview, total_rows_estimate = _read_csv(
                buf, file_bytes, preview=preview and is_large
            )
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(buf)
            if is_large:
                sampling_note = (
                    f"Excel 파일 ({file_size // 1024 // 1024} MB): "
                    "전체 로드됨 (Excel은 스트리밍 미지원)"
                )
        elif suffix == ".json":
            df = _read_json(buf)
            if is_large:
                sampling_note = (
                    f"JSON 파일 ({file_size // 1024 // 1024} MB): "
                    "전체 로드됨"
                )
        elif suffix == ".parquet":
            df, is_preview, total_rows_estimate = _read_parquet(
                buf, preview=preview and is_large
            )
        else:
            return DataLoadResult(
                file_name=file_name,
                error=f"Unhandled tabular suffix {suffix}",
            )
    except Exception as exc:
        return DataLoadResult(file_name=file_name, error=f"Parse error: {exc}")

    df = _clean_dataframe(df)

    # Build sampling note
    if is_preview:
        est = f" (전체 추정: {total_rows_estimate:,})" if total_rows_estimate else ""
        sampling_note = (
            f"⚠ 미리보기 모드: {len(df):,}/{PREVIEW_ROW_LIMIT:,} 행{est} — "
            "전체 로드: 파일 선택 후 Shift+클릭"
        )
    elif len(df) > WARN_ROW_THRESHOLD:
        sampling_note = (
            f"ℹ {len(df):,} 행 전체 로드됨 — "
            "시각화는 자동 다운샘플링 적용됩니다"
        )

    return DataLoadResult(
        df=df,
        file_name=file_name,
        file_type="tabular",
        raw_bytes=file_bytes,
        is_preview=is_preview,
        total_rows_estimate=total_rows_estimate,
        sampling_note=sampling_note,
    )


def _read_csv(
    buf: io.BytesIO,
    raw_bytes: bytes,
    preview: bool = False,
) -> tuple[pd.DataFrame, bool, Optional[int]]:
    """Try common CSV encodings. Returns (df, is_preview, total_rows_estimate)."""
    nrows = PREVIEW_ROW_LIMIT if preview else None

    # Count total rows for estimate (cheap scan of newlines)
    total_estimate: Optional[int] = None
    if preview:
        try:
            total_estimate = raw_bytes.count(b"\n")
        except Exception:
            pass

    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            buf.seek(0)
            df = pd.read_csv(buf, encoding=enc, nrows=nrows, low_memory=True)
            is_preview = preview and (nrows is not None) and (len(df) >= nrows)
            return df, is_preview, total_estimate
        except UnicodeDecodeError:
            continue

    buf.seek(0)
    df = pd.read_csv(buf, encoding="utf-8", errors="replace", nrows=nrows, low_memory=True)
    is_preview = preview and (nrows is not None) and (len(df) >= nrows)
    return df, is_preview, total_estimate


def _read_json(buf: io.BytesIO) -> pd.DataFrame:
    """Handle both JSON arrays and JSON objects."""
    raw = json.loads(buf.read())
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    elif isinstance(raw, dict):
        return pd.DataFrame.from_dict(raw, orient="index").reset_index()
    raise ValueError("JSON root must be an array or object.")


def _read_parquet(
    buf: io.BytesIO,
    preview: bool = False,
) -> tuple[pd.DataFrame, bool, Optional[int]]:
    """Read Parquet. Returns (df, is_preview, total_rows_estimate)."""
    try:
        import pyarrow.parquet as pq
        buf.seek(0)
        pf = pq.ParquetFile(buf)
        total = pf.metadata.num_rows

        if preview and total > PREVIEW_ROW_LIMIT:
            # Read only enough row groups to cover PREVIEW_ROW_LIMIT
            collected: list[pd.DataFrame] = []
            rows_read = 0
            for rg in range(pf.metadata.num_row_groups):
                batch = pf.read_row_group(rg).to_pandas()
                collected.append(batch)
                rows_read += len(batch)
                if rows_read >= PREVIEW_ROW_LIMIT:
                    break
            df = pd.concat(collected, ignore_index=True).head(PREVIEW_ROW_LIMIT)
            return df, True, total
        else:
            buf.seek(0)
            df = pd.read_parquet(buf)
            return df, False, total
    except ImportError:
        # Fall back to pandas read_parquet (whole file)
        buf.seek(0)
        df = pd.read_parquet(buf)
        return df, False, None


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
