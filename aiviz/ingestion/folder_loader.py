"""
Folder-level bulk data loader.

Scans a folder for supported tabular files (CSV/Excel/JSON/Parquet),
loads each one safely, and provides combine or per-file modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from aiviz.ingestion.loader import load_file
from aiviz.utils.schema_utils import normalize_columns

logger = logging.getLogger("aiviz.folder_loader")

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


@dataclass
class FileLoadSummary:
    path: Path
    ok: bool
    rows: int = 0
    cols: int = 0
    columns: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class FolderLoadResult:
    folder: Path
    summaries: list[FileLoadSummary] = field(default_factory=list)
    combined_df: Optional[pd.DataFrame] = None
    per_file_dfs: dict[str, pd.DataFrame] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def total(self) -> int:
        return len(self.summaries)

    @property
    def loaded(self) -> int:
        return sum(1 for s in self.summaries if s.ok)

    @property
    def failed(self) -> int:
        return sum(1 for s in self.summaries if not s.ok)

    @property
    def ok(self) -> bool:
        return self.error is None and self.loaded > 0

    def summary_text(self) -> str:
        lines = [
            f"폴더: {self.folder}",
            f"전체 파일: {self.total}",
            f"로드 성공: {self.loaded}",
            f"로드 실패: {self.failed}",
        ]
        if self.combined_df is not None:
            r, c = self.combined_df.shape
            lines.append(f"결합 데이터셋: {r:,}행 × {c}열")
        for s in self.summaries:
            status = "✓" if s.ok else "✗"
            detail = f"{s.rows}행 × {s.cols}열" if s.ok else s.error
            lines.append(f"  {status} {s.path.name}: {detail}")
        return "\n".join(lines)


def load_folder(
    folder: str | Path,
    recursive: bool = False,
    combine: bool = True,
    union_columns: bool = True,
) -> FolderLoadResult:
    """
    Load all supported tabular files from a folder.

    Args:
        folder:        Path to the folder.
        recursive:     Whether to scan sub-folders.
        combine:       Whether to concatenate all DataFrames.
        union_columns: If True, use union of all columns (missing → NaN).
                       If False, only use common columns.

    Returns:
        FolderLoadResult with per-file DFs and optional combined DF.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return FolderLoadResult(
            folder=folder, error=f"폴더가 존재하지 않습니다: {folder}"
        )

    pattern = "**/*" if recursive else "*"
    candidates = [
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not candidates:
        return FolderLoadResult(
            folder=folder, error="지원되는 파일이 없습니다 (CSV/Excel/JSON/Parquet)."
        )

    result = FolderLoadResult(folder=folder)
    dfs: list[pd.DataFrame] = []

    for path in sorted(candidates):
        try:
            raw = path.read_bytes()
            load_result = load_file(raw, path.name)
            if not load_result.ok or load_result.df is None:
                err = load_result.error or "알 수 없는 오류"
                result.summaries.append(FileLoadSummary(path=path, ok=False, error=err))
                logger.warning("Failed to load %s: %s", path.name, err)
                continue

            df = normalize_columns(load_result.df)
            result.per_file_dfs[path.name] = df
            result.summaries.append(FileLoadSummary(
                path=path, ok=True,
                rows=len(df), cols=len(df.columns),
                columns=list(df.columns),
            ))
            dfs.append(df)
            logger.info("Loaded %s: %d rows × %d cols", path.name, len(df), len(df.columns))

        except Exception as exc:
            result.summaries.append(FileLoadSummary(path=path, ok=False, error=str(exc)))
            logger.exception("Unexpected error loading %s", path.name)

    if combine and dfs:
        try:
            result.combined_df = _safe_concat(dfs, union_columns=union_columns)
        except Exception as exc:
            logger.exception("Failed to combine DataFrames: %s", exc)
            # Still return per-file results even if combine fails

    return result


def _safe_concat(dfs: list[pd.DataFrame], union_columns: bool) -> pd.DataFrame:
    """
    Concatenate DataFrames with mismatched schemas.

    union_columns=True  → outer join (missing values become NaN)
    union_columns=False → inner join (only common columns kept)
    """
    join = "outer" if union_columns else "inner"
    combined = pd.concat(dfs, axis=0, ignore_index=True, join=join, sort=False)
    return combined
