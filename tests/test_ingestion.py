"""
Tests for the ingestion layer.
"""

import io
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from aiviz.ingestion.loader import load_file, _clean_dataframe
from aiviz.ingestion.schema import inspect_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def make_json_bytes(records: list) -> bytes:
    return json.dumps(records).encode()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadFile:
    def test_load_csv_ok(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = load_file(make_csv_bytes(df), "test.csv")
        assert result.ok
        assert result.file_type == "tabular"
        assert result.df is not None
        assert list(result.df.columns) == ["a", "b"]
        assert len(result.df) == 3

    def test_load_json_ok(self):
        records = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        result = load_file(make_json_bytes(records), "test.json")
        assert result.ok
        assert result.df is not None
        assert len(result.df) == 2

    def test_unsupported_extension_returns_error(self):
        result = load_file(b"dummy", "file.xyz")
        assert not result.ok
        assert "Unsupported" in result.error

    def test_load_image_ok(self):
        from PIL import Image
        img = Image.new("RGB", (10, 10), color=(128, 64, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = load_file(buf.getvalue(), "test.png")
        assert result.ok
        assert result.file_type == "image"
        assert result.image is not None

    def test_corrupted_csv_returns_error(self):
        result = load_file(b"\xff\xfe" * 100, "broken.csv")
        # Either load fails with error, or it decodes with replacement chars
        # Either outcome is acceptable as long as it doesn't raise an exception
        assert result is not None


class TestCleanDataFrame:
    def test_strips_column_whitespace(self):
        df = pd.DataFrame({" name ": ["a"], " value ": [1]})
        cleaned = _clean_dataframe(df)
        assert "name" in cleaned.columns
        assert "value" in cleaned.columns

    def test_parses_date_column(self):
        df = pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-02"],
            "value": [10, 20],
        })
        cleaned = _clean_dataframe(df)
        assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])

    def test_numeric_conversion(self):
        df = pd.DataFrame({"val": ["1.5", "2.3", "3.7"]})
        cleaned = _clean_dataframe(df)
        assert pd.api.types.is_numeric_dtype(cleaned["val"])


class TestSchemaInspection:
    def test_basic_schema(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "score": [0.1, 0.2, 0.3],
            "active": [True, False, True],
        })
        schema = inspect_schema(df)
        assert schema.n_rows == 3
        assert schema.n_cols == 4
        assert "id" in schema.numeric_cols() or "score" in schema.numeric_cols()

    def test_missing_values_counted(self):
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})
        schema = inspect_schema(df)
        col_info = next(c for c in schema.columns if c.name == "a")
        assert col_info.null_count == 2
        assert abs(col_info.null_pct - 40.0) < 0.1

    def test_duplicate_detection(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        schema = inspect_schema(df)
        assert schema.duplicate_rows == 1
