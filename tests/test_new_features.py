"""
Tests for new features added in v0.3.0:
- schema_utils (safe column access)
- derived_column_service
- clustering_service
- dl_service
- signal_processing_service
- folder_loader
"""

from __future__ import annotations

import io
import math
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# schema_utils
# ---------------------------------------------------------------------------

class TestSchemaUtils:
    def _df(self):
        return pd.DataFrame({
            "actLoad": [1.0, 2.0, 3.0],
            " command_RPM ": [100, 200, 300],
            "Signal A": [0.1, 0.2, 0.3],
        })

    def test_resolve_exact(self):
        from aiviz.utils.schema_utils import resolve_column
        df = self._df()
        assert resolve_column(df, "actLoad") == "actLoad"

    def test_resolve_strip(self):
        from aiviz.utils.schema_utils import resolve_column
        df = self._df()
        # Column " command_RPM " has leading/trailing space
        result = resolve_column(df, "command_RPM")
        assert result == " command_RPM "

    def test_resolve_missing_returns_none(self):
        from aiviz.utils.schema_utils import resolve_column
        df = self._df()
        assert resolve_column(df, "nonexistent_col") is None

    def test_safe_col_returns_series(self):
        from aiviz.utils.schema_utils import safe_col
        df = self._df()
        s = safe_col(df, "actLoad")
        assert s is not None
        assert len(s) == 3

    def test_safe_col_missing_returns_none(self):
        from aiviz.utils.schema_utils import safe_col
        df = self._df()
        assert safe_col(df, "no_such_column") is None

    def test_normalize_columns(self):
        from aiviz.utils.schema_utils import normalize_columns
        df = self._df()
        norm = normalize_columns(df)
        assert "command_RPM" in norm.columns
        assert "actLoad" in norm.columns

    def test_assert_columns(self):
        from aiviz.utils.schema_utils import assert_columns
        df = self._df()
        missing = assert_columns(df, ["actLoad", "nonexistent"])
        assert "nonexistent" in missing
        assert "actLoad" not in missing


# ---------------------------------------------------------------------------
# derived_column_service
# ---------------------------------------------------------------------------

class TestDerivedColumn:
    def _df(self):
        return pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

    def test_basic_arithmetic(self):
        from aiviz.analytics.derived_column_service import create_derived_column
        df = self._df()
        result = create_derived_column(df, "a_plus_b", "a + b")
        assert result.ok
        np.testing.assert_array_almost_equal(result.series.values, [11, 22, 33, 44, 55])

    def test_division(self):
        from aiviz.analytics.derived_column_service import create_derived_column
        df = self._df()
        result = create_derived_column(df, "ratio", "b / a")
        assert result.ok
        assert result.series.iloc[0] == pytest.approx(10.0)

    def test_empty_name_fails(self):
        from aiviz.analytics.derived_column_service import create_derived_column
        df = self._df()
        result = create_derived_column(df, "", "a + b")
        assert not result.ok

    def test_forbidden_keyword_fails(self):
        from aiviz.analytics.derived_column_service import create_derived_column
        df = self._df()
        result = create_derived_column(df, "bad", "import os")
        assert not result.ok
        assert "허용되지 않는" in result.error

    def test_preview_shape(self):
        from aiviz.analytics.derived_column_service import create_derived_column
        df = self._df()
        result = create_derived_column(df, "c", "a * 2")
        assert result.ok
        assert result.preview is not None
        assert "c" in result.preview.columns

    def test_apply_derived_column(self):
        from aiviz.analytics.derived_column_service import create_derived_column, apply_derived_column
        df = self._df()
        result = create_derived_column(df, "doubled", "a * 2")
        new_df = apply_derived_column(df, result)
        assert "doubled" in new_df.columns
        assert len(new_df) == len(df)


# ---------------------------------------------------------------------------
# clustering_service
# ---------------------------------------------------------------------------

class TestClustering:
    def _df(self):
        np.random.seed(0)
        return pd.DataFrame({
            "x": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)]),
            "y": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)]),
        })

    def test_kmeans_basic(self):
        pytest.importorskip("sklearn")
        from aiviz.analytics.clustering_service import run_kmeans
        df = self._df()
        result = run_kmeans(df, ["x", "y"], n_clusters=2)
        assert result.ok
        assert result.n_clusters == 2
        assert len(result.labels) == len(df)
        assert sum(result.cluster_sizes.values()) == len(df)

    def test_kmeans_too_few_points(self):
        pytest.importorskip("sklearn")
        from aiviz.analytics.clustering_service import run_kmeans
        df = pd.DataFrame({"x": [1.0], "y": [1.0]})
        result = run_kmeans(df, ["x", "y"], n_clusters=5)
        assert not result.ok

    def test_dbscan_basic(self):
        pytest.importorskip("sklearn")
        from aiviz.analytics.clustering_service import run_dbscan
        df = self._df()
        result = run_dbscan(df, ["x", "y"])
        assert result.ok
        assert len(result.labels) == len(df)


# ---------------------------------------------------------------------------
# dl_service
# ---------------------------------------------------------------------------

class TestDLService:
    def _df(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.1
        return pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    def test_regression(self):
        pytest.importorskip("sklearn")
        from aiviz.analytics.dl_service import run_mlp
        df = self._df()
        result = run_mlp(df, "y", ["x1", "x2"], task="regression")
        assert result.ok
        assert "RMSE" in result.metrics
        assert result.n_train > 0
        assert result.n_test > 0

    def test_too_few_rows(self):
        pytest.importorskip("sklearn")
        from aiviz.analytics.dl_service import run_mlp
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
        result = run_mlp(df, "y", ["x"])
        assert not result.ok


# ---------------------------------------------------------------------------
# signal_processing_service
# ---------------------------------------------------------------------------

class TestACAnalysis:
    def test_dc_removal(self):
        from aiviz.analytics.signal_processing_service import analyze_ac
        s = pd.Series(np.sin(2 * np.pi * 0.1 * np.arange(100)) + 5.0)
        result = analyze_ac(s)
        assert result.ok
        assert abs(result.dc_offset - 5.0) < 0.1
        assert result.ac_rms > 0
        assert abs(result.ac_component.mean()) < 0.01

    def test_too_short(self):
        from aiviz.analytics.signal_processing_service import analyze_ac
        result = analyze_ac(pd.Series([1.0]))
        assert not result.ok

    def test_remove_dc(self):
        from aiviz.analytics.signal_processing_service import remove_dc
        s = pd.Series([3.0, 4.0, 5.0])
        ac = remove_dc(s)
        assert abs(ac.mean()) < 1e-10


# ---------------------------------------------------------------------------
# folder_loader
# ---------------------------------------------------------------------------

class TestFolderLoader:
    def test_nonexistent_folder(self):
        from aiviz.ingestion.folder_loader import load_folder
        result = load_folder("/nonexistent/path/xyz123")
        assert not result.ok
        assert result.error is not None

    def test_empty_folder(self, tmp_path):
        from aiviz.ingestion.folder_loader import load_folder
        result = load_folder(tmp_path)
        assert not result.ok

    def test_single_csv(self, tmp_path):
        from aiviz.ingestion.folder_loader import load_folder
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        (tmp_path / "test.csv").write_text(df.to_csv(index=False))
        result = load_folder(tmp_path)
        assert result.ok
        assert result.loaded == 1
        assert result.combined_df is not None
        assert list(result.combined_df.columns) == ["a", "b"]

    def test_multiple_csv_same_schema(self, tmp_path):
        from aiviz.ingestion.folder_loader import load_folder
        for i in range(3):
            df = pd.DataFrame({"x": [i, i+1], "y": [i*2, i*3]})
            (tmp_path / f"file_{i}.csv").write_text(df.to_csv(index=False))
        result = load_folder(tmp_path, combine=True)
        assert result.ok
        assert result.loaded == 3
        assert len(result.combined_df) == 6

    def test_malformed_file_skipped(self, tmp_path):
        from aiviz.ingestion.folder_loader import load_folder
        df = pd.DataFrame({"a": [1, 2]})
        (tmp_path / "good.csv").write_text(df.to_csv(index=False))
        (tmp_path / "bad.csv").write_bytes(b"\x00\xff\xfe bad binary data !!!!")
        result = load_folder(tmp_path)
        # Should load at least the good file without crashing
        assert result.loaded >= 1
        assert result.total == 2
