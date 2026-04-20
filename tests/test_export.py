"""Tests for the export layer (HTML and PDF)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile
import numpy as np
import pandas as pd
import pytest

from aiviz.export.html_exporter import HTMLExporter
from aiviz.export.pdf_exporter import PDFExporter, is_available


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "a": np.random.randn(30),
        "b": np.random.randn(30),
        "cat": np.random.choice(["x", "y", "z"], 30),
    })


class TestHTMLExporter:
    def test_build_returns_html(self, sample_df):
        exp = HTMLExporter(file_name="test.csv", df=sample_df)
        html = exp.build()
        assert "<!DOCTYPE html>" in html
        assert "AIViz" in html

    def test_dataset_overview_section(self, sample_df):
        exp = HTMLExporter(df=sample_df)
        exp.add_dataset_overview()
        html = exp.build()
        assert "30" in html  # row count

    def test_schema_table_section(self, sample_df):
        exp = HTMLExporter(df=sample_df)
        exp.add_schema_table()
        html = exp.build()
        assert "a" in html
        assert "b" in html

    def test_stats_table(self, sample_df):
        exp = HTMLExporter(df=sample_df)
        exp.add_stats_table()
        html = exp.build()
        assert "mean" in html or "count" in html

    def test_text_section(self, sample_df):
        exp = HTMLExporter(df=sample_df)
        exp.add_text_section("My Title", "Hello world content")
        html = exp.build()
        assert "My Title" in html
        assert "Hello world content" in html

    def test_chart_image_embedding(self, sample_df):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        png = buf.read()

        exp = HTMLExporter(df=sample_df)
        exp.add_chart_image(png, title="Test Chart")
        html = exp.build()
        assert "data:image/png;base64," in html
        assert "Test Chart" in html

    def test_save_to_file(self, sample_df, tmp_path):
        exp = HTMLExporter(file_name="t.csv", df=sample_df)
        exp.add_dataset_overview()
        out = str(tmp_path / "report.html")
        exp.save(out)
        assert os.path.exists(out)
        content = open(out).read()
        assert "<!DOCTYPE html>" in content


@pytest.mark.skipif(not is_available(), reason="fpdf2 not installed")
class TestPDFExporter:
    def test_build_and_save(self, sample_df, tmp_path):
        exp = PDFExporter(file_name="test.csv", df=sample_df)
        exp.add_dataset_overview()
        out = str(tmp_path / "report.pdf")
        exp.save(out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000  # non-trivial file

    def test_schema_table(self, sample_df, tmp_path):
        exp = PDFExporter(file_name="test.csv", df=sample_df)
        exp.add_schema_table()
        out = str(tmp_path / "schema.pdf")
        exp.save(out)
        assert os.path.exists(out)

    def test_chart_image(self, sample_df, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt, io
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        png = buf.read()

        exp = PDFExporter(file_name="test.csv", df=sample_df)
        exp.add_chart_image(png, "Test")
        out = str(tmp_path / "chart.pdf")
        exp.save(out)
        assert os.path.getsize(out) > 1000
