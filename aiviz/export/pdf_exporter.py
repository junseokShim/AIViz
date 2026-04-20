"""
PDF report exporter using fpdf2.

Generates a structured PDF with:
- Header and metadata
- Dataset summary metrics
- Schema table
- Embedded chart images (PNG)
- AI-generated text sections

To add a new section to the PDF report:
  1. Add a method to PDFExporter following the existing pattern.
  2. Call it from the export panel.

Requires: pip install fpdf2
"""

from __future__ import annotations

import datetime
import io
import tempfile
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from config import APP

try:
    from fpdf import FPDF, XPos, YPos
    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False


def is_available() -> bool:
    return _FPDF_AVAILABLE


class PDFExporter:
    """
    Build and save a structured PDF analytics report.

    Usage:
        exp = PDFExporter(file_name="data.csv", df=df)
        exp.add_dataset_overview()
        exp.add_chart_image(png_bytes, title="FFT")
        exp.add_text_section("AI Insight", text)
        exp.save("/path/to/report.pdf")
    """

    def __init__(self, file_name: str = "", df: Optional[pd.DataFrame] = None):
        if not _FPDF_AVAILABLE:
            raise ImportError("fpdf2 is required: pip install fpdf2")

        self._file_name = file_name
        self._df = df
        self._pdf = FPDF()
        self._pdf.set_auto_page_break(auto=True, margin=15)
        self._pdf.add_page()
        self._pdf.set_margins(left=15, top=15, right=15)

        self._setup_fonts()
        self._add_header()

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def add_dataset_overview(self) -> "PDFExporter":
        if self._df is None:
            return self
        df = self._df
        self._section_title("Dataset Overview")
        rows = [
            ("File", self._file_name),
            ("Rows", f"{len(df):,}"),
            ("Columns", str(len(df.columns))),
            ("Missing values", str(int(df.isna().sum().sum()))),
            ("Duplicate rows", str(int(df.duplicated().sum()))),
            ("Numeric columns", str(len(df.select_dtypes(include="number").columns))),
        ]
        self._key_value_table(rows)
        return self

    def add_schema_table(self) -> "PDFExporter":
        if self._df is None:
            return self
        from aiviz.ingestion.schema import inspect_schema
        schema = inspect_schema(self._df)
        self._section_title("Schema")
        headers = ["Column", "Type", "Role", "Non-null", "Null %", "Unique"]
        data = [
            [c.name, c.dtype, c.inferred_role,
             str(c.non_null), f"{c.null_pct:.1f}%", str(c.unique)]
            for c in schema.columns
        ]
        self._table(headers, data)
        return self

    def add_stats_table(self) -> "PDFExporter":
        if self._df is None:
            return self
        num = self._df.select_dtypes(include="number")
        if num.empty:
            return self
        stats = num.describe().round(4)
        self._section_title("Descriptive Statistics")
        headers = ["stat"] + stats.columns.tolist()
        data = [
            [str(idx)] + [f"{stats.loc[idx, c]:.4g}" for c in stats.columns]
            for idx in stats.index
        ]
        self._table(headers, data)
        return self

    def add_chart_image(self, png_bytes: bytes, title: str = "Chart") -> "PDFExporter":
        """Embed a chart PNG into the PDF."""
        self._section_title(title)
        # Save to a temporary file (fpdf2 requires a file path)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            tmp.write(png_bytes)
            tmp.flush()
            tmp.close()
            # Fit image to page width
            page_w = self._pdf.w - 30  # margins
            self._pdf.image(tmp.name, w=page_w)
            self._pdf.ln(4)
        finally:
            os.unlink(tmp.name)
        return self

    def add_text_section(self, title: str, content: str) -> "PDFExporter":
        self._section_title(title)
        self._pdf.set_font("Helvetica", size=10)
        self._pdf.set_text_color(60, 60, 60)
        # Sanitise non-Latin chars to avoid fpdf encoding errors
        safe = content.encode("latin-1", errors="replace").decode("latin-1")
        self._pdf.multi_cell(0, 5, safe)
        self._pdf.ln(4)
        return self

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._pdf.output(path)

    def get_bytes(self) -> bytes:
        return self._pdf.output()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_fonts(self) -> None:
        # fpdf2 uses Helvetica by default; no extra font files needed
        pass

    def _add_header(self) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        pdf = self._pdf

        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(30, 90, 180)
        pdf.cell(0, 10, f"{APP.name} Analytics Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"File: {self._file_name}   |   Generated: {ts}   |   {APP.name} v{APP.version}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)
        pdf.set_draw_color(89, 139, 250)
        pdf.set_line_width(0.5)
        pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
        pdf.ln(6)

    def _section_title(self, title: str) -> None:
        pdf = self._pdf
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 90, 180)
        pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(200, 200, 220)
        pdf.set_line_width(0.3)
        pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
        pdf.ln(3)

    def _key_value_table(self, rows: list[tuple]) -> None:
        pdf = self._pdf
        for key, val in rows:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(60, 80, 140)
            pdf.cell(50, 6, key + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", size=10)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 6, str(val), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    def _table(self, headers: list[str], data: list[list]) -> None:
        pdf = self._pdf
        col_count = len(headers)
        available = pdf.w - 30
        col_w = min(available / col_count, 35)

        # Header row
        pdf.set_fill_color(50, 80, 160)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        for h in headers:
            text = str(h)[:14]
            pdf.cell(col_w, 6, text, border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(6)

        # Data rows
        pdf.set_font("Helvetica", size=8)
        for i, row in enumerate(data):
            fill = i % 2 == 0
            pdf.set_fill_color(240, 244, 255) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(30, 30, 30)
            for cell in row:
                text = str(cell)[:14]
                pdf.cell(col_w, 5, text, border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln(5)
        pdf.ln(4)
