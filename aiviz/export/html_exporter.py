"""
HTML report exporter.

Generates a self-contained HTML file with:
- Dataset summary table
- Inline base64-encoded chart images
- AI-generated textual summary
- Basic CSS styling (dark-ish, professional)

To add a new section to the HTML report:
  1. Add a `_section_<name>(...)` method returning HTML string.
  2. Call it from `build_report()` in the appropriate order.
"""

from __future__ import annotations

import base64
import datetime
import io
from pathlib import Path
from typing import Optional

import pandas as pd

from config import APP


# ---------------------------------------------------------------------------
# HTML template pieces
# ---------------------------------------------------------------------------

_CSS = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 24px;
    line-height: 1.6;
}
h1 { color: #89b4fa; font-size: 26px; margin-bottom: 4px; }
h2 { color: #89b4fa; font-size: 18px; margin: 24px 0 8px; border-bottom: 1px solid #313244; padding-bottom: 4px; }
h3 { color: #b4befe; font-size: 14px; margin: 16px 0 6px; }
.meta { color: #a6adc8; font-size: 12px; margin-bottom: 20px; }
.card {
    background: #16213e;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 8px;
}
th {
    background: #313244;
    color: #89b4fa;
    padding: 6px 10px;
    text-align: left;
    font-weight: bold;
}
td {
    padding: 5px 10px;
    border-bottom: 1px solid #313244;
    color: #cdd6f4;
}
tr:nth-child(even) td { background: #1e1e2e; }
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px;
    margin-top: 8px;
}
.metric {
    background: #313244;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: center;
}
.metric .value { font-size: 22px; font-weight: bold; color: #89b4fa; }
.metric .label { font-size: 11px; color: #a6adc8; }
img.chart { width: 100%; border-radius: 6px; margin-top: 10px; }
pre {
    background: #0f0e17;
    color: #cdd6f4;
    padding: 12px;
    border-radius: 6px;
    font-size: 11px;
    overflow-x: auto;
    white-space: pre-wrap;
}
.tag {
    display: inline-block;
    background: #45475a;
    color: #cdd6f4;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    margin: 2px;
}
footer { margin-top: 40px; color: #585b70; font-size: 11px; text-align: center; }
</style>
"""


class HTMLExporter:
    """
    Builds a self-contained HTML analytics report.

    Usage:
        exp = HTMLExporter(file_name="my_data.csv", df=df)
        exp.add_summary(summary_text)
        exp.add_chart_image(png_bytes, title="FFT Analysis")
        exp.add_ai_insight(ai_text)
        html = exp.build()
        exp.save("/path/to/report.html")
    """

    def __init__(self, file_name: str = "", df: Optional[pd.DataFrame] = None):
        self._file_name = file_name
        self._df = df
        self._sections: list[str] = []
        self._ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ------------------------------------------------------------------
    # Section builders (call these before build())
    # ------------------------------------------------------------------

    def add_dataset_overview(self) -> "HTMLExporter":
        """Add automatic dataset metrics from the loaded DataFrame."""
        if self._df is None:
            return self
        df = self._df
        rows, cols = df.shape
        missing = int(df.isna().sum().sum())
        dup = int(df.duplicated().sum())
        num_cols = df.select_dtypes(include="number").columns.tolist()

        metrics_html = ""
        for label, val in [
            ("Rows", f"{rows:,}"),
            ("Columns", cols),
            ("Missing", missing),
            ("Duplicates", dup),
            ("Numeric cols", len(num_cols)),
        ]:
            metrics_html += (
                f'<div class="metric">'
                f'<div class="value">{val}</div>'
                f'<div class="label">{label}</div>'
                f'</div>'
            )

        self._sections.append(
            f'<div class="card">'
            f'<h2>Dataset Overview</h2>'
            f'<div class="metric-grid">{metrics_html}</div>'
            f'</div>'
        )
        return self

    def add_schema_table(self) -> "HTMLExporter":
        """Add a schema/column-types table."""
        if self._df is None:
            return self
        from aiviz.ingestion.schema import inspect_schema
        schema = inspect_schema(self._df)
        rows_html = ""
        for c in schema.columns:
            rows_html += (
                f"<tr><td>{c.name}</td><td>{c.dtype}</td><td>{c.inferred_role}</td>"
                f"<td>{c.non_null}</td><td>{c.null_pct:.1f}%</td><td>{c.unique}</td></tr>"
            )
        self._sections.append(
            '<div class="card"><h2>Schema</h2>'
            '<table><thead><tr>'
            '<th>Column</th><th>Type</th><th>Role</th>'
            '<th>Non-null</th><th>Null %</th><th>Unique</th>'
            '</tr></thead><tbody>'
            f'{rows_html}'
            '</tbody></table></div>'
        )
        return self

    def add_stats_table(self) -> "HTMLExporter":
        """Add descriptive statistics table."""
        if self._df is None:
            return self
        num = self._df.select_dtypes(include="number")
        if num.empty:
            return self
        stats = num.describe().round(4)
        header = "<tr><th>stat</th>" + "".join(f"<th>{c}</th>" for c in stats.columns) + "</tr>"
        rows_html = ""
        for idx in stats.index:
            rows_html += "<tr>" + f"<td>{idx}</td>" + "".join(
                f"<td>{stats.loc[idx, c]}</td>" for c in stats.columns
            ) + "</tr>"
        self._sections.append(
            '<div class="card"><h2>Descriptive Statistics</h2>'
            f'<table><thead>{header}</thead><tbody>{rows_html}</tbody></table></div>'
        )
        return self

    def add_chart_image(self, png_bytes: bytes, title: str = "Chart") -> "HTMLExporter":
        """Embed a chart as a base64-encoded PNG image."""
        b64 = base64.b64encode(png_bytes).decode()
        self._sections.append(
            f'<div class="card">'
            f'<h2>{title}</h2>'
            f'<img class="chart" src="data:image/png;base64,{b64}" alt="{title}">'
            f'</div>'
        )
        return self

    def add_text_section(self, title: str, content: str) -> "HTMLExporter":
        """Add a freeform text section (e.g. AI insight, summary, notes)."""
        # Convert newlines to <br> for display
        formatted = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        formatted = formatted.replace("\n", "<br>")
        self._sections.append(
            f'<div class="card"><h2>{title}</h2><p>{formatted}</p></div>'
        )
        return self

    def add_preformatted(self, title: str, content: str) -> "HTMLExporter":
        """Add a monospaced preformatted block (e.g. model summary)."""
        escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._sections.append(
            f'<div class="card"><h2>{title}</h2><pre>{escaped}</pre></div>'
        )
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Assemble all sections into a complete HTML document."""
        body_parts = "\n".join(self._sections)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{APP.name} Report – {self._file_name}</title>
{_CSS}
</head>
<body>
<h1>📊 {APP.name} Analytics Report</h1>
<p class="meta">
  File: <strong>{self._file_name}</strong> &nbsp;|&nbsp;
  Generated: {self._ts} &nbsp;|&nbsp;
  {APP.name} v{APP.version}
</p>
{body_parts}
<footer>Generated by {APP.name} v{APP.version} · Local AI Analytics Platform</footer>
</body>
</html>"""

    def save(self, path: str) -> None:
        """Write the report to a file."""
        Path(path).write_text(self.build(), encoding="utf-8")
