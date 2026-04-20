# AIViz Developer Guide

Step-by-step recipes for extending AIViz.

---

## Adding a New Chart Type

**Example: violin plot**

### 1. Add factory function in `aiviz/visualization/mpl_charts.py`

```python
def plot_violin(ax: Axes, df: pd.DataFrame, columns: list[str]) -> None:
    """Violin plot for one or more numeric columns."""
    data = [df[c].dropna().values for c in columns]
    parts = ax.violinplot(data, positions=range(len(columns)), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(SERIES_COLORS[0])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=20, fontsize=9)
    ax.set_title("Violin Plot")
```

### 2. Register in `aiviz/ui/panel_charts.py`

```python
CHART_TYPES = [
    "Line Chart", "Scatter Plot", "Bar Chart",
    "Histogram", "Box Plot", "Correlation",
    "Violin Plot",   # ← add here
]
```

### 3. Add rendering branch in `_generate_chart()`

```python
elif chart_type == "Violin Plot":
    cols = y_items if y_items else num_cols[:5]
    if cols:
        mpl_charts.plot_violin(ax, self._df, cols)
```

---

## Adding a New Analysis Function

**Example: autocorrelation**

### 1. Add to `aiviz/analytics/timeseries.py`

```python
def compute_autocorrelation(series: pd.Series, max_lag: int = 50) -> pd.DataFrame:
    """Compute ACF for lags 0..max_lag. Returns DataFrame with columns: lag, acf."""
    try:
        from statsmodels.tsa.stattools import acf
        values = acf(series.dropna(), nlags=max_lag)
    except ImportError:
        # Simple manual fallback
        s = series.dropna().values
        values = [np.corrcoef(s[:-i], s[i:])[0,1] if i else 1.0 for i in range(max_lag+1)]
    return pd.DataFrame({"lag": range(len(values)), "acf": values})
```

### 2. Add chart factory

```python
# mpl_charts.py
def plot_acf(ax: Axes, acf_df: pd.DataFrame) -> None:
    ax.bar(acf_df["lag"], acf_df["acf"], color=C_BLUE, width=0.8)
    ax.axhline(y=0, color=C_OVERLAY)
    ax.axhline(y=1.96/np.sqrt(len(acf_df)), color=C_ORANGE, linestyle="--", label="95% CI")
    ax.axhline(y=-1.96/np.sqrt(len(acf_df)), color=C_ORANGE, linestyle="--")
    ax.set_title("Autocorrelation Function (ACF)")
    ax.set_xlabel("Lag")
    ax.legend(fontsize=9)
```

### 3. Wire into UI panel (`panel_timeseries.py`)

```python
# In the multi-signal tab or as a new sub-tab
acf_df = compute_autocorrelation(series, max_lag=50)
ax = self._acf_plot.get_ax()
mpl_charts.plot_acf(ax, acf_df)
self._acf_plot.redraw()
```

### 4. Write a test

```python
# tests/test_analytics.py
def test_autocorrelation():
    s = pd.Series(np.sin(2 * np.pi * 0.1 * np.arange(200)))
    acf_df = compute_autocorrelation(s, max_lag=50)
    assert len(acf_df) == 51
    assert abs(acf_df.loc[0, "acf"] - 1.0) < 0.01
```

---

## Adding a New AI Tool or Prompt Template

**Example: anomaly root-cause suggestion**

### 1. Add prompt to `aiviz/ai/prompts.py`

```python
def anomaly_rootcause_prompt(col: str, anomaly_values: list, context: str) -> str:
    return (
        f"You are AIViz, an anomaly analysis expert.\n\n"
        f"Column '{col}' has {len(anomaly_values)} anomalous values:\n"
        f"{anomaly_values[:10]}\n\n"
        f"Dataset context:\n{context}\n\n"
        "Suggest 3 possible root causes. Be specific and practical."
    )
```

### 2. Add method to `AnalysisAgent` in `aiviz/ai/agent.py`

```python
def suggest_anomaly_causes(
    self, col: str, anomaly_vals: list, df: pd.DataFrame
) -> AgentResult:
    context = truncate_str(df_to_context_string(df, max_rows=5), 1500)
    prompt = prompts.anomaly_rootcause_prompt(col, anomaly_vals, context)
    return self._call(
        prompt,
        fallback=f"{len(anomaly_vals)} anomalies in '{col}'. (AI unavailable)"
    )
```

### 3. Call from panel using WorkerThread

```python
# panel_timeseries.py
def _suggest_causes(self) -> None:
    anomaly_vals = self._result.original[self._result.anomalies].tolist()
    agent = AnalysisAgent()
    worker = WorkerThread(
        agent.suggest_anomaly_causes,
        self._sig_combo.currentText(), anomaly_vals, self._df
    )
    worker.result_ready.connect(lambda r: QMessageBox.information(self, "AI", r.answer))
    worker.start()
    self._worker = worker
```

---

## Adding Support for a New File Type

**Example: `.tsv` (tab-separated)**

### 1. Register extension in `config.py`

```python
supported_tabular: tuple = (".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv")
```

### 2. Add parser in `aiviz/ingestion/loader.py`

```python
elif suffix == ".tsv":
    df = pd.read_csv(buf, sep="\t")
```

### 3. Add test

```python
def test_load_tsv(self):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    tsv = df.to_csv(index=False, sep="\t").encode()
    result = load_file(tsv, "test.tsv")
    assert result.ok
    assert len(result.df) == 2
```

Note: The sidebar file dialog picks up the new extension automatically since it reads `APP.supported_tabular` dynamically.

---

## Adding a New Export Section

**Example: adding a "Time-Series Summary" section**

### 1. Add method to `HTMLExporter`

```python
def add_timeseries_section(self, col: str, stats: dict) -> "HTMLExporter":
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4g}</td></tr>"
        for k, v in stats.items()
    )
    self._sections.append(
        f'<div class="card"><h2>Time-Series: {col}</h2>'
        f'<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )
    return self
```

### 2. Add the same to `PDFExporter`

```python
def add_timeseries_section(self, col: str, stats: dict) -> "PDFExporter":
    self._section_title(f"Time-Series: {col}")
    rows = [(k, f"{v:.4g}" if isinstance(v, float) else str(v)) for k, v in stats.items()]
    self._key_value_table(rows)
    return self
```

### 3. Call from `panel_export.py`

```python
# In _export_html():
if self._chk_timeseries.isChecked() and self._ts_stats:
    exp.add_timeseries_section(self._ts_col, self._ts_stats)
```

---

## Development Workflow

```bash
# Install dependencies
pip install -r requirements.txt

# Generate fresh sample data
python examples/generate_sample_data.py

# Run tests (fast, no Ollama needed)
python -m pytest tests/ -v

# Launch the app
python main.py

# Verify all imports without launching
python -c "
import matplotlib; matplotlib.use('Agg')
from PyQt6.QtWidgets import QApplication; import sys
app = QApplication(sys.argv)
from aiviz.app.main_window import MainWindow
w = MainWindow()
print('OK')
"
```
