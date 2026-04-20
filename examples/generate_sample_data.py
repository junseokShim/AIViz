"""
Generate sample datasets for AIViz testing and demonstration.

Run:
    python examples/generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__))


def generate_timeseries_csv() -> None:
    """Multi-signal time-series with trend, noise, and injected anomalies."""
    np.random.seed(42)
    n = 500
    t = np.linspace(0, 50, n)

    # Signal A: sine with trend + noise
    signal_a = 2 * np.sin(2 * np.pi * 0.1 * t) + 0.05 * t + np.random.normal(0, 0.2, n)

    # Signal B: square-wave approximation
    signal_b = np.sign(np.sin(2 * np.pi * 0.05 * t)) + np.random.normal(0, 0.1, n)

    # Signal C: exponential decay with noise
    signal_c = 5 * np.exp(-0.03 * t) + np.random.normal(0, 0.3, n)

    # Inject anomalies into A
    anomaly_idx = [50, 150, 300, 420]
    for i in anomaly_idx:
        signal_a[i] += np.random.choice([-1, 1]) * 4

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "signal_a": signal_a,
        "signal_b": signal_b,
        "signal_c": signal_c,
        "temperature": 20 + 5 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 0.5, n),
        "category": np.random.choice(["alpha", "beta", "gamma"], size=n),
    })

    # Inject some missing values
    df.loc[np.random.choice(df.index, 10), "signal_b"] = np.nan

    path = os.path.join(OUT_DIR, "sample_timeseries.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(df)} rows)")


def generate_sales_csv() -> None:
    """Tabular sales dataset for chart-builder demos."""
    np.random.seed(7)
    n = 300
    regions = ["North", "South", "East", "West"]
    products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Service Z"]

    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="1D"),
        "region": np.random.choice(regions, n),
        "product": np.random.choice(products, n),
        "units_sold": np.random.randint(1, 200, n),
        "unit_price": np.round(np.random.uniform(9.99, 299.99, n), 2),
        "discount_pct": np.random.choice([0, 5, 10, 15, 20], n),
        "customer_rating": np.round(np.random.uniform(2.0, 5.0, n), 1),
    })
    df["revenue"] = (df["units_sold"] * df["unit_price"] * (1 - df["discount_pct"] / 100)).round(2)

    path = os.path.join(OUT_DIR, "sample_sales.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(df)} rows)")


def generate_frequency_csv() -> None:
    """Signal with known frequencies for FFT demo."""
    np.random.seed(0)
    fs = 1000  # Hz
    t = np.arange(0, 2, 1 / fs)  # 2 seconds

    # Composite signal: 50 Hz + 120 Hz + 200 Hz + noise
    signal = (
        1.0 * np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 120 * t)
        + 0.25 * np.sin(2 * np.pi * 200 * t)
        + np.random.normal(0, 0.05, len(t))
    )

    df = pd.DataFrame({
        "time_s": t,
        "signal": signal,
        "noisy_signal": signal + np.random.normal(0, 0.3, len(t)),
    })

    path = os.path.join(OUT_DIR, "sample_frequency.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(df)} rows, fs={fs} Hz)")
    print(f"  → Expected FFT peaks at: 50, 120, 200 Hz")


if __name__ == "__main__":
    generate_timeseries_csv()
    generate_sales_csv()
    generate_frequency_csv()
    print("\nAll sample datasets generated in examples/")
