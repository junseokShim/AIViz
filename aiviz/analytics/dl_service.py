"""
Simple Deep Learning module – MLP tabular regression/classification.

Uses scikit-learn's MLPRegressor/MLPClassifier as the backend.
This is intentionally lightweight – the goal is a working baseline,
not a full AutoML pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("aiviz.dl_service")


@dataclass
class DLResult:
    task: str           # 'regression' | 'classification'
    target_col: str
    feature_cols: list[str]
    n_train: int
    n_test: int
    metrics: dict       # rmse/mae for regression; accuracy/f1 for classification
    predictions: Optional[np.ndarray] = None
    test_targets: Optional[np.ndarray] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def metrics_text(self) -> str:
        lines = []
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"{k}: {v:.4f}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)


def run_mlp(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    task: str = "auto",
    test_size: float = 0.2,
    hidden_layer_sizes: tuple = (64, 32),
    max_iter: int = 300,
    random_state: int = 42,
) -> DLResult:
    """
    Train a simple MLP on tabular data.

    Args:
        df:                 Input DataFrame.
        target_col:         Column to predict.
        feature_cols:       Feature columns (must be numeric after encoding).
        task:               'regression', 'classification', or 'auto' (inferred).
        test_size:          Fraction of data for test set.
        hidden_layer_sizes: MLP architecture tuple.
        max_iter:           Maximum training iterations.

    Returns:
        DLResult with metrics and predictions.
    """
    try:
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error,
            accuracy_score, f1_score,
        )
    except ImportError:
        return DLResult(
            task=task, target_col=target_col, feature_cols=feature_cols,
            n_train=0, n_test=0, metrics={},
            error="scikit-learn이 설치되지 않았습니다. pip install scikit-learn"
        )

    # Subset and drop NaN
    cols = feature_cols + [target_col]
    sub = df[cols].dropna()

    if len(sub) < 20:
        return DLResult(
            task=task, target_col=target_col, feature_cols=feature_cols,
            n_train=0, n_test=0, metrics={},
            error=f"데이터가 너무 적습니다 ({len(sub)}행). 최소 20행 이상 필요합니다."
        )

    X = sub[feature_cols].values.astype(float)
    y_raw = sub[target_col]

    # Infer task
    if task == "auto":
        n_unique = y_raw.nunique()
        task = "classification" if (n_unique <= 20 and y_raw.dtype == object or n_unique <= 10) else "regression"

    # Encode target for classification
    le = None
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values.astype(float)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    except Exception as exc:
        return DLResult(
            task=task, target_col=target_col, feature_cols=feature_cols,
            n_train=0, n_test=0, metrics={}, error=f"데이터 분할 오류: {exc}"
        )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    try:
        if task == "regression":
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            metrics = {"RMSE": rmse, "MAE": mae, "반복 횟수": model.n_iter_}
        else:
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            acc = float(accuracy_score(y_test, preds))
            try:
                avg = "binary" if len(np.unique(y)) == 2 else "weighted"
                f1 = float(f1_score(y_test, preds, average=avg))
            except Exception:
                f1 = float("nan")
            metrics = {"정확도": acc, "F1": f1, "클래스 수": len(np.unique(y)), "반복 횟수": model.n_iter_}

        return DLResult(
            task=task,
            target_col=target_col,
            feature_cols=feature_cols,
            n_train=len(X_train),
            n_test=len(X_test),
            metrics=metrics,
            predictions=preds,
            test_targets=y_test,
        )

    except Exception as exc:
        return DLResult(
            task=task, target_col=target_col, feature_cols=feature_cols,
            n_train=len(X_train), n_test=len(X_test), metrics={},
            error=f"학습 오류: {exc}"
        )
