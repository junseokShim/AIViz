"""
Clustering module – KMeans (required) and DBSCAN (optional).

Uses scikit-learn. Falls back gracefully if not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("aiviz.clustering")


@dataclass
class ClusterResult:
    method: str
    n_clusters: int
    labels: np.ndarray          # cluster label per row (-1 = noise for DBSCAN)
    cluster_sizes: dict[int, int]
    feature_cols: list[str]
    centers: Optional[np.ndarray] = None   # KMeans centroids
    inertia: Optional[float] = None        # KMeans inertia
    silhouette: Optional[float] = None     # silhouette score if computed
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_label_series(self, index) -> pd.Series:
        return pd.Series(self.labels, index=index, name="cluster_label")


def run_kmeans(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 3,
    scale: bool = True,
    compute_silhouette: bool = True,
    random_state: int = 42,
) -> ClusterResult:
    """
    Run KMeans clustering on selected numeric columns.

    Args:
        df:           Input DataFrame.
        feature_cols: Columns to cluster on (must be numeric).
        n_clusters:   Number of clusters.
        scale:        StandardScaler preprocessing.
        compute_silhouette: Compute silhouette score (may be slow for large data).

    Returns:
        ClusterResult.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return ClusterResult(
            method="KMeans", n_clusters=n_clusters, labels=np.array([]),
            cluster_sizes={}, feature_cols=feature_cols,
            error="scikit-learn이 설치되지 않았습니다. pip install scikit-learn"
        )

    sub = df[feature_cols].dropna()
    if len(sub) < n_clusters:
        return ClusterResult(
            method="KMeans", n_clusters=n_clusters, labels=np.array([]),
            cluster_sizes={}, feature_cols=feature_cols,
            error=f"데이터 행({len(sub)})이 클러스터 수({n_clusters})보다 적습니다."
        )

    try:
        X = sub.values.astype(float)
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)

        sizes = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}

        silhouette = None
        if compute_silhouette and len(set(labels)) > 1 and len(sub) > n_clusters:
            try:
                from sklearn.metrics import silhouette_score
                silhouette = float(silhouette_score(X, labels, sample_size=min(2000, len(sub))))
            except Exception:
                pass

        return ClusterResult(
            method="KMeans",
            n_clusters=n_clusters,
            labels=labels,
            cluster_sizes=sizes,
            feature_cols=feature_cols,
            centers=km.cluster_centers_,
            inertia=float(km.inertia_),
            silhouette=silhouette,
        )

    except Exception as exc:
        return ClusterResult(
            method="KMeans", n_clusters=n_clusters, labels=np.array([]),
            cluster_sizes={}, feature_cols=feature_cols,
            error=f"KMeans 오류: {exc}"
        )


def run_dbscan(
    df: pd.DataFrame,
    feature_cols: list[str],
    eps: float = 0.5,
    min_samples: int = 5,
    scale: bool = True,
) -> ClusterResult:
    """
    Run DBSCAN clustering. Noise points get label -1.
    """
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return ClusterResult(
            method="DBSCAN", n_clusters=0, labels=np.array([]),
            cluster_sizes={}, feature_cols=feature_cols,
            error="scikit-learn이 설치되지 않았습니다."
        )

    sub = df[feature_cols].dropna()
    if len(sub) < min_samples:
        return ClusterResult(
            method="DBSCAN", n_clusters=0, labels=np.array([]),
            cluster_sizes={}, feature_cols=feature_cols,
            error=f"데이터가 너무 적습니다 ({len(sub)}행)."
        )

    try:
        X = sub.values.astype(float)
        if scale:
            X = StandardScaler().fit_transform(X)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        unique, counts = np.unique(labels, return_counts=True)
        sizes = {int(k): int(v) for k, v in zip(unique, counts)}
        n_clusters = len([k for k in unique if k >= 0])

        return ClusterResult(
            method="DBSCAN",
            n_clusters=n_clusters,
            labels=labels,
            cluster_sizes=sizes,
            feature_cols=feature_cols,
        )

    except Exception as exc:
        return ClusterResult(
            method="DBSCAN", n_clusters=0, labels=np.array([]),
            cluster_sizes={}, feature_cols=feature_cols,
            error=f"DBSCAN 오류: {exc}"
        )
