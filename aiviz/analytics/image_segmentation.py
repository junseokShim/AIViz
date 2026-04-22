"""
Basic image segmentation module for AIViz.

Supported methods:
- threshold:  Global threshold → binary mask → labeled regions
- adaptive:   Locally adaptive threshold (local-mean based)
- kmeans:     K-Means colour/intensity clustering (requires scikit-learn)

All functions accept a PIL Image and return a SegmentResult
with a labeled integer mask and region count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    method: str
    mask: np.ndarray        # 2-D int32 label array  (0 = background, 1…N = regions)
    n_regions: int
    params: dict
    colormap: np.ndarray    # (N+1, 3) uint8 RGB colours for each label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_image(
    img: Image.Image,
    method: str = "threshold",
    threshold: int = 128,
    block_size: int = 51,
    offset: int = 10,
    n_clusters: int = 3,
) -> SegmentResult:
    """
    Segment *img* into regions.

    Args:
        img:        Input PIL Image (any mode).
        method:     'threshold' | 'adaptive' | 'kmeans'
        threshold:  Global binarisation threshold (threshold method).
        block_size: Neighbourhood size for adaptive thresholding (must be odd, ≥3).
        offset:     Offset subtracted from local mean in adaptive method.
        n_clusters: Number of K-Means clusters.

    Returns:
        SegmentResult with integer label mask and region count.
    """
    method = method.lower()

    if method == "threshold":
        return _threshold(img, threshold)
    elif method == "adaptive":
        return _adaptive(img, block_size, offset)
    elif method == "kmeans":
        return _kmeans(img, n_clusters)
    else:
        raise ValueError(f"Unknown segmentation method: '{method}'. "
                         f"Choose from: threshold, adaptive, kmeans")


def mask_to_pil(result: SegmentResult, alpha: int = 180) -> Image.Image:
    """
    Convert a SegmentResult mask to an RGBA overlay image.

    Each region gets a distinct colour; background (label=0) is transparent.
    """
    h, w = result.mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for label in range(1, result.n_regions + 1):
        m = result.mask == label
        if not m.any():
            continue
        color = result.colormap[label % len(result.colormap)]
        rgba[m, :3] = color
        rgba[m, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------------
# Private implementations
# ---------------------------------------------------------------------------

def _make_colormap(n: int) -> np.ndarray:
    """Generate N+1 distinct RGB colours (index 0 is reserved / black)."""
    rng = np.random.default_rng(42)
    colors = np.zeros((n + 1, 3), dtype=np.uint8)
    # Use HSV sampling for distinct hues
    hues = np.linspace(0, 1, n, endpoint=False)
    for i, h in enumerate(hues):
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
        colors[i + 1] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


def _threshold(img: Image.Image, threshold: int) -> SegmentResult:
    gray = np.array(img.convert("L"), dtype=np.uint8)
    binary = (gray > threshold).astype(np.int32)
    labeled, n = ndimage.label(binary)
    return SegmentResult(
        method="threshold",
        mask=labeled.astype(np.int32),
        n_regions=n,
        params={"threshold": threshold},
        colormap=_make_colormap(n),
    )


def _adaptive(img: Image.Image, block_size: int, offset: int) -> SegmentResult:
    gray = np.array(img.convert("L"), dtype=np.float32)
    # Ensure block_size is odd and ≥3
    block_size = max(3, block_size | 1)
    local_mean = ndimage.uniform_filter(gray, size=block_size)
    binary = (gray > (local_mean - offset)).astype(np.int32)
    labeled, n = ndimage.label(binary)
    return SegmentResult(
        method="adaptive",
        mask=labeled.astype(np.int32),
        n_regions=n,
        params={"block_size": block_size, "offset": offset},
        colormap=_make_colormap(n),
    )


def _kmeans(img: Image.Image, n_clusters: int) -> SegmentResult:
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "scikit-learn is required for K-Means segmentation.\n"
            "Install with:  pip install scikit-learn"
        )

    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    flat = gray.ravel().reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(flat)
    mask = labels.reshape(h, w).astype(np.int32)
    # Re-label from 1 so 0 is background
    mask = mask + 1
    return SegmentResult(
        method="kmeans",
        mask=mask,
        n_regions=n_clusters,
        params={"n_clusters": n_clusters},
        colormap=_make_colormap(n_clusters),
    )
