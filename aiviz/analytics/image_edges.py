"""
Edge detection module for AIViz image analysis.

Supported methods:
- canny:     Canny edge detector (uses cv2 if available, else scipy fallback)
- sobel:     Sobel gradient magnitude (scipy.ndimage)
- laplacian: Laplacian of Gaussian (scipy.ndimage)

All functions accept a PIL Image and return an EdgeResult with a
2-D uint8 edge map that can be directly displayed or overlaid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage

# Optional OpenCV – graceful fallback if not installed
try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _cv2 = None          # type: ignore[assignment]
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EdgeResult:
    method: str
    edges: np.ndarray    # 2-D uint8 array (0 = no edge, 255 = edge)
    params: dict
    has_cv2: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_edges(
    img: Image.Image,
    method: str = "canny",
    low_threshold: int = 50,
    high_threshold: int = 150,
    sigma: float = 1.5,
) -> EdgeResult:
    """
    Run edge detection on *img*.

    Args:
        img:            Input PIL Image (any mode; converted to grayscale internally).
        method:         'canny' | 'sobel' | 'laplacian'
        low_threshold:  Canny lower hysteresis threshold (0-255).
        high_threshold: Canny upper hysteresis threshold (0-255).
        sigma:          Gaussian blur sigma before Canny / Laplacian.

    Returns:
        EdgeResult with a 2-D uint8 edge map.
    """
    gray = np.array(img.convert("L"), dtype=np.uint8)

    method = method.lower()

    if method == "canny":
        edges, used_cv2 = _canny(gray, low_threshold, high_threshold, sigma)
        return EdgeResult(
            method="canny",
            edges=edges,
            params={"low_threshold": low_threshold, "high_threshold": high_threshold, "sigma": sigma},
            has_cv2=used_cv2,
        )

    elif method == "sobel":
        edges = _sobel(gray)
        return EdgeResult(method="sobel", edges=edges, params={})

    elif method == "laplacian":
        edges = _laplacian(gray, sigma=sigma)
        return EdgeResult(method="laplacian", edges=edges, params={"sigma": sigma})

    else:
        raise ValueError(f"Unknown edge method: '{method}'. Choose from: canny, sobel, laplacian")


def edges_to_pil(result: EdgeResult) -> Image.Image:
    """Convert an EdgeResult edge map to a PIL grayscale Image."""
    return Image.fromarray(result.edges, mode="L")


# ---------------------------------------------------------------------------
# Private implementations
# ---------------------------------------------------------------------------

def _canny(
    gray: np.ndarray,
    low: int,
    high: int,
    sigma: float,
) -> tuple[np.ndarray, bool]:
    """Canny detector. Prefers cv2; falls back to scipy-based implementation."""
    if _HAS_CV2:
        blurred = _cv2.GaussianBlur(gray, (0, 0), sigma) if sigma > 0 else gray
        edges = _cv2.Canny(blurred, low, high)
        return edges.astype(np.uint8), True
    else:
        return _numpy_canny(gray, low, high, sigma), False


def _numpy_canny(
    gray: np.ndarray,
    low_threshold: int,
    high_threshold: int,
    sigma: float,
) -> np.ndarray:
    """
    Simplified Canny implementation using scipy (no cv2 dependency).

    Steps: Gaussian blur → Sobel gradient → NMS (simplified) → double threshold → hysteresis
    """
    # 1. Gaussian blur
    if sigma > 0:
        blurred = ndimage.gaussian_filter(gray.astype(np.float32), sigma=sigma)
    else:
        blurred = gray.astype(np.float32)

    # 2. Gradient (Sobel)
    gx = ndimage.sobel(blurred, axis=1)
    gy = ndimage.sobel(blurred, axis=0)
    magnitude = np.hypot(gx, gy)

    # 3. Scale to 0-255
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max * 255.0

    # 4. Double threshold
    strong = (magnitude >= high_threshold).astype(np.uint8)
    weak   = ((magnitude >= low_threshold) & (magnitude < high_threshold)).astype(np.uint8)

    # 5. Hysteresis: keep weak pixels connected to strong ones
    combined = strong | weak
    labeled, _ = ndimage.label(combined)
    strong_labels = set(np.unique(labeled[strong > 0])) - {0}
    result = np.zeros_like(magnitude, dtype=np.uint8)
    for lbl in strong_labels:
        result[labeled == lbl] = 255
    return result


def _sobel(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, normalised to 0-255."""
    gx = ndimage.sobel(gray.astype(np.float32), axis=1)
    gy = ndimage.sobel(gray.astype(np.float32), axis=0)
    mag = np.hypot(gx, gy)
    mx = mag.max()
    if mx > 0:
        mag = mag / mx * 255.0
    return mag.astype(np.uint8)


def _laplacian(gray: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Laplacian of Gaussian edge detector, normalised to 0-255."""
    if sigma > 0:
        blurred = ndimage.gaussian_filter(gray.astype(np.float32), sigma=sigma)
    else:
        blurred = gray.astype(np.float32)
    lap = ndimage.laplace(blurred)
    lap = np.abs(lap)
    mx = lap.max()
    if mx > 0:
        lap = lap / mx * 255.0
    return lap.astype(np.uint8)
