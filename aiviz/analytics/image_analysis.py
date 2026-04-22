"""
Image analysis module.

Provides metadata extraction, pixel statistics, histogram computation,
and basic feature extraction from PIL Images.

Designed to be extended with richer CV analysis (CLIP embeddings,
edge detection, texture descriptors) without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


@dataclass
class ImageAnalysisResult:
    file_name: str
    width: int
    height: int
    mode: str                   # 'RGB', 'L', 'RGBA', etc.
    n_channels: int
    n_pixels: int
    format: Optional[str]       # 'JPEG', 'PNG', etc.
    # Per-channel stats
    channel_stats: pd.DataFrame  # columns: channel, mean, std, min, max
    # Histogram data
    histograms: dict[str, tuple[np.ndarray, np.ndarray]]  # channel -> (counts, bin_edges)
    # Basic feature flags
    has_transparency: bool
    is_grayscale: bool
    aspect_ratio: float
    dominant_colors: Optional[pd.DataFrame]   # top colors (reduced palette)
    metadata: dict                            # EXIF or other metadata


def analyze_image(img: Image.Image, file_name: str = "", n_colors: int = 8) -> ImageAnalysisResult:
    """
    Run full analytics on a PIL Image.

    Args:
        img:       Loaded PIL Image.
        file_name: Original file name for reference.
        n_colors:  Number of dominant colors to extract.

    Returns:
        ImageAnalysisResult with all analytics populated.
    """
    w, h = img.size
    mode = img.mode
    n_channels = len(img.getbands())
    has_transparency = mode in ("RGBA", "LA", "PA")
    is_grayscale = mode in ("L", "LA", "1")

    # Per-channel statistics (use numpy for numerical stability)
    # PIL ImageStat uses the computational variance formula (sum(x²)/n - mean²)
    # which can yield tiny negative variances due to floating-point rounding,
    # causing ValueError in sqrt.  numpy's std() is always non-negative.
    bands = img.getbands()
    arr = np.array(img)
    channel_rows = []
    for i, band in enumerate(bands):
        if arr.ndim == 2:
            ch_arr = arr.astype(np.float64)
        else:
            ch_arr = arr[:, :, i].astype(np.float64)
        channel_rows.append({
            "channel": band,
            "mean": float(np.mean(ch_arr)),
            "std": float(np.std(ch_arr)),   # always ≥ 0
            "min": float(np.min(ch_arr)),
            "max": float(np.max(ch_arr)),
        })
    channel_stats = pd.DataFrame(channel_rows)

    # Histograms  (arr already computed above)
    histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if arr.ndim == 2:
        counts, edges = np.histogram(arr.ravel(), bins=64, range=(0, 256))
        histograms["L"] = (counts, edges)
    else:
        for i, band in enumerate(bands[:3]):  # skip alpha for histogram
            counts, edges = np.histogram(arr[:, :, i].ravel(), bins=64, range=(0, 256))
            histograms[band] = (counts, edges)

    # Dominant colors via quantize
    dominant_colors: Optional[pd.DataFrame] = None
    try:
        rgb_img = img.convert("RGB")
        quantized = rgb_img.quantize(colors=n_colors, dither=0)
        palette = quantized.getpalette()
        if palette is None:
            raise ValueError("Empty palette")
        # palette is a flat list [R,G,B, R,G,B, ...] of length 768
        actual_colors = min(n_colors, len(palette) // 3)
        if actual_colors == 0:
            raise ValueError("No colors in palette")
        colors = [tuple(palette[i * 3: i * 3 + 3]) for i in range(actual_colors)]
        raw_hist = quantized.histogram()
        hist = raw_hist[:actual_colors]
        total = sum(hist) or 1
        # Filter out palette entries with zero count (unused colors)
        entries = [(c, h) for c, h in zip(colors, hist) if h > 0]
        if not entries:
            raise ValueError("All colors have zero count")
        colors_f, hist_f = zip(*entries)
        dominant_colors = pd.DataFrame({
            "color_rgb": [f"rgb{c}" for c in colors_f],
            "r": [c[0] for c in colors_f],
            "g": [c[1] for c in colors_f],
            "b": [c[2] for c in colors_f],
            "fraction": [h / total for h in hist_f],
        })
    except Exception:
        dominant_colors = None

    # EXIF / metadata
    metadata: dict = {}
    try:
        exif_data = img._getexif() if hasattr(img, "_getexif") else None
        if exif_data:
            from PIL.ExifTags import TAGS
            metadata = {TAGS.get(k, k): str(v) for k, v in exif_data.items()}
    except Exception:
        pass

    return ImageAnalysisResult(
        file_name=file_name,
        width=w,
        height=h,
        mode=mode,
        n_channels=n_channels,
        n_pixels=w * h,
        format=getattr(img, "format", None),
        channel_stats=channel_stats,
        histograms=histograms,
        has_transparency=has_transparency,
        is_grayscale=is_grayscale,
        aspect_ratio=w / h if h > 0 else 1.0,
        dominant_colors=dominant_colors,
        metadata=metadata,
    )


def to_grayscale(img: Image.Image) -> Image.Image:
    """Convert image to grayscale."""
    return img.convert("L")


def resize_for_display(img: Image.Image, max_size: int = 800) -> Image.Image:
    """Downscale image for display while preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
