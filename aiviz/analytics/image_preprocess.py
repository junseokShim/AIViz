"""
Image preprocessing pipeline.

Provides a composable set of preprocessing operations:
- Brightness / Contrast / Saturation adjustment (PIL ImageEnhance)
- Pixel clipping with optional re-normalization
- Grayscale conversion
- Normalization (stretch to full 0-255 range)

Design: stateless functions + PreprocessParams dataclass.
The original image is never modified; all functions return a new Image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance


# ---------------------------------------------------------------------------
# Params dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessParams:
    """Holds all preprocessing parameters with safe defaults (identity transform)."""
    brightness: float = 1.0    # 0.1 – 3.0  (1.0 = no change)
    contrast: float = 1.0      # 0.1 – 3.0
    saturation: float = 1.0    # 0.0 – 3.0  (only for color images)
    clip_min: int = 0          # 0 – 255  (pixels below are clipped)
    clip_max: int = 255        # 0 – 255  (pixels above are clipped)
    grayscale: bool = False    # convert to single-channel grayscale
    normalize: bool = False    # stretch clipped range to 0-255

    def is_identity(self) -> bool:
        """Return True when no operation changes the image."""
        return (
            self.brightness == 1.0
            and self.contrast == 1.0
            and self.saturation == 1.0
            and self.clip_min == 0
            and self.clip_max == 255
            and not self.grayscale
            and not self.normalize
        )

    def to_dict(self) -> dict:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "grayscale": self.grayscale,
            "normalize": self.normalize,
        }

    @staticmethod
    def from_dict(d: dict) -> "PreprocessParams":
        return PreprocessParams(
            brightness=float(d.get("brightness", 1.0)),
            contrast=float(d.get("contrast", 1.0)),
            saturation=float(d.get("saturation", 1.0)),
            clip_min=int(d.get("clip_min", 0)),
            clip_max=int(d.get("clip_max", 255)),
            grayscale=str(d.get("grayscale", "False")).lower() == "true",
            normalize=str(d.get("normalize", "False")).lower() == "true",
        )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def apply_preprocessing(img: Image.Image, params: PreprocessParams) -> Image.Image:
    """
    Apply the preprocessing pipeline to *img*.

    Returns a new PIL Image; the original is never mutated.
    Operations are applied in this order:
      brightness → contrast → saturation → clip → normalize → grayscale
    """
    if params.is_identity():
        return img.copy()

    result = img.copy()

    # ── Brightness ────────────────────────────────────────────────
    if params.brightness != 1.0:
        result = ImageEnhance.Brightness(result).enhance(
            max(0.1, min(3.0, params.brightness))
        )

    # ── Contrast ──────────────────────────────────────────────────
    if params.contrast != 1.0:
        result = ImageEnhance.Contrast(result).enhance(
            max(0.1, min(3.0, params.contrast))
        )

    # ── Saturation (color images only) ────────────────────────────
    if params.saturation != 1.0 and result.mode not in ("L", "1", "I", "F"):
        result = ImageEnhance.Color(result).enhance(
            max(0.0, min(3.0, params.saturation))
        )

    # ── Pixel clipping ────────────────────────────────────────────
    clip_min = max(0, min(254, params.clip_min))
    clip_max = max(clip_min + 1, min(255, params.clip_max))
    if clip_min != 0 or clip_max != 255:
        arr = np.array(result, dtype=np.float32)
        arr = np.clip(arr, clip_min, clip_max)
        if params.normalize:
            # Stretch clipped range back to 0-255
            rng = clip_max - clip_min
            arr = (arr - clip_min) / rng * 255.0
        result = Image.fromarray(arr.astype(np.uint8))

    # ── Grayscale ─────────────────────────────────────────────────
    if params.grayscale:
        result = result.convert("L")

    return result


def reset_params() -> PreprocessParams:
    """Return a default (identity) PreprocessParams."""
    return PreprocessParams()
