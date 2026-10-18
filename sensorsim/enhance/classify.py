"""Is an uploaded image a photograph or a diagram?

The two need different treatment. A photograph has continuous tone, sensor
noise and texture, which suits ``RealESRGAN_x4plus`` and the curve stage. A
diagram, chart or screenshot has a few flat colours, hard edges and text, and
the photographic generator smears it. Three measurements separate them:

* **palette concentration** - share of pixels in the eight most common colours
  after quantising to 32 levels per channel;
* **flat share** - share of pixels whose 3x3 neighbourhood barely varies;
* **edge density** - share of strong-gradient pixels, which is high for line
  art and text.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def features(image: np.ndarray) -> dict[str, float]:
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)

    q = (img >> 3).astype(np.int32)
    codes = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    counts = np.bincount(codes.ravel())
    top8 = float(np.sort(counts)[-8:].sum()) / codes.size

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (3, 3))
    var = cv2.blur(gray * gray, (3, 3)) - mean * mean
    flat = float(np.mean(var < 4.0))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edges = float(np.mean(np.hypot(gx, gy) > 200.0))
    return {"palette_top8": top8, "flat_share": flat, "edge_density": edges}


def classify_image(image: np.ndarray) -> tuple[str, dict[str, Any]]:
    """``("diagram" | "photo", features)``."""
    f = features(image)
    diagram = (f["palette_top8"] > 0.5 and f["flat_share"] > 0.4) or f["palette_top8"] > 0.75
    return ("diagram" if diagram else "photo"), f
