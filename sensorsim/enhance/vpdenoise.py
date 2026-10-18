"""VP-denoise: a guided filter whose strength follows the propagated noise map.

He, Sun and Tang's guided filter (ECCV 2010 / TPAMI 2013) fits, in every
window, a linear model ``q = a I + b`` with ``a = cov / (var + eps)``. A window
whose variance is much larger than ``eps`` is kept (``a`` near 1); one whose
variance is about ``eps`` or less is flattened toward its mean (``a`` near 0).
That makes ``eps`` a noise variance threshold, and the usual practice of one
global ``eps`` assumes every pixel has the same noise.

After a curve has brightened shadows and a tone curve has stretched them, the
noise is far from uniform. Here ``eps`` is a map, ``(k * sigma)^2``, built
from the noise std that :mod:`sensorsim.enhance.noise` carried through each
stage. The filter is strong exactly where processing amplified noise and
nearly off where it did not, with no global strength to tune.
"""

from __future__ import annotations

import cv2
import numpy as np


def _box(x: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    return cv2.blur(x, (k, k), borderType=cv2.BORDER_REFLECT)


def guided_filter_varying(
    image: np.ndarray,
    sigma: np.ndarray,
    radius: int = 3,
    k: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Self-guided filter per channel with a per-pixel ``eps = (k sigma)^2``.

    Returns the filtered image and the noise std left in it. For a self-guided
    window the noise passes with weight about ``mean(a)``, and the averaged
    ``b`` term carries the remainder reduced by the window's pixel count.
    """
    img = np.asarray(image, dtype=np.float32)
    sig = np.asarray(sigma, dtype=np.float32)
    if sig.shape != img.shape:
        sig = np.broadcast_to(sig[..., None] if sig.ndim == 2 else sig, img.shape).astype(np.float32)

    out = np.empty_like(img)
    residual = np.empty_like(img)
    n = float((2 * radius + 1) ** 2)
    for c in range(img.shape[-1]):
        p = img[..., c]
        eps = _box((k * sig[..., c]) ** 2, radius)
        mean = _box(p, radius)
        var = np.maximum(_box(p * p, radius) - mean * mean, 0.0)
        a = var / (var + eps + 1e-12)
        b = mean - a * mean
        a_bar, b_bar = _box(a, radius), _box(b, radius)
        out[..., c] = a_bar * p + b_bar
        residual[..., c] = sig[..., c] * np.sqrt(a_bar**2 + (1.0 - a_bar) ** 2 / n)
    return np.clip(out, 0.0, 1.0), residual
