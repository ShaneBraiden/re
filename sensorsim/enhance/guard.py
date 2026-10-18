"""G-ESRGAN: generative super-resolution, kept to what the measurement supports.

Real-ESRGAN adds high-frequency detail learned from training images. Where the
input contains real structure, that detail mostly sharpens it. Where the input
is noise or flat, the same mechanism invents texture, edges and letterforms.
The paper's H2/H3 results measure exactly this as substitution. The guard does
two things to the detail a generator adds, ``d = SR(y) - U(y)``:

1. **A Wiener-style gate.** The 4-neighbour Laplacian has kernel energy 20, so
   on white noise of std ``sigma`` its response has variance ``20 sigma^2``.
   Over a small window, the high-frequency energy of the measurement ``E_y``
   therefore splits into signal plus about ``20 sigma^2`` of noise, and::

       gamma = clip((E_y - 20 sigma^2) / (E_y + eps), 0, 1)

   is the fraction of local detail energy that is signal - the Wiener gain. The
   added detail is scaled by ``gamma``: kept on real edges, removed where the
   measurement holds nothing above the noise. ``sigma`` comes from
   :mod:`sensorsim.enhance.noise`, so the gate tracks every stage before it.

2. **Low-pass back-projection** (after Irani and Peleg, 1991). With ``D`` the
   known capture operator (PSF blur, then area downsampling by the binning
   factor)::

       x <- x + lambda * U(G * (y - D(x)))

   pulls the output back toward consistency with what was measured. The
   residual is smoothed by ``G`` first, because ``y`` is noisy and an
   unsmoothed back-projection would copy that noise straight into ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from sensorsim.enhance.noise import luma_sigma

LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
#: Sum of squared Laplacian taps: its gain on white-noise variance.
LAPLACIAN_NOISE_GAIN = float((LAPLACIAN**2).sum())


def _luma(img: np.ndarray) -> np.ndarray:
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def resize(img: np.ndarray, shape: tuple[int, int], down: bool | None = None) -> np.ndarray:
    h, w = shape
    if img.shape[:2] == (h, w):
        return img
    shrinking = down if down is not None else (h * w < img.shape[0] * img.shape[1])
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def wiener_gate(y: np.ndarray, sigma: np.ndarray, window: int = 7, eps: float = 1e-6,
                smooth: float = 1.0) -> np.ndarray:
    """Per-pixel fraction of local high-frequency energy that is signal."""
    luma = _luma(np.asarray(y, dtype=np.float32))
    s = luma_sigma(sigma) if sigma.ndim == 3 else sigma.astype(np.float32)
    response = cv2.filter2D(luma, cv2.CV_32F, LAPLACIAN, borderType=cv2.BORDER_REFLECT)
    energy = cv2.blur(response * response, (window, window))
    noise = LAPLACIAN_NOISE_GAIN * cv2.blur(s * s, (window, window))
    gamma = np.clip((energy - noise) / (energy + eps), 0.0, 1.0)
    if smooth > 0:
        gamma = cv2.GaussianBlur(gamma, (0, 0), smooth)
    return gamma.astype(np.float32)


def make_capture_operator(shape_lr: tuple[int, int], psf: np.ndarray | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """``D``: blur with the capture PSF (if known), then area-downsample."""

    def down(x: np.ndarray) -> np.ndarray:
        if psf is not None and psf.size > 1:
            x = cv2.filter2D(x, -1, psf.astype(np.float32), borderType=cv2.BORDER_REFLECT)
        return resize(x, shape_lr, down=True)

    return down


def back_project(
    x: np.ndarray,
    y: np.ndarray,
    down: Callable[[np.ndarray], np.ndarray],
    iterations: int = 3,
    step: float = 1.0,
    residual_smooth: float = 1.0,
) -> tuple[np.ndarray, list[float]]:
    """Low-pass back-projection. Returns the result and ||y - D(x)|| per step."""
    history: list[float] = []
    for _ in range(max(0, iterations) + 1):
        residual = y - down(x)
        history.append(float(np.sqrt(np.mean(residual**2))))
        if len(history) > iterations:
            break
        if residual_smooth > 0:
            residual = cv2.GaussianBlur(residual, (0, 0), residual_smooth)
        x = np.clip(x + step * resize(residual, x.shape[:2], down=False), 0.0, 1.0)
    return x, history


@dataclass
class GuardResult:
    image: np.ndarray
    gate: np.ndarray
    consistency: list[float]
    unguarded: np.ndarray


def guarded_sr(
    y: np.ndarray,
    sigma: np.ndarray,
    sr: Callable[[np.ndarray], np.ndarray],
    target_shape: tuple[int, int],
    psf: np.ndarray | None = None,
    gate: bool = True,
    bp_iterations: int = 3,
) -> GuardResult:
    """Super-resolve ``y`` to ``target_shape`` with the gate and back-projection.

    ``sr`` takes a float RGB image in [0, 1] and returns a larger one; it is
    resized to ``target_shape`` here, so any model scale works.
    """
    y = np.asarray(y, dtype=np.float32)
    x_sr = resize(np.clip(sr(y), 0.0, 1.0).astype(np.float32), target_shape)
    base = resize(y, target_shape, down=False)

    if gate:
        g = resize(wiener_gate(y, sigma), target_shape, down=False)
        x = base + g[..., None] * (x_sr - base)
    else:
        g = np.ones(target_shape, np.float32)
        x = x_sr

    history: list[float] = []
    if bp_iterations > 0:
        x, history = back_project(x, y, make_capture_operator(y.shape[:2], psf), iterations=bp_iterations)
    return GuardResult(np.clip(x, 0.0, 1.0), g, history, x_sr)
