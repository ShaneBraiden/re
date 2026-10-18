"""Spatial sampling and quantisation - the sensor's discrete side.

Binning emulates a smaller effective pixel pitch: charge from a ``b x b`` block
of same-colour photosites is combined before readout. The CFA constrains how:
you can only combine pixels that share a colour, so a factor of ``b`` on a Bayer
grid means combining ``b x b`` samples *within* each of the four sub-planes,
which is a stride of ``2b`` on the mosaic.

Binning is applied in this model by averaging the normalised signal while
telling the noise model the well is ``b^2`` times deeper (see
``well_multiplier`` in :func:`sensorsim.degrader.noise.apply_photon_noise`).
That is the same thing as summing charge, but it keeps every array in the same
normalised units, and it preserves the property that matters: binning buys a
factor of ``b`` in SNR and pays for it in resolution.
"""

from __future__ import annotations

import numpy as np

from sensorsim.io.raw import bayer_to_planes, planes_to_bayer


def bin_bayer(raw: np.ndarray, factor: int, pattern: str = "RGGB") -> np.ndarray:
    """Bin a mosaic by ``factor`` within each colour plane.

    Returns a mosaic of shape ``(H // (2*factor) * 2, W // (2*factor) * 2)`` with
    the same CFA pattern. Edge pixels that do not fill a whole block are dropped
    rather than padded - padding would invent signal at the frame border.
    """
    if factor <= 1:
        return np.asarray(raw, dtype=np.float32)

    planes = bayer_to_planes(np.asarray(raw, dtype=np.float64), pattern)
    binned: dict[str, np.ndarray] = {}
    for key, plane in planes.items():
        h = plane.shape[0] // factor * factor
        w = plane.shape[1] // factor * factor
        if h == 0 or w == 0:
            raise ValueError(f"binning factor {factor} exceeds plane size {plane.shape}")
        cropped = plane[:h, :w]
        binned[key] = cropped.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return planes_to_bayer(binned, pattern)


def quantize(
    raw: np.ndarray,
    bit_depth: int,
    black_level_frac: float = 0.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Requantise normalised signal to ``bit_depth`` bits.

    The ADC steps live in DN, so the normalised signal is scaled to the code
    range, rounded, and scaled back. ``black_level_frac`` reserves the bottom of
    the range the way a real sensor does, so the usable code count is smaller
    than ``2^bit_depth``.

    ``floor`` is the lowest representable value. Pass the negative pedestal
    headroom (see :func:`sensorsim.degrader.noise.apply_photon_noise`) so read
    noise below black level survives quantisation instead of being rectified.
    """
    if bit_depth <= 0 or bit_depth > 32:
        raise ValueError(f"bit_depth must be in 1..32, got {bit_depth}")
    raw = np.asarray(raw, dtype=np.float64)
    levels = (2**bit_depth - 1) * (1.0 - black_level_frac)
    return (np.round(np.clip(raw, floor, 1.0) * levels) / levels).astype(np.float32)


def upsample_to(image: np.ndarray, shape: tuple[int, int], order: int = 1) -> np.ndarray:
    """Resize an image (2-D or H x W x C) to ``shape``.

    Used by restoration pipelines to bring a binned frame back to reference
    resolution so the evaluator can compare like with like. ``order=1`` is
    bilinear - deliberately dumb, so any resolution a pipeline recovers is
    attributable to the pipeline and not to a clever resampler.
    """
    import cv2

    h, w = shape
    if image.shape[:2] == (h, w):
        return image
    interp = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 3: cv2.INTER_CUBIC}[order]
    return cv2.resize(image, (w, h), interpolation=interp)


def effective_resolution(shape: tuple[int, int], factor: int) -> tuple[int, int]:
    """Shape a mosaic will have after :func:`bin_bayer` at ``factor``."""
    if factor <= 1:
        return shape
    h, w = shape
    return ((h // 2 // factor) * 2, (w // 2 // factor) * 2)
