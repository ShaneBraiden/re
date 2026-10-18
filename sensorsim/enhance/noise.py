"""Per-pixel noise maps, and how they change as an image is processed.

This is the common thread of the NAG method (see :mod:`sensorsim.enhance`).
Every stage that brightens, filters or upsamples an image also changes how
much noise each pixel carries. If that is tracked, later stages can tell a
pixel whose fine structure is signal from one whose fine structure is noise.

**Exact map from the capture metadata.** The degrader's sensor model is
Poisson-Gaussian (``sensorsim.degrader.noise.apply_photon_noise``). For a
pixel whose gain-corrected linear value is ``u``, captured at photon scale
``alpha`` with binning ``b``, the variance in normalised units is::

    var(u) = g*u / (FW*b^2)                 shot noise, g = 1/alpha
           + (g*sigma_r / (span*b^2))^2     read noise, in DN over the DN span
           + g^2 * dark_e / (FW^2 * b^2)    dark current
           + (prnu*u)^2                     fixed-pattern gain error
           + (g*q)^2 / 12                   ADC quantisation, q = 1/(2^bits - 1)

White balance multiplies a channel's standard deviation by its gain. A tone
curve ``T`` maps a standard deviation ``s`` at level ``x`` through the symmetric
secant ``(T(x+s) - T(x-s)) / 2``. That is the first-order delta method when
``s`` is small, and it stays bounded near black, where the slope of the sRGB
curve is steep and the delta method overshoots.

**Blind map when there is no metadata.** An uploaded file has no capture
record, so its noise level function is estimated from the image: Immerkaer's
operator gives a local noise estimate, and a low percentile of that estimate
within each intensity bin rejects texture. The result is labelled *estimated*
wherever it is reported.

**Demosaicing** averages neighbouring samples, which lowers the variance. On an
RGGB mosaic, R and B are measured at 1/4 of the sites. Half of the remaining
sites average two samples and a quarter average four, so the mean variance is
``(1/4)(1) + (1/2)(1/2) + (1/4)(1/4) = 9/16`` of the per-sample variance, and
the std factor is 3/4. G is measured at half the sites, and the edge-aware
demosaic fills the rest from two neighbours, so its variance factor is
``1/2 + (1/2)(1/2) = 3/4`` and its std factor is about 0.87. With these factors, the predicted map on flat fields
across the sweep lands within a few percent of the measured std
(``tests/test_nag.py``). The exception is the darkest 10-bit conditions. There
the ADC step exceeds the read noise, so quantisation stops behaving like noise
and the ``q^2/12`` term underestimates it by about 10%.
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from sensorsim.restore.classical import tone_curve

#: Std reduction from demosaicing an RGGB mosaic, per output channel (see above).
DEMOSAIC_STD_FACTOR = np.array([0.75, np.sqrt(0.75), 0.75], dtype=np.float32)

#: Immerkaer's 3x3 operator. Orthogonal to locally linear signal; its response to
#: white noise of variance s^2 has variance 36 s^2.
IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def variance_linear(
    u: np.ndarray,
    *,
    alpha: float,
    binning: int,
    full_well: float,
    sigma_read_dn: float,
    span_dn: float,
    bit_depth: int,
    dark_current_e: float = 0.0,
    prnu: float = 0.0,
) -> np.ndarray:
    """Noise variance of gain-corrected linear signal ``u`` (see module docstring)."""
    g = 1.0 / max(alpha, 1e-6)
    b2 = float(max(binning, 1)) ** 2
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, None)
    q = 1.0 / max(2.0**bit_depth - 1.0, 1.0)
    var = (
        g * u / (full_well * b2)
        + (g * sigma_read_dn / (span_dn * b2)) ** 2
        + g * g * dark_current_e / (full_well * full_well * b2)
        + (prnu * u) ** 2
        + (g * q) ** 2 / 12.0
    )
    return var


def sigma_map_from_context(
    rgb_linear: np.ndarray,
    degradation: dict[str, Any],
    sensor: dict[str, float],
    wb_gains: tuple[float, float, float],
    span_dn: float,
) -> np.ndarray:
    """Per-pixel, per-channel noise std of a gained, white-balanced linear frame.

    ``rgb_linear`` is what the classical front end produces (gain, demosaic,
    white balance) at capture resolution. ``degradation`` and ``sensor`` are the
    dicts a :class:`~sensorsim.restore.RestoreContext` carries.
    """
    gains = np.asarray(wb_gains, dtype=np.float64)
    pre_wb = np.clip(rgb_linear, 0.0, None) / gains
    var = variance_linear(
        pre_wb,
        alpha=float(degradation.get("alpha", 1.0)),
        binning=int(degradation.get("binning", 1)),
        full_well=float(sensor.get("full_well", 15000.0)),
        sigma_read_dn=float(sensor.get("sigma_read_dn", degradation.get("sigma_r_dn", 0.0))),
        span_dn=span_dn,
        bit_depth=int(degradation.get("bit_depth", 14)),
        dark_current_e=float(sensor.get("dark_current_e", 0.0)),
        prnu=float(sensor.get("prnu", 0.0)),
    )
    return (np.sqrt(var) * gains * DEMOSAIC_STD_FACTOR).astype(np.float32)


def propagate(curve: Callable[[np.ndarray], np.ndarray], x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Noise std after a pointwise monotone curve, by the symmetric secant."""
    hi = curve(np.clip(x + sigma, 0.0, 1.0))
    lo = curve(np.clip(x - sigma, 0.0, 1.0))
    return (np.abs(hi - lo) * 0.5).astype(np.float32)


def propagate_tone(x_linear: np.ndarray, sigma_linear: np.ndarray, contrast: float = 0.0) -> np.ndarray:
    """Noise std after the sRGB tone curve the classical front end applies."""
    return propagate(lambda v: tone_curve(v, contrast=contrast), x_linear, sigma_linear)


def luma_sigma(sigma_rgb: np.ndarray) -> np.ndarray:
    """Std of Rec.709 luma, for independent per-channel noise."""
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.sqrt(np.sum((sigma_rgb.astype(np.float32) * w) ** 2, axis=-1))


#: On white Gaussian noise, the 25th percentile of the 9x9 box-averaged
#: Immerkaer estimate sits at 0.894 of the true std (measured on 512^2 fields;
#: it is the price of rejecting texture with a low percentile). Dividing by it
#: makes the estimator unbiased on pure noise.
BLIND_PERCENTILE = 25.0
BLIND_WINDOW = 9
BLIND_BIAS = 0.894


def _channel_nlf(channel: np.ndarray, bins: int, floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = BLIND_WINDOW
    response = np.abs(cv2.filter2D(channel, cv2.CV_32F, IMMERKAER, borderType=cv2.BORDER_REFLECT))
    # E|N(0, 36 s^2)| = 6 s sqrt(2/pi), so s = |r| sqrt(pi/2) / 6.
    local = cv2.blur(response * np.float32(np.sqrt(np.pi / 2.0) / 6.0), (w, w))
    level = cv2.blur(channel, (w, w))

    edges = np.linspace(0.0, 1.0, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    nlf = np.full(bins, np.nan, dtype=np.float64)
    interior = (slice(w, -w), slice(w, -w)) if min(channel.shape) > 4 * w else (slice(None),) * 2
    lv, lc = level[interior].ravel(), local[interior].ravel()
    for i in range(bins):
        sel = (lv >= edges[i]) & (lv < edges[i + 1] if i < bins - 1 else lv <= edges[i + 1])
        if sel.sum() >= 64:
            nlf[i] = np.percentile(lc[sel], BLIND_PERCENTILE) / BLIND_BIAS

    known = np.isfinite(nlf)
    nlf = np.interp(centres, centres[known], nlf[known]) if known.any() else np.full(bins, floor)
    nlf = np.maximum(nlf, floor)
    return np.interp(level, centres, nlf).astype(np.float32), centres, nlf


def blind_sigma_map(
    image: np.ndarray,
    bins: int = 16,
    floor: float = 0.5 / 255.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a per-channel noise level function from the image alone.

    Returns the per-pixel, per-channel std map, plus the fitted curves (bin
    centres and std per bin, per channel) so a report can show what was
    estimated.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    if img.ndim == 2:
        sigma, centres, nlf = _channel_nlf(img, bins, floor)
        return sigma, {"centres": centres.tolist(), "sigma": [nlf.tolist()], "estimated": True}
    maps, curves = [], []
    for c in range(img.shape[-1]):
        m, centres, nlf = _channel_nlf(np.ascontiguousarray(img[..., c]), bins, floor)
        maps.append(m)
        curves.append(nlf.tolist())
    return np.stack(maps, axis=-1), {"centres": centres.tolist(), "sigma": curves, "estimated": True}


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------

def apply_curves_np(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Zero-DCE curves on numpy arrays. ``r`` is (H, W, 3k), ``x`` is (H, W, 3)."""
    out = np.asarray(x, dtype=np.result_type(x, np.float32))
    for n in range(r.shape[-1] // 3):
        r_n = r[..., 3 * n:3 * n + 3]
        out = out + r_n * (out * out - out)
    return out


def curve_slope_np(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """d LE / dx for the iterated curve, in closed form.

    Each iteration ``LE_n = LE_{n-1} + r_n (LE_{n-1}^2 - LE_{n-1})`` has
    derivative ``1 + r_n (2 LE_{n-1} - 1)`` with respect to its input, and the
    chain rule multiplies them. ``r`` is treated as fixed with respect to the
    pixel's own value - the curve maps vary smoothly in space, so that is the
    first-order term.
    """
    out = np.asarray(x, dtype=np.result_type(x, np.float32))
    slope = np.ones_like(out)
    for n in range(r.shape[-1] // 3):
        r_n = r[..., 3 * n:3 * n + 3]
        slope = slope * (1.0 + r_n * (2.0 * out - 1.0))
        out = out + r_n * (out * out - out)
    return slope


def curve_slope_torch(x, r):
    """Torch version of :func:`curve_slope_np` for (N, 3, H, W) / (N, 3k, H, W)."""
    import torch

    out = x
    slope = torch.ones_like(x)
    for r_n in torch.split(r, 3, dim=1):
        slope = slope * (1.0 + r_n * (2.0 * out - 1.0))
        out = out + r_n * (out * out - out)
    return slope, out
