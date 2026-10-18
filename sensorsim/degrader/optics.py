"""Point-spread functions and the optical half of the forward model.

Two PSF families cover the deficits the study cares about:

``gaussian``
    A stand-in for aberration and defocus in a cheap lens. ``r_psf`` is the
    standard deviation in pixels.
``airy``
    Diffraction through a small aperture - the limit a phone camera actually
    runs into. ``r_psf`` is the radius of the first dark ring in pixels.

Applying a PSF to a *mosaiced* frame needs care. A lens blurs the continuous
optical image before the colour filter array samples it, so convolving the raw
Bayer grid directly - which mixes red into green and back - is wrong. Each
colour channel is instead blurred at its own sample positions by normalised
convolution: blur the sparse channel and the sample mask with the same kernel
and divide. That reconstructs the blurred continuous channel where it was
sampled, without ever inventing a colour that was not measured.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, special

from sensorsim.io.raw import cfa_masks

# Radius of the first zero of the Airy pattern, in units of its argument.
AIRY_FIRST_ZERO = 3.8317059702075125


def gaussian_psf(sigma: float, size: int | None = None) -> np.ndarray:
    """Normalised 2-D Gaussian kernel with standard deviation ``sigma`` px."""
    if sigma <= 0:
        return np.ones((1, 1), dtype=np.float64)
    size = size or max(3, int(2 * np.ceil(3 * sigma) + 1))
    size += (size + 1) % 2  # force odd so the kernel has a centre pixel
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return k / k.sum()


def airy_psf(first_zero_radius: float, size: int | None = None) -> np.ndarray:
    """Normalised Airy disk whose first dark ring sits at ``first_zero_radius`` px."""
    if first_zero_radius <= 0:
        return np.ones((1, 1), dtype=np.float64)
    size = size or max(3, int(2 * np.ceil(3 * first_zero_radius) + 1))
    size += (size + 1) % 2
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    r = np.hypot(xx, yy)
    x = AIRY_FIRST_ZERO * r / first_zero_radius
    k = np.ones_like(x)
    nz = x > 1e-9
    k[nz] = (2.0 * special.j1(x[nz]) / x[nz]) ** 2
    return k / k.sum()


def make_psf(kind: str, radius: float, size: int | None = None) -> np.ndarray:
    """Dispatch to a PSF family by name."""
    if kind == "gaussian":
        return gaussian_psf(radius, size)
    if kind == "airy":
        return airy_psf(radius, size)
    raise ValueError(f"unknown PSF kind {kind!r}; expected one of: gaussian, airy")


def apply_psf_mosaic(raw: np.ndarray, psf: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Blur a mosaiced frame per colour channel by normalised convolution.

    Each channel is blurred where it was sampled, so red never leaks into a green
    site. The division by the blurred mask is what makes this an interpolating
    blur rather than a darkening one - without it, every channel would lose the
    energy that fell on the other channels' pixels.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if psf.shape == (1, 1):
        return raw.astype(np.float32)

    masks = cfa_masks(raw.shape, pattern)
    out = np.zeros_like(raw)
    for channel, mask in masks.items():
        m = mask.astype(np.float64)
        num = ndimage.convolve(raw * m, psf, mode="reflect")
        den = ndimage.convolve(m, psf, mode="reflect")
        blurred = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        out[mask] = blurred[mask]
    return out.astype(np.float32)


def apply_psf_rgb(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Blur a demosaiced image. Used by the validation harness, not the degrader."""
    image = np.asarray(image, dtype=np.float64)
    if psf.shape == (1, 1):
        return image.astype(np.float32)
    if image.ndim == 2:
        return ndimage.convolve(image, psf, mode="reflect").astype(np.float32)
    out = np.stack([ndimage.convolve(image[..., c], psf, mode="reflect") for c in range(image.shape[-1])], axis=-1)
    return out.astype(np.float32)


def mtf_from_psf(psf: np.ndarray, n_samples: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged modulation transfer function of a PSF.

    Returns ``(frequency, mtf)`` with frequency in cycles per pixel, running from
    0 to Nyquist (0.5).
    """
    psf = np.asarray(psf, dtype=np.float64)
    if psf.shape == (1, 1):
        f = np.linspace(0.0, 0.5, n_samples)
        return f, np.ones_like(f)

    pad = 256
    padded = np.zeros((pad, pad))
    h, w = psf.shape
    y0, x0 = (pad - h) // 2, (pad - w) // 2
    padded[y0 : y0 + h, x0 : x0 + w] = psf

    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    mtf2d = np.abs(otf)
    mtf2d /= mtf2d.max()

    cy, cx = pad // 2, pad // 2
    yy, xx = np.mgrid[:pad, :pad]
    r = np.hypot(yy - cy, xx - cx)
    freq_per_bin = 1.0 / pad  # cycles per pixel per radial bin

    max_bin = int(0.5 / freq_per_bin)
    radial = np.zeros(max_bin)
    for i in range(max_bin):
        sel = (r >= i) & (r < i + 1)
        radial[i] = mtf2d[sel].mean() if sel.any() else np.nan

    freqs = np.arange(max_bin) * freq_per_bin
    good = ~np.isnan(radial)
    return freqs[good], radial[good]


def mtf50(freqs: np.ndarray, mtf: np.ndarray) -> float:
    """Frequency where the MTF crosses 0.5, in cycles per pixel.

    The standard single-number summary of how much resolution an optical path
    retains. Linear interpolation across the crossing; NaN if it never crosses.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    mtf = np.asarray(mtf, dtype=np.float64)
    below = np.nonzero(mtf < 0.5)[0]
    if below.size == 0:
        return float(freqs[-1])
    i = int(below[0])
    if i == 0:
        return float(freqs[0])
    x0, x1 = freqs[i - 1], freqs[i]
    y0, y1 = mtf[i - 1], mtf[i]
    if y0 == y1:
        return float(x0)
    return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))


def slanted_edge_mtf(patch: np.ndarray, oversample: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """MTF from a slanted-edge patch, in the spirit of ISO 12233.

    A near-vertical edge tilted a few degrees lets each image row sample the edge
    at a different sub-pixel phase; pooling the rows gives a super-resolved edge
    spread function whose derivative is the line spread function.

    Parameters
    ----------
    patch:
        2-D grayscale patch containing one near-vertical, high-contrast edge.

    Returns
    -------
    ``(frequency, mtf)`` in cycles per pixel.
    """
    patch = np.asarray(patch, dtype=np.float64)
    if patch.ndim != 2 or min(patch.shape) < 8:
        raise ValueError(f"slanted-edge patch must be 2-D and at least 8x8, got {patch.shape}")

    h, w = patch.shape
    # Per-row edge location by centroid of the row derivative.
    x = np.arange(w - 1) + 0.5
    centroids = np.full(h, np.nan)
    for r in range(h):
        d = np.abs(np.diff(patch[r]))
        total = d.sum()
        if total > 1e-9:
            centroids[r] = float((d * x).sum() / total)

    rows = np.nonzero(~np.isnan(centroids))[0]
    if rows.size < 4:
        raise ValueError("no detectable edge in patch")
    slope, intercept = np.polyfit(rows.astype(float), centroids[rows], 1)

    # Project every pixel onto the edge-normal axis and bin at sub-pixel spacing.
    yy, xx = np.mgrid[:h, :w]
    distance = xx - (slope * yy + intercept)
    bins = np.round(distance * oversample).astype(int)
    lo, hi = bins.min(), bins.max()
    esf = np.zeros(hi - lo + 1)
    counts = np.zeros_like(esf)
    np.add.at(esf, bins.ravel() - lo, patch.ravel())
    np.add.at(counts, bins.ravel() - lo, 1.0)
    valid = counts > 0
    esf = esf[valid] / counts[valid]
    if esf.size < 8:
        raise ValueError("edge spread function too short to transform")

    lsf = np.diff(esf)
    lsf *= np.hanning(lsf.size)  # taper so the FFT sees no step at the ends
    total = np.abs(lsf).sum()
    if total < 1e-12:
        raise ValueError("degenerate line spread function - is the patch flat?")

    spectrum = np.abs(np.fft.rfft(lsf))
    spectrum /= spectrum[0]
    freqs = np.fft.rfftfreq(lsf.size, d=1.0 / oversample)

    keep = freqs <= 0.5
    return freqs[keep], spectrum[keep]
