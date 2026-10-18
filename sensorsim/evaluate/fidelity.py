"""Fidelity metrics - is the output *true* to the scene.

Reference-based, computed against the clean ground-truth rendering. These are
the axis that H2 predicts will plateau or decline once processing crosses the
knee, so they are computed on exactly the same outputs as the perceptual
metrics - never on separately tuned renders.

PSNR and SSIM are implemented here rather than pulled from scikit-image so the
core install stays small and the exact window, sigma and boundary handling are
visible in the repository. The constants follow Wang et al. 2004.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

SSIM_K1 = 0.01
SSIM_K2 = 0.03
SSIM_SIGMA = 1.5
SSIM_WINDOW = 11


def _as_float(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float64)


def _check_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = _as_float(a), _as_float(b)
    if a.shape != b.shape:
        raise ValueError(
            f"metric inputs must match in shape, got {a.shape} and {b.shape}. "
            "A pipeline that resampled to the wrong size should have been caught at Stage 2."
        )
    return a, b


def mse(restored: np.ndarray, reference: np.ndarray) -> float:
    a, b = _check_pair(restored, reference)
    return float(np.mean((a - b) ** 2))


def psnr(restored: np.ndarray, reference: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB.

    Returns ``inf`` for identical inputs, which is correct but worth knowing
    about: aggregate PSNR with the median, or drop the clean-vs-clean row, or a
    single perfect match will swallow a mean.
    """
    error = mse(restored, reference)
    if error <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / error))


def _gaussian_filter(image: np.ndarray) -> np.ndarray:
    truncate = (SSIM_WINDOW - 1) / 2 / SSIM_SIGMA
    return ndimage.gaussian_filter(image, sigma=SSIM_SIGMA, truncate=truncate, mode="reflect")


def ssim(
    restored: np.ndarray,
    reference: np.ndarray,
    data_range: float = 255.0,
    return_map: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Structural similarity, Gaussian-weighted, averaged over channels."""
    a, b = _check_pair(restored, reference)
    if a.ndim == 3:
        per_channel = [ssim(a[..., c], b[..., c], data_range, return_map) for c in range(a.shape[2])]
        if return_map:
            values = [v for v, _ in per_channel]  # type: ignore[misc]
            maps = np.stack([m for _, m in per_channel], axis=-1)  # type: ignore[misc]
            return float(np.mean(values)), maps
        return float(np.mean(per_channel))  # type: ignore[arg-type]

    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2

    mu_a = _gaussian_filter(a)
    mu_b = _gaussian_filter(b)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    # Unbiased-ish local (co)variance: E[x^2] - E[x]^2 under the same window.
    sigma_aa = _gaussian_filter(a * a) - mu_aa
    sigma_bb = _gaussian_filter(b * b) - mu_bb
    sigma_ab = _gaussian_filter(a * b) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_aa + mu_bb + c1) * (sigma_aa + sigma_bb + c2)
    ssim_map = numerator / denominator

    # Trim the border where the window ran off the edge and the estimate is
    # dominated by reflected pixels.
    pad = (SSIM_WINDOW - 1) // 2
    valid = ssim_map[pad:-pad, pad:-pad] if min(ssim_map.shape) > 2 * pad else ssim_map
    value = float(np.mean(valid))
    return (value, ssim_map) if return_map else value


def delta_e_76(restored: np.ndarray, reference: np.ndarray) -> float:
    """Mean CIE76 colour difference in CIELAB.

    PSNR and SSIM are largely insensitive to a uniform colour cast, which a
    white-balance failure produces. This catches it.
    """
    import cv2

    a, b = _check_pair(restored, reference)
    lab_a = cv2.cvtColor((a / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    lab_b = cv2.cvtColor((b / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    return float(np.mean(np.sqrt(np.sum((lab_a - lab_b) ** 2, axis=-1))))


def compute(restored: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Every fidelity metric, as a flat dict."""
    return {
        "psnr": psnr(restored, reference),
        "ssim": ssim(restored, reference),  # type: ignore[dict-item]
        "delta_e": delta_e_76(restored, reference),
    }


#: Direction of each metric, for tables and figures.
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "delta_e": False}
