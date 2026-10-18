"""Perceptual metrics - does the output *look* good.

Two kinds live here, and the difference matters for how the paper may talk
about them:

**Optional published metrics.** LPIPS (reference-based), NIQE and CLIP-IQA
(no-reference). These are what the literature compares against. Each needs a
dependency or a pristine-statistics file; absent that, the metric reports
itself unavailable and leaves a gap. It is never silently swapped for a proxy.

**Built-in proxies.** Sharpness, noise sigma and colourfulness, plus a combined
``nr_quality`` index. These run everywhere, they are what the demo and the
tests use, and they behave the way no-reference quality scores behave - they
reward acutance and punish visible noise, so they keep climbing after fidelity
has plateaued, which is the H2 signature.

The proxies are a proxy. ARCHITECTURE.md section 8 already commits to saying
"no human perceptual study; perceptual claims rest on proxy metrics" - this
module is what that sentence is about, and ``nr_quality`` is a hand-written
formula, not a validated perceptual model. Report it as such.
"""

from __future__ import annotations

import numpy as np

_LPIPS_CACHE: dict[str, object] = {}


# --------------------------------------------------------------------------
# Built-in proxies
# --------------------------------------------------------------------------

def _luma(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 2:
        return a
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def sharpness(image: np.ndarray) -> float:
    """Acutance as the variance of the Laplacian, normalised by data range.

    Rises with genuine detail *and* with synthesised detail *and* with noise -
    which is exactly why it must be read next to :func:`noise_sigma` and never
    on its own.
    """
    import cv2

    luma = _luma(image).astype(np.float32)
    return float(cv2.Laplacian(luma, cv2.CV_32F).var() / (255.0**2) * 1e4)


def noise_sigma(image: np.ndarray) -> float:
    """Immerkaer's fast noise estimate on the **luma** channel, in image units.

    Convolves with a kernel orthogonal to smooth signal and takes a robust
    absolute average, so image structure contributes little and additive noise
    contributes almost all of it.

    Measured on luma, not per channel, because that is what a viewer sees. For
    independent noise of sigma in each of R, G and B, luma noise is
    ``sqrt(0.2126^2 + 0.7152^2 + 0.0722^2) = 0.75`` of sigma - so this returns
    about three quarters of the per-channel sigma, by design. Compare outputs
    against each other, not against a per-channel figure from the sensor model.
    """
    import cv2

    luma = _luma(image).astype(np.float32)
    h, w = luma.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(luma, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    # Interior only: the border response is contaminated by the reflected edge.
    # The 6 and the sqrt(pi/2) are Immerkaer's constants for this kernel.
    interior = np.abs(response[1:-1, 1:-1])
    return float(np.sqrt(np.pi / 2) * interior.sum() / (6.0 * (h - 2) * (w - 2)))


def colorfulness(image: np.ndarray) -> float:
    """Hasler & Suesstrunk colourfulness. Saturation-pushing arms score high."""
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 3:
        return 0.0
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def nr_quality(image: np.ndarray) -> float:
    """A no-reference index: detail rewarded, visible noise punished.

        nr_quality = log1p(sharpness) * exp(-noise_sigma / 6)

    Deliberately simple and fully visible, because it is a stand-in. It moves
    the way a no-reference score moves - a smooth, plausible, invented texture
    scores well - which makes it useful for demonstrating the H2 divergence and
    useless as evidence about human preference.
    """
    return float(np.log1p(sharpness(image)) * np.exp(-noise_sigma(image) / 6.0))


# --------------------------------------------------------------------------
# Optional published metrics
# --------------------------------------------------------------------------

def lpips_available() -> tuple[bool, str]:
    try:
        import lpips  # noqa: F401
        import torch  # noqa: F401

        return True, ""
    except ImportError as exc:
        return False, f"LPIPS needs torch + lpips (pip install sensorsim[learned]): {exc}"


def lpips(restored: np.ndarray, reference: np.ndarray, net: str = "alex") -> float:
    """Learned perceptual image patch similarity. Lower is better.

    Zhang et al., *The Unreasonable Effectiveness of Deep Features as a
    Perceptual Metric*, CVPR 2018.
    """
    ok, reason = lpips_available()
    if not ok:
        raise RuntimeError(reason)

    import lpips as lpips_lib  # type: ignore
    import torch  # type: ignore

    if net not in _LPIPS_CACHE:
        _LPIPS_CACHE[net] = lpips_lib.LPIPS(net=net)
    model = _LPIPS_CACHE[net]

    def to_tensor(x: np.ndarray):
        t = torch.from_numpy(np.asarray(x, dtype=np.float32).transpose(2, 0, 1))[None]
        return t / 127.5 - 1.0  # LPIPS expects [-1, 1]

    with torch.no_grad():
        value = model(to_tensor(restored), to_tensor(reference))  # type: ignore[operator]
    return float(value.item())


def niqe_available() -> tuple[bool, str]:
    """NIQE needs the authors' pristine multivariate-Gaussian parameters.

    Those are a data file, not a formula, and re-deriving them from a different
    pristine corpus produces a *different metric* that should not be reported
    under the same name. So NIQE is unavailable until that file is present.
    """
    try:
        from importlib.resources import files

        path = files("sensorsim.evaluate").joinpath("niqe_pristine.npz")
        if path.is_file():
            return True, ""
        return False, "NIQE needs sensorsim/evaluate/niqe_pristine.npz (authors' pristine MVG parameters)"
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
        return False, f"NIQE parameter file unavailable: {exc}"


def clip_iqa_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import clip  # noqa: F401

        return True, ""
    except ImportError as exc:
        return False, f"CLIP-IQA needs torch + clip: {exc}"


def compute(
    restored: np.ndarray,
    reference: np.ndarray | None = None,
    include_optional: bool = True,
) -> dict[str, float]:
    """Every perceptual metric that can run here, as a flat dict.

    Metrics that cannot run are simply absent from the dict - the evaluator
    turns that absence into an explicit unavailable row, so a gap in the table
    is visible rather than inferred.
    """
    out: dict[str, float] = {
        "sharpness": sharpness(restored),
        "noise_sigma": noise_sigma(restored),
        "colorfulness": colorfulness(restored),
        "nr_quality": nr_quality(restored),
    }
    if include_optional and reference is not None and lpips_available()[0]:
        out["lpips"] = lpips(restored, reference)
    return out


def unavailable() -> dict[str, str]:
    """Optional metrics that cannot run here, and why."""
    reasons: dict[str, str] = {}
    for name, check in (("lpips", lpips_available), ("niqe", niqe_available), ("clip_iqa", clip_iqa_available)):
        ok, reason = check()
        if not ok:
            reasons[name] = reason
    return reasons


HIGHER_IS_BETTER = {
    "sharpness": True,
    "noise_sigma": False,
    "colorfulness": True,
    "nr_quality": True,
    "lpips": False,
    "niqe": False,
    "clip_iqa": True,
}
