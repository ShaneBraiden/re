"""Classical restoration arms - the ones that need no weights.

These four run anywhere the core dependencies are installed, which makes them
the backbone of the testbench: a full sweep, a metrics table and every figure
can be produced before a single model checkpoint is downloaded.

``classical_isp``
    The control condition. Black level, digital gain, demosaic, white balance,
    mild denoise, tone curve. Everything else is measured against this.
``nlm_denoise``
    Non-local means on top of the ISP. Real denoising, no learned prior - the
    honest midpoint between the control and the learned bank.
``curve_tone``
    Iterative light-enhancement curves with analytically estimated
    coefficients, in the spirit of Zero-DCE but *not* Zero-DCE: no network, no
    weights. Named for what it is so nobody cites it as the published method.
``rl_deconv``
    Richardson-Lucy deconvolution against the recorded PSF. The principled way
    to invert blur, and a useful contrast with generative detail synthesis:
    it can only redistribute measured energy, never invent new detail.

A note on exposure. Every arm applies digital gain of ``1/alpha`` from the
degradation record. That is metadata a real camera has about its own capture,
and matching brightness across arms is what makes the comparison about the
algorithms rather than about auto-exposure. ``gain_mode="auto"`` estimates it
from the frame instead, for real captures where the true scale is unknown.
"""

from __future__ import annotations

import cv2
import numpy as np

from sensorsim.degrader.optics import make_psf
from sensorsim.degrader.sampling import upsample_to
from sensorsim.io.raw import RawImage
from sensorsim.restore.base import Pipeline, RestoreContext, register

# OpenCV names Bayer codes from a different corner than the sensor convention
# does, so RGGB is COLOR_BayerBG2*. Verified by round-tripping a pure-red frame
# in tests/test_pipelines.py - do not "fix" these to match the letters.
BAYER_CODES = {
    "RGGB": cv2.COLOR_BayerBG2BGR_EA,
    "BGGR": cv2.COLOR_BayerRG2BGR_EA,
    "GRBG": cv2.COLOR_BayerGB2BGR_EA,
    "GBRG": cv2.COLOR_BayerGR2BGR_EA,
}


def demosaic(raw: RawImage) -> np.ndarray:
    """Edge-aware demosaic to linear float RGB in [0, 1].

    OpenCV demosaics integers, so the frame is taken to 16-bit first. That is
    not a loss worth worrying about: the deepest ADC in the study is 14-bit.
    """
    code = BAYER_CODES.get(raw.cfa_pattern)
    if code is None:
        raise ValueError(f"no demosaic code for CFA pattern {raw.cfa_pattern!r}")
    as_u16 = np.clip(raw.data, 0.0, 1.0)
    as_u16 = (as_u16 * 65535.0).round().astype(np.uint16)
    bgr = cv2.demosaicing(as_u16, code).astype(np.float32) / 65535.0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def apply_white_balance(rgb: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    """Neutralise the illuminant with the as-shot gains."""
    return np.clip(rgb * np.asarray(gains, dtype=np.float32), 0.0, 1.0)


def estimate_gain(raw: RawImage, meta: RestoreContext, mode: str = "recorded") -> float:
    """How much digital gain to apply before demosaicing."""
    if mode == "recorded":
        return 1.0 / max(meta.alpha, 1e-6)
    if mode == "auto":
        # Put the 99.5th percentile near white - a plain auto-exposure rule.
        p = float(np.percentile(np.clip(raw.data, 0.0, None), 99.5))
        return float(np.clip(0.9 / max(p, 1e-6), 1.0, 200.0))
    if mode == "none":
        return 1.0
    raise ValueError(f"unknown gain_mode {mode!r}; expected recorded, auto or none")


def tone_curve(linear: np.ndarray, gamma: float = 2.4, contrast: float = 0.0) -> np.ndarray:
    """Linear light to display-referred sRGB, with optional S-curve contrast."""
    lin = np.clip(linear, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1.0 / gamma) - 0.055)
    if contrast:
        srgb = np.clip(srgb + contrast * np.sin(2.0 * np.pi * srgb) / (2.0 * np.pi), 0.0, 1.0)
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


class _ClassicalBase(Pipeline):
    """Shared front end: gain, demosaic, white balance, resample."""

    def linear_rgb(self, raw: RawImage, meta: RestoreContext) -> np.ndarray:
        gain = estimate_gain(raw, meta, self.params.get("gain_mode", "recorded"))
        # Clamping at zero happens here, after the gain, not in the sensor model.
        # A real ISP clips negative excursions; the *model* must not, or read
        # noise gets rectified and the darkest frames come out biased bright.
        boosted = raw.replace(np.clip(raw.data.astype(np.float32) * gain, 0.0, 1.0))
        rgb = demosaic(boosted)
        rgb = apply_white_balance(rgb, raw.wb_gains)
        return upsample_to(rgb, meta.target_shape, order=self.params.get("resample_order", 1))


@register("classical_isp")
class ClassicalISP(_ClassicalBase):
    """The control condition: a plain, competent, non-learned ISP."""

    def restore(self, degraded_raw: RawImage, meta: RestoreContext) -> np.ndarray:
        rgb = self.linear_rgb(degraded_raw, meta)
        srgb = tone_curve(rgb, contrast=self.params.get("contrast", 0.0))

        strength = float(self.params.get("denoise_strength", 0.35))
        if strength > 0:
            # Mild bilateral smoothing - edge-preserving, and deliberately weak.
            # The control is meant to be a fair floor, not a straw man.
            smoothed = cv2.bilateralFilter(srgb, d=5, sigmaColor=0.08, sigmaSpace=3)
            srgb = np.clip(srgb * (1 - strength) + smoothed * strength, 0.0, 1.0)
        return to_uint8(srgb)


@register("nlm_denoise")
class NonLocalMeansDenoise(_ClassicalBase):
    """ISP plus non-local means. Denoising without a learned prior."""

    def restore(self, degraded_raw: RawImage, meta: RestoreContext) -> np.ndarray:
        rgb = self.linear_rgb(degraded_raw, meta)
        srgb = to_uint8(tone_curve(rgb))

        # Scale the filter strength with the noise the condition implies. A fixed
        # h over-smooths bright frames and under-cleans dark ones.
        h = float(self.params.get("h", 0.0)) or self._auto_strength(meta)
        return cv2.fastNlMeansDenoisingColored(
            srgb,
            None,
            h=h,
            hColor=h,
            templateWindowSize=int(self.params.get("template_window", 7)),
            searchWindowSize=int(self.params.get("search_window", 21)),
        )

    @staticmethod
    def _auto_strength(meta: RestoreContext) -> float:
        """Filter strength implied by the condition's noise, not by its severity.

        Shot noise after digital gain grows like ``1/sqrt(alpha)``, and read
        noise scales with it too once the gain is applied. Binning then works
        the other way: ``b x b`` pixels are averaged before readout, so SNR
        improves by ``b`` and the frame arrives *already* smoothed. Ignoring
        that term over-filters every binned frame - it strips detail to chase
        noise that the sensor already removed, which shows up as a denoising
        arm that loses to the control on a frame it should have matched.
        """
        alpha = max(meta.alpha, 1e-3)
        binning = max(1, meta.binning)
        sigma_r = float(meta.degradation.get("sigma_r_dn", 0.0))
        implied = (3.0 + 0.6 * sigma_r) / np.sqrt(alpha) / binning
        return float(np.clip(implied, 1.0, 30.0))


@register("curve_tone")
class CurveTone(_ClassicalBase):
    """Iterative light-enhancement curves with analytic coefficients.

    Applies ``LE(x) = x + A(x) * x * (1 - x)`` a few times. The curve shape is
    the one Zero-DCE learns; here ``A`` is computed from a local luminance
    estimate rather than predicted by a network. It brightens shadows without
    clipping highlights, which is what makes it a fair low-light arm - but it is
    a hand-written estimator and must not be reported as Zero-DCE.
    """

    def restore(self, degraded_raw: RawImage, meta: RestoreContext) -> np.ndarray:
        rgb = self.linear_rgb(degraded_raw, meta)
        srgb = tone_curve(rgb)

        iterations = int(self.params.get("iterations", 6))
        strength = float(self.params.get("strength", 0.5))

        luma = cv2.cvtColor(srgb, cv2.COLOR_RGB2GRAY)
        local = cv2.GaussianBlur(luma, (0, 0), max(1.0, min(srgb.shape[:2]) / 64.0))
        # Dark neighbourhoods get a larger coefficient; bright ones are left alone.
        amount = strength * (1.0 - np.clip(local, 0.0, 1.0)) ** 2
        amount = amount[..., None]

        out = srgb.copy()
        for _ in range(iterations):
            out = np.clip(out + amount * out * (1.0 - out), 0.0, 1.0)

        if self.params.get("denoise_after", True):
            out = cv2.bilateralFilter(out.astype(np.float32), d=7, sigmaColor=0.10, sigmaSpace=5)
        return to_uint8(out)


@register("rl_deconv")
class RichardsonLucyDeconv(_ClassicalBase):
    """Denoise, then Richardson-Lucy deconvolution against the recorded PSF.

    Deconvolution before denoising amplifies noise into ringing, so the order
    matters. Iterations are capped low for the same reason: RL converges toward
    the noisy measurement, so more iterations eventually means more noise, not
    more detail.

    The PSF is taken from the degradation record. That is oracle knowledge - a
    real pipeline would estimate it - and it is deliberate: this arm answers
    "how much does inverting the *known* optics buy?", which is the ceiling for
    non-generative deblurring and therefore the right thing to compare
    generative arms against.
    """

    def restore(self, degraded_raw: RawImage, meta: RestoreContext) -> np.ndarray:
        rgb = self.linear_rgb(degraded_raw, meta)

        denoised = cv2.bilateralFilter(rgb.astype(np.float32), d=7, sigmaColor=0.06, sigmaSpace=5)

        radius = float(meta.degradation.get("r_psf", 0.0))
        binning = max(1, meta.binning)
        # The frame was upsampled back to reference resolution, so the PSF the
        # image now carries is the capture PSF scaled by the binning factor.
        effective_radius = radius * binning
        if effective_radius <= 0.05:
            return to_uint8(tone_curve(denoised))

        psf = make_psf(meta.degradation.get("psf_kind", "gaussian"), effective_radius).astype(np.float32)
        iterations = int(self.params.get("iterations", 12))
        out = np.stack(
            [_richardson_lucy(denoised[..., c], psf, iterations) for c in range(3)], axis=-1
        )
        return to_uint8(tone_curve(np.clip(out, 0.0, 1.0)))


def _richardson_lucy(image: np.ndarray, psf: np.ndarray, iterations: int) -> np.ndarray:
    """Richardson-Lucy deconvolution of a single channel."""
    image = np.clip(image.astype(np.float32), 1e-6, None)
    psf = psf.astype(np.float32)
    psf_mirror = psf[::-1, ::-1].copy()
    estimate = image.copy()
    for _ in range(max(0, iterations)):
        blurred = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)
        ratio = image / np.clip(blurred, 1e-6, None)
        correction = cv2.filter2D(ratio, -1, psf_mirror, borderType=cv2.BORDER_REFLECT)
        estimate = np.clip(estimate * correction, 0.0, 4.0)
    return estimate
