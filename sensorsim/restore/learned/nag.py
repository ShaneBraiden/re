"""The NAG arms: NC-DCE, G-ESRGAN, and the two chained.

They share one front end with every other arm: recorded gain, demosaic and
white balance. Each additionally computes the exact per-pixel noise map that
the frame's own capture record implies (``RestoreContext.degradation`` and
``.sensor``) and carries it through every stage. That map is metadata a real
camera has about its own capture - the same justification the classical arms
use for applying the recorded gain - and it never touches the clean reference.

``nc_dce``    noise-conditioned curves only
``g_esrgan``  Real-ESRGAN with the Wiener gate and back-projection
``nag``       NC-DCE -> VP-denoise -> G-ESRGAN

The ablations are the same kinds with a switch off (``gate: false``,
``bp_iterations: 0``, ``denoise: false``) or an ablated checkpoint, so they
share code and differ only in the config entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sensorsim.degrader.optics import make_psf
from sensorsim.degrader.sampling import upsample_to
from sensorsim.io.manifest import file_checksum
from sensorsim.io.raw import RawImage
from sensorsim.restore.base import Availability, RestoreContext, register
from sensorsim.restore.classical import apply_white_balance, demosaic, estimate_gain, to_uint8, tone_curve
from sensorsim.restore.learned.torch_pipelines import TorchPipeline, torch_available


def front_end(raw: RawImage, meta: RestoreContext) -> tuple[np.ndarray, np.ndarray]:
    """sRGB frame at capture resolution, plus its per-pixel noise std map."""
    from sensorsim.enhance.noise import propagate_tone, sigma_map_from_context

    gain = estimate_gain(raw, meta, "recorded")
    boosted = raw.replace(np.clip(raw.data.astype(np.float32) * gain, 0.0, 1.0))
    linear = apply_white_balance(demosaic(boosted), raw.wb_gains)
    sigma_linear = sigma_map_from_context(
        linear, meta.degradation, meta.sensor, raw.wb_gains, span_dn=raw.white_level - raw.black_level
    )
    return tone_curve(linear), propagate_tone(linear, sigma_linear)


def capture_psf(meta: RestoreContext) -> np.ndarray | None:
    radius = float(meta.degradation.get("r_psf", 0.0))
    if radius <= 0.05:
        return None
    return make_psf(meta.degradation.get("psf_kind", "gaussian"), radius).astype(np.float32)


class _NAGBase(TorchPipeline):
    """Checkpoint plumbing shared by the three arms."""

    #: params keys that name checkpoint files
    weight_keys: tuple[str, ...] = ("weights",)

    def availability(self) -> Availability:
        missing: list[str] = []
        if not torch_available():
            missing.append("torch")
        for key in self.weight_keys:
            value = self.params.get(key)
            if not value:
                missing.append(f"{key} (path to a checkpoint)")
            elif not Path(value).exists():
                from sensorsim.enhance.weights import diverged_reason

                missing.append(diverged_reason(value) or f"{key} file {value}")
        if missing:
            return Availability(False, f"{self.name} needs: " + "; ".join(missing), tuple(missing))
        return Availability(True)

    def weights_version(self) -> str:
        parts = []
        for key in self.weight_keys:
            value = self.params.get(key)
            parts.append(file_checksum(value) if value and Path(value).exists() else "no-weights")
        return "+".join(parts)

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "params": {k: v for k, v in self.params.items() if k not in self.weight_keys},
            "weights_version": self.weights_version(),
        }

    def _enhancer(self, key: str):
        from sensorsim.enhance.ncdce import CurveEnhancer

        cached = getattr(self, "_curve", None)
        if cached is None:
            cached = CurveEnhancer(self.params[key], device=self.params.get("device", "auto"))
            self._curve = cached
        return cached

    def _sr(self, key: str):
        from sensorsim.enhance.sr import SuperResolver

        cached = getattr(self, "_sr_model", None)
        if cached is None:
            cached = SuperResolver(
                self.params[key],
                kind=self.params.get("sr_kind", "x4plus"),
                device=self.params.get("device", "auto"),
                tile=int(self.params.get("tile", 256)),
            )
            self._sr_model = cached
        return cached


@register("nc_dce")
class NCDCE(_NAGBase):
    """Noise-conditioned deep curve estimation (this work)."""

    upstream = "NC-DCE (this work; extends Zero-DCE, Guo et al., CVPR 2020)"

    def restore(self, degraded_raw: RawImage, meta: RestoreContext) -> np.ndarray:
        srgb, sigma = front_end(degraded_raw, meta)
        out, _, _ = self._enhancer("weights")(srgb, sigma)
        return to_uint8(upsample_to(out, meta.target_shape, order=1))


@register("g_esrgan")
class GuardedESRGAN(_NAGBase):
    """Real-ESRGAN with a noise-driven Wiener gate and back-projection (this work)."""

    upstream = "G-ESRGAN (this work; extends Real-ESRGAN, Wang et al., ICCVW 2021)"

    def restore(self, degraded_raw: RawImage, meta: RestoreContext) -> np.ndarray:
        from sensorsim.enhance.guard import guarded_sr

        srgb, sigma = front_end(degraded_raw, meta)
        result = guarded_sr(
            srgb, sigma, self._sr("weights"), tuple(meta.target_shape), psf=capture_psf(meta),
            gate=bool(self.params.get("gate", True)),
            bp_iterations=int(self.params.get("bp_iterations", 3)),
        )
        return to_uint8(result.image)


@register("nag")
class NoiseAwareGuarded(_NAGBase):
    """NC-DCE, then VP-denoise, then G-ESRGAN, sharing one noise map (this work)."""

    upstream = "NAG (this work)"
    weight_keys = ("dce_weights", "sr_weights")

    def restore(self, degraded_raw: RawImage, meta: RestoreContext) -> np.ndarray:
        from sensorsim.enhance.guard import guarded_sr
        from sensorsim.enhance.vpdenoise import guided_filter_varying

        srgb, sigma = front_end(degraded_raw, meta)
        curved, sigma_curved, _ = self._enhancer("dce_weights")(srgb, sigma)
        if self.params.get("denoise", True):
            curved, sigma_curved = guided_filter_varying(
                curved, sigma_curved, radius=int(self.params.get("radius", 3)), k=float(self.params.get("k", 1.5))
            )
        result = guarded_sr(
            curved, sigma_curved, self._sr("sr_weights"), tuple(meta.target_shape), psf=capture_psf(meta),
            gate=bool(self.params.get("gate", True)),
            bp_iterations=int(self.params.get("bp_iterations", 3)),
        )
        return to_uint8(result.image)
