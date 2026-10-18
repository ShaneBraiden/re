"""Learned restoration arms.

Each arm here is a *specification*: the interface, the pre/post-processing, the
weights contract and the device handling are all implemented, and the only
missing piece is the checkpoint plus the architecture module it was trained
with. That split is deliberate. Model code and weights are large, licensed
separately, and pinned by checksum, so they are vendored per deployment rather
than committed here.

**These arms never fabricate a result.** With torch or the weights absent,
:meth:`availability` returns false and the runner records a gap in the metrics
table. It never quietly falls back to a classical pipeline, because a row
labelled ``nafnet`` that was actually produced by a bilateral filter is a wrong
number wearing the right name, and it would survive review.

To activate an arm:

1. ``pip install sensorsim[learned]``
2. Put the checkpoint somewhere and point the config's ``weights`` at it.
3. Provide the architecture: either set ``arch`` to an importable factory
   (``mypkg.models:build_nafnet``) or vendor the upstream model code and set
   ``arch`` to its constructor path.
4. Record the checksum - ``sensorsim weights pin`` writes it into the config.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sensorsim.degrader.sampling import upsample_to
from sensorsim.io.raw import RawImage
from sensorsim.restore.base import Availability, Pipeline, RestoreContext, register
from sensorsim.restore.classical import demosaic, apply_white_balance, estimate_gain, to_uint8, tone_curve


def torch_available() -> bool:
    try:
        importlib.import_module("torch")
        return True
    except ImportError:
        return False


class TorchPipeline(Pipeline):
    """Common machinery for a checkpoint-backed restoration arm.

    Parameters
    ----------
    weights:
        Path to the checkpoint. Required.
    arch:
        ``module:factory`` for a callable returning an ``nn.Module``.
    device:
        ``cuda``, ``cpu``, or ``auto``.
    tile:
        Tile size for large frames, 0 to run whole-frame.
    input_domain:
        ``rgb`` for arms that take a demosaiced image, ``raw`` for end-to-end
        RAW-to-RGB models that take the mosaic directly.
    """

    input_domain = "rgb"
    #: Human-readable name of the upstream method, for error messages.
    upstream = ""

    def availability(self) -> Availability:
        missing: list[str] = []
        if not torch_available():
            missing.append("torch")
        weights = self.params.get("weights")
        if not weights:
            missing.append("weights path (set params.weights in the pipeline config)")
        elif not Path(weights).exists():
            missing.append(f"weights file {weights}")
        if not self.params.get("arch"):
            missing.append("arch (module:factory returning the nn.Module)")
        if missing:
            return Availability(
                ok=False,
                reason=f"{self.name} needs: " + "; ".join(missing),
                missing=tuple(missing),
            )
        return Availability(True)

    # -- model plumbing ----------------------------------------------------

    def _load_model(self) -> Any:  # pragma: no cover - needs torch + a checkpoint
        import torch

        cached = getattr(self, "_model", None)
        if cached is not None:
            return cached

        module_path, _, factory_name = str(self.params["arch"]).partition(":")
        if not factory_name:
            raise ValueError(
                f"arch must be 'module:factory', got {self.params['arch']!r}. "
                "The factory takes no required arguments and returns an nn.Module."
            )
        factory: Callable[..., Any] = getattr(importlib.import_module(module_path), factory_name)
        model = factory(**self.params.get("arch_kwargs", {}))

        from sensorsim.restore.learned.arch import unwrap_state

        state = unwrap_state(torch.load(self.params["weights"], map_location="cpu", weights_only=True))
        model.load_state_dict(state, strict=self.params.get("strict_load", True))

        model.eval().to(self._device())
        self._model = model
        return model

    def _device(self) -> str:  # pragma: no cover - needs torch
        import torch

        want = self.params.get("device", "auto")
        if want == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return want

    def _infer(self, array: np.ndarray) -> np.ndarray:  # pragma: no cover - needs torch
        """Run the model on an (H, W, C) float array in [0, 1]."""
        import torch

        model = self._load_model()
        device = self._device()
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))[None].float().to(device)

        tile = int(self.params.get("tile", 0))
        with torch.no_grad():
            if tile <= 0:
                out = model(tensor)
            else:
                out = _tiled_forward(model, tensor, tile, int(self.params.get("tile_overlap", 32)))
        out = out[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        return out.astype(np.float32)

    # -- interface ---------------------------------------------------------

    def restore(self, degraded_raw: RawImage, meta: RestoreContext) -> np.ndarray:  # pragma: no cover
        availability = self.availability()
        if not availability:
            raise RuntimeError(availability.reason)

        if self.input_domain == "raw":
            gain = estimate_gain(degraded_raw, meta, self.params.get("gain_mode", "recorded"))
            packed = _pack_bayer(np.clip(degraded_raw.data * gain, 0.0, 1.0), degraded_raw.cfa_pattern)
            out = self._infer(packed)
        else:
            gain = estimate_gain(degraded_raw, meta, self.params.get("gain_mode", "recorded"))
            boosted = degraded_raw.replace(np.clip(degraded_raw.data * gain, 0.0, 1.0))
            rgb = apply_white_balance(demosaic(boosted), degraded_raw.wb_gains)
            out = self._infer(tone_curve(rgb))

        if out.shape[0] * out.shape[1] > meta.target_shape[0] * meta.target_shape[1]:
            # A super-resolution model returns more pixels than the reference
            # has. Bilinear decimation by 4 would alias, and the arm would be
            # charged for the resampler's error, so shrink with area averaging.
            import cv2

            out = cv2.resize(out, (meta.target_shape[1], meta.target_shape[0]), interpolation=cv2.INTER_AREA)
        out = upsample_to(out, meta.target_shape, order=1)
        return to_uint8(out)


def _pack_bayer(raw: np.ndarray, pattern: str) -> np.ndarray:
    """Bayer mosaic to a 4-channel half-resolution stack, as RAW models expect."""
    from sensorsim.io.raw import bayer_to_planes

    planes = bayer_to_planes(raw, pattern)
    return np.stack([planes["R"], planes["G1"], planes["G2"], planes["B"]], axis=-1).astype(np.float32)


def _tiled_forward(model: Any, tensor: Any, tile: int, overlap: int) -> Any:
    """Run a model tile by tile, for frames that will not fit in memory at once.

    Tiles overlap by ``overlap`` pixels, and each one contributes only its
    interior: half the overlap is trimmed from every side that borders another
    tile. Averaging the overlaps instead would mix in the pixels nearest each
    tile's edge, which the network computed from zero padding, and leave faint
    seams. A model whose receptive-field radius is within ``overlap / 2``
    therefore tiles exactly.

    The output scale is read off the first patch rather than assumed to be 1, so
    super-resolution models tile too.
    """
    import torch

    _, _, h, w = tensor.shape
    stride = max(1, tile - overlap)
    half = overlap // 2
    output = None
    scale = 1
    for top in range(0, max(h - overlap, 1), stride):
        bottom = min(top + tile, h)
        for left in range(0, max(w - overlap, 1), stride):
            right = min(left + tile, w)
            patch = model(tensor[..., top:bottom, left:right])
            if output is None:
                scale = max(1, round(patch.shape[-1] / (right - left)))
                output = torch.zeros((patch.shape[0], patch.shape[1], h * scale, w * scale),
                                     device=patch.device, dtype=patch.dtype)
            y0 = top + (half if top > 0 else 0)
            y1 = bottom - (half if bottom < h else 0)
            x0 = left + (half if left > 0 else 0)
            x1 = right - (half if right < w else 0)
            output[..., y0 * scale:y1 * scale, x0 * scale:x1 * scale] = patch[
                ..., (y0 - top) * scale:(y1 - top) * scale, (x0 - left) * scale:(x1 - left) * scale]
    return output


# --------------------------------------------------------------------------
# The bank
# --------------------------------------------------------------------------

@register("zero_dce")
class ZeroDCE(TorchPipeline):
    """Zero-DCE: low-light enhancement by learned curve estimation.

    Guo et al., *Zero-Reference Deep Curve Estimation for Low-Light Image
    Enhancement*, CVPR 2020.
    """

    upstream = "Zero-DCE (Guo et al., CVPR 2020)"


@register("nafnet")
class NAFNet(TorchPipeline):
    """NAFNet: simple baselines for image restoration.

    Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022.
    """

    upstream = "NAFNet (Chen et al., ECCV 2022)"


@register("restormer")
class Restormer(TorchPipeline):
    """Restormer: efficient transformer for high-resolution restoration.

    Zamir et al., CVPR 2022.
    """

    upstream = "Restormer (Zamir et al., CVPR 2022)"


@register("real_esrgan")
class RealESRGAN(TorchPipeline):
    """Real-ESRGAN: generative detail synthesis.

    The hallucination-prone arm, included on purpose. It is the arm expected to
    win on perceptual metrics while losing on fidelity and machine-vision
    metrics, which is the whole of H2 and H3. Read its machine-task rows before
    its LPIPS rows.

    Wang et al., *Real-ESRGAN: Training Real-World Blind Super-Resolution with
    Pure Synthetic Data*, ICCVW 2021.
    """

    upstream = "Real-ESRGAN (Wang et al., ICCVW 2021)"


@register("learned_isp")
class LearnedISP(TorchPipeline):
    """End-to-end RAW to RGB, trained on Zurich RAW-to-RGB.

    Ignatov et al., *Replacing Mobile Camera ISP with a Single Deep Learning
    Model*, CVPRW 2020.
    """

    upstream = "Learned ISP (Ignatov et al., CVPRW 2020)"
    input_domain = "raw"
