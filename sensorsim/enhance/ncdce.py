"""NC-DCE: Zero-DCE curves that know how noisy each pixel is.

Zero-DCE (Guo et al., CVPR 2020) brightens an image by applying eight learned
quadratic curves per pixel. It learns them without reference images, from
four losses on the output: spatial consistency, exposure, colour constancy,
and smoothness of the curve maps. None of those losses can see noise, so the
network lifts a noisy shadow exactly as hard as a clean one, and the noise is
lifted with it.

NC-DCE changes two things:

* **Input.** A fourth channel carries the luma noise std map from
  :mod:`sensorsim.enhance.noise`. The network can tell a dark, clean region
  from a dark, noisy one.
* **Loss.** The curve's slope is known in closed form
  (:func:`~sensorsim.enhance.noise.curve_slope_torch`), so the noise std
  after the curve is ``s * sigma``. A new term penalises lifting it above
  ``max(sigma, tau)`` in flat regions (see :mod:`sensorsim.train.nc_dce`).

Two further changes suit this testbench:

* **Exposure loss.** It is one-sided: every arm already receives the frame at
  matched brightness, so the curves should lift under-exposed regions but never
  pull a correctly exposed one down to the 0.6 target.
* **Initialisation.** Training starts from the published Zero-DCE weights,
  with the noise channel's input weights at zero.

This module is inference only. It loads a checkpoint and returns the enhanced
image together with its propagated noise map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sensorsim.enhance.noise import luma_sigma, propagate


class CurveEnhancer:
    """Load a DCE-Net checkpoint (3- or 4-channel) and run it on one image."""

    def __init__(self, weights: str | Path, in_ch: int | None = None, device: str = "auto"):
        import torch

        from sensorsim.restore.learned.arch import build_dce, unwrap_state, widen_dce_state

        state = unwrap_state(torch.load(str(weights), map_location="cpu", weights_only=True))
        bad = [k for k, v in state.items() if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
        if bad:
            raise ValueError(f"{weights} holds non-finite weights ({bad[0]}, ...); the training run diverged")
        meta_in = int(state.pop("_in_ch").item()) if "_in_ch" in state else state["e_conv1.weight"].shape[1]
        self.in_ch = int(in_ch or meta_in)
        model = build_dce(self.in_ch)
        model.load_state_dict(widen_dce_state(state, self.in_ch))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.eval().to(device)

    def __call__(self, srgb: np.ndarray, sigma: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Enhance ``srgb`` (H, W, 3) in [0, 1].

        Returns ``(out, sigma_out, r)``: the image, the noise std carried
        through the curves (``None`` if no ``sigma`` was given), and the
        (H, W, 24) curve maps.
        """
        import torch

        from sensorsim.restore.learned.arch import DCENet

        x = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
        planes = [x.transpose(2, 0, 1)]
        if self.in_ch == 4:
            if sigma is None:
                raise ValueError("this NC-DCE checkpoint needs a noise map")
            planes.append(luma_sigma(sigma)[None])
        stack = torch.from_numpy(np.ascontiguousarray(np.concatenate(planes, 0)))[None].to(self.device)
        with torch.no_grad():
            r = self.model.curves(stack)
            out = DCENet.apply_curves(stack[:, :3], r)
        out_np = out[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        r_np = r[0].cpu().numpy().transpose(1, 2, 0)

        sigma_out = None
        if sigma is not None:
            from sensorsim.enhance.noise import apply_curves_np

            sigma_out = propagate(lambda v: apply_curves_np(v, r_np), x, sigma.astype(np.float32))
        return out_np.astype(np.float32), sigma_out, r_np
