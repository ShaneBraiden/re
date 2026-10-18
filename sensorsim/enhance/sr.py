"""Real-ESRGAN inference on one float image, tiled so large frames fit on a laptop GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR_ARCH = {
    "x4plus": "build_rrdbnet",
    "general": "build_srvgg",
}


class SuperResolver:
    """``RealESRGAN_x4plus`` (photos) or ``realesr-general-x4v3`` (line art, text)."""

    scale = 4

    def __init__(self, weights: str | Path, kind: str = "x4plus", device: str = "auto",
                 tile: int = 256, overlap: int = 16, half: bool = True):
        import torch

        from sensorsim.restore.learned import arch

        if kind not in SR_ARCH:
            raise ValueError(f"unknown SR model kind {kind!r}; expected one of {sorted(SR_ARCH)}")
        model = getattr(arch, SR_ARCH[kind])()
        model.load_state_dict(arch.unwrap_state(torch.load(str(weights), map_location="cpu", weights_only=True)))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.half = half and device.startswith("cuda")
        self.model = model.eval().to(device)
        if self.half:
            self.model = self.model.half()
        self.kind, self.tile, self.overlap = kind, tile, overlap

    def __call__(self, srgb: np.ndarray) -> np.ndarray:
        import torch

        from sensorsim.restore.learned.torch_pipelines import _tiled_forward

        x = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.device)
        if self.half:
            t = t.half()
        with torch.no_grad():
            if self.tile and max(x.shape[:2]) > self.tile:
                out = _tiled_forward(self.model, t, self.tile, self.overlap)
            else:
                out = self.model(t)
        return out[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
