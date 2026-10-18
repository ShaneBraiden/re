"""Network architectures for the checkpoint-backed arms.

The published checkpoints are plain ``state_dict``s, so loading one needs the
module it was trained with, layer names included. Each class below is written
to the published description and keeps the upstream parameter names, so the
released weights load with ``strict=True`` and nothing is silently left at its
random initialisation.

``RRDBNet``
    The Real-ESRGAN generator (``RealESRGAN_x4plus``). Wang et al., ICCVW 2021,
    on the ESRGAN residual-in-residual dense block of Wang et al., ECCVW 2018.
    Layer naming follows BasicSR (Apache-2.0) so its checkpoints load.
``SRVGGNetCompact``
    The compact Real-ESRGAN generator (``realesr-general-x4v3``): a plain conv
    stack with a pixel-shuffle tail. Kinder to line art and text than RRDBNet.
``DCENet``
    The Zero-DCE curve estimator. Guo et al., CVPR 2020. Seven 3x3 convolutions
    with symmetric skip concatenations, predicting eight per-pixel curve maps
    per colour channel. ``in_ch=4`` adds a noise-level channel for NC-DCE (see
    :mod:`sensorsim.enhance.ncdce`); the first three input channels keep the
    published weights, the fourth starts at zero.

Each factory takes no required arguments, which is the ``arch`` contract
:class:`~sensorsim.restore.learned.torch_pipelines.TorchPipeline` expects.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------
# Real-ESRGAN x4plus: residual-in-residual dense blocks
# --------------------------------------------------------------------------

class ResidualDenseBlock(nn.Module):
    """Five densely connected convolutions with a scaled residual."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Three dense blocks inside one more scaled residual."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN generator, x4, nearest-neighbour upsampling between convolutions."""

    scale = 4

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64,
                 num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# --------------------------------------------------------------------------
# Real-ESRGAN general-x4v3: compact VGG-style generator
# --------------------------------------------------------------------------

class SRVGGNetCompact(nn.Module):
    """Conv + PReLU stack, pixel shuffle, plus a nearest-upsampled skip."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64,
                 num_conv: int = 32, upscale: int = 4):
        super().__init__()
        self.scale = upscale
        layers: list[nn.Module] = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        layers.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(layers)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")


# --------------------------------------------------------------------------
# Zero-DCE curve estimator
# --------------------------------------------------------------------------

class DCENet(nn.Module):
    """Predicts eight light-enhancement curves per pixel and applies them.

    Each iteration applies ``LE(x) = x + r * (x^2 - x)`` with a per-pixel,
    per-channel ``r`` in [-1, 1]. That is the published curve with the sign
    folded into ``r``, kept that way so the released weights mean what they
    were trained to mean (negative ``r`` brightens).

    Only the image goes through ``forward``, which is what the restoration
    interface needs. :meth:`curves` also returns the curve maps, which the
    NC-DCE training loss and the noise propagation both need.
    """

    iterations = 8

    def __init__(self, in_ch: int = 3, num_feat: int = 32):
        super().__init__()
        self.in_ch = in_ch
        self.e_conv1 = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(num_feat * 2, 3 * self.iterations, 3, 1, 1)

    def curves(self, x: torch.Tensor) -> torch.Tensor:
        """The (N, 24, H, W) curve parameter maps for an input stack."""
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        return torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

    @staticmethod
    def apply_curves(rgb: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        out = rgb
        for r_n in torch.split(r, 3, dim=1):
            out = out + r_n * (out * out - out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_curves(x[:, :3], self.curves(x))


# --------------------------------------------------------------------------
# Factories - the ``arch`` contract: no required arguments
# --------------------------------------------------------------------------

def build_rrdbnet(num_block: int = 23, num_feat: int = 64, num_grow_ch: int = 32) -> RRDBNet:
    return RRDBNet(num_block=num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)


def build_srvgg(num_conv: int = 32, num_feat: int = 64, upscale: int = 4) -> SRVGGNetCompact:
    return SRVGGNetCompact(num_conv=num_conv, num_feat=num_feat, upscale=upscale)


def build_dce(in_ch: int = 3) -> DCENet:
    return DCENet(in_ch=in_ch)


def unwrap_state(state: object) -> dict:
    """The tensor dict inside a checkpoint, whatever wrapper it was saved in.

    BasicSR saves ``{"params_ema": ...}`` or ``{"params": ...}``; plain
    training loops save the dict itself, sometimes with a DataParallel
    ``module.`` prefix on every key.
    """
    if isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint holds {type(state).__name__}, not a state dict")
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def widen_dce_state(state: dict, in_ch: int) -> dict:
    """Published 3-channel Zero-DCE weights, padded to ``in_ch`` inputs.

    The extra input channels get zero weights, so at initialisation the widened
    network computes exactly what the published one does and training starts
    from the base method rather than from noise.
    """
    weight = state["e_conv1.weight"]
    if weight.shape[1] == in_ch:
        return state
    pad = torch.zeros(weight.shape[0], in_ch - weight.shape[1], *weight.shape[2:],
                      dtype=weight.dtype, device=weight.device)
    return {**state, "e_conv1.weight": torch.cat([weight, pad], dim=1)}
