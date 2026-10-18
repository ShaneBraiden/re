"""Train NC-DCE: zero-reference, noise-aware curve estimation.

    sensorsim train nc-dce [--config configs/train/nc_dce.yaml] [--variant NAME]

Each sample is a procedural scene (:mod:`sensorsim.train.scenes`) degraded by
the project's own forward model at a random severity, then passed through the
same front end the restoration arms use. That front end also yields the exact
noise map. Clean scenes are never used as targets, so training stays
zero-reference like the base method.

Loss = the four Zero-DCE terms (with a one-sided exposure term; see
:mod:`sensorsim.enhance.ncdce`) + the noise-amplification term::

    s        = prod_n (1 + r_n (2 LE_{n-1} - 1))          curve slope, closed form
    sigma'   = |s| * sigma                                 noise std after the curve
    w_flat   = 1 - gamma                                   gamma: Wiener gate (guard.py)
    L_noise  = mean( w_flat * relu(sigma' - max(sigma, tau))^2 ) / tau^2

Brightening multiplies signal and noise alike, so in textured regions it costs
no SNR and the term stays quiet. In flat regions the noise is what a viewer
sees, so it may not grow beyond its input level or ``tau``, whichever is
larger.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sensorsim.degrader.model import DegradationParams, degrade
from sensorsim.data.synthetic import Scene, scene_to_raw
from sensorsim.enhance.noise import luma_sigma
from sensorsim.restore.base import RestoreContext
from sensorsim.train.scenes import training_scene


@dataclass
class TrainConfig:
    name: str = "nc_dce"
    out: str = "models/nc_dce_v1.pth"
    init: str = "models/zero_dce_epoch99.pth"
    in_ch: int = 4
    noise_loss: bool = True
    samples: int = 512
    size: int = 256
    seed: int = 1000
    epochs: int = 60
    batch: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    exposure_target: float = 0.6
    exposure_patch: int = 16
    #: "one_sided" (NC-DCE: lift only) or "two_sided" (the published Zero-DCE term)
    exposure: str = "one_sided"
    tau: float = 2.0 / 255.0
    w_spa: float = 1.0
    w_exp: float = 10.0
    w_col: float = 5.0
    w_tv: float = 200.0
    w_noise: float = 1.0
    device: str = "auto"
    log_dir: str = "experiments/enhance_study/training"
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def make_sample(seed: int, size: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """(srgb, sigma_srgb, condition) for one degraded training scene."""
    from sensorsim.restore.learned.nag import front_end

    rng = np.random.default_rng(seed + 7_000_003)
    params = DegradationParams(
        alpha=float(np.exp(rng.uniform(np.log(0.03), 0.0))),
        sigma_r_dn=float(rng.uniform(0.0, 10.0)),
        r_psf=float(rng.uniform(0.0, 2.4)),
        binning=1,
        bit_depth=int(rng.choice([10, 12, 14])),
    )
    scene = Scene(f"train_{seed}", "natural", training_scene(seed, size), {})
    clean = scene_to_raw(scene)
    degraded, record = degrade(clean, params, seed=seed)
    ctx = RestoreContext(scene.image_id, clean.shape, record.params, record.sensor, "natural")
    srgb, sigma = front_end(degraded, ctx)
    return srgb, sigma, {"alpha": params.alpha, "sigma_r_dn": params.sigma_r_dn}


def build_dataset(cfg: TrainConfig, say=print) -> tuple[np.ndarray, np.ndarray]:
    xs, ss = [], []
    for i in range(cfg.samples):
        srgb, sigma, _ = make_sample(cfg.seed + i, cfg.size)
        xs.append(srgb.transpose(2, 0, 1))
        ss.append(sigma.transpose(2, 0, 1))
        if (i + 1) % 128 == 0:
            say(f"  generated {i + 1}/{cfg.samples} training frames")
    return np.stack(xs).astype(np.float32), np.stack(ss).astype(np.float32)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def loss_spatial(enhanced, original):
    """Zero-DCE spatial consistency: keep neighbour contrast of 4x4 region means."""
    import torch.nn.functional as F

    e = F.avg_pool2d(enhanced.mean(1, keepdim=True), 4)
    o = F.avg_pool2d(original.mean(1, keepdim=True), 4)
    total = 0.0
    for dy, dx in ((0, 1), (1, 0)):
        de = e - e.roll(shifts=(dy, dx), dims=(2, 3))
        do = o - o.roll(shifts=(dy, dx), dims=(2, 3))
        total = total + ((de - do) ** 2)[..., 1:-1, 1:-1].mean()
    return 2.0 * total


def loss_exposure(enhanced, target: float, patch: int, two_sided: bool = False):
    """Exposure control. One-sided (NC-DCE) lifts regions under ``target`` and
    never darkens; two-sided is the published Zero-DCE term."""
    import torch.nn.functional as F

    level = F.avg_pool2d(enhanced.mean(1, keepdim=True), patch)
    if two_sided:
        return ((level - target) ** 2).mean()
    return (F.relu(target - level) ** 2).mean()


def loss_colour(enhanced):
    """Zero-DCE colour constancy (grey-world on the channel means)."""
    m = enhanced.mean(dim=(2, 3))
    d_rg, d_rb, d_gb = (m[:, 0] - m[:, 1]) ** 2, (m[:, 0] - m[:, 2]) ** 2, (m[:, 1] - m[:, 2]) ** 2
    return (d_rg**2 + d_rb**2 + d_gb**2).sqrt().mean()


def loss_tv(r):
    """Illumination smoothness of the curve maps."""
    dh = (r[..., 1:, :] - r[..., :-1, :]) ** 2
    dw = (r[..., :, 1:] - r[..., :, :-1]) ** 2
    return dh.sum(1).mean() + dw.sum(1).mean()


def flat_weight(x, sigma, window: int = 7):
    """``1 - gamma``: the Wiener gate's complement, so flat = 1, structure = 0."""
    import torch
    import torch.nn.functional as F

    w709 = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device).view(1, 3, 1, 1)
    luma = (x * w709).sum(1, keepdim=True)
    s2 = ((sigma * w709) ** 2).sum(1, keepdim=True)
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    resp = F.conv2d(F.pad(luma, (1, 1, 1, 1), mode="reflect"), lap)
    energy = F.avg_pool2d(resp * resp, window, stride=1, padding=window // 2, count_include_pad=False)
    noise = 20.0 * F.avg_pool2d(s2, window, stride=1, padding=window // 2, count_include_pad=False)
    gamma = ((energy - noise) / (energy + 1e-6)).clamp(0.0, 1.0)
    return 1.0 - gamma


def loss_noise(x, sigma, r, tau: float, w_flat=None):
    """Noise-amplification penalty (this work). See module docstring."""
    import torch.nn.functional as F

    from sensorsim.enhance.noise import curve_slope_torch

    slope, _ = curve_slope_torch(x, r)
    sigma_out = slope.abs() * sigma
    budget = sigma.clamp(min=tau)
    if w_flat is None:
        w_flat = flat_weight(x, sigma)
    return (w_flat * F.relu(sigma_out - budget) ** 2).mean() / (tau * tau)


def total_loss(model, x, sigma, cfg: TrainConfig) -> tuple[Any, dict[str, float]]:
    import torch

    from sensorsim.restore.learned.arch import DCENet

    inp = x if cfg.in_ch == 3 else torch.cat([x, luma_sigma_t(sigma)], 1)
    r = model.curves(inp)
    y = DCENet.apply_curves(x, r)
    parts = {
        "spa": cfg.w_spa * loss_spatial(y, x),
        "exp": cfg.w_exp * loss_exposure(y, cfg.exposure_target, cfg.exposure_patch,
                                         two_sided=cfg.exposure == "two_sided"),
        "col": cfg.w_col * loss_colour(y),
        "tv": cfg.w_tv * loss_tv(r),
    }
    # The noise term is measured for every variant so the ablation can report
    # it, but it only enters the gradient when the variant switches it on.
    noise = loss_noise(x, sigma, r, cfg.tau)
    if cfg.noise_loss:
        parts["noise"] = cfg.w_noise * noise
    total = sum(parts.values())
    log = {k: float(v.detach()) for k, v in parts.items()}
    log["noise_metric"] = float(noise.detach())
    log["total"] = float(total.detach())
    return total, log


def luma_sigma_t(sigma):
    import torch

    w = torch.tensor([0.2126, 0.7152, 0.0722], device=sigma.device).view(1, 3, 1, 1)
    return ((sigma * w) ** 2).sum(1, keepdim=True).sqrt()


# --------------------------------------------------------------------------
# Loop
# --------------------------------------------------------------------------

def train(cfg: TrainConfig, say=print, dataset: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    import torch

    from sensorsim.enhance.weights import sha256
    from sensorsim.restore.learned.arch import build_dce, unwrap_state, widen_dce_state

    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = build_dce(cfg.in_ch)
    if cfg.init and Path(cfg.init).exists():
        state = unwrap_state(torch.load(cfg.init, map_location="cpu", weights_only=True))
        model.load_state_dict(widen_dce_state(state, cfg.in_ch))
        say(f"[{cfg.name}] initialised from {cfg.init}")
    model.to(device).train()

    started = time.perf_counter()
    if dataset is None:
        say(f"[{cfg.name}] generating {cfg.samples} degraded frames at {cfg.size}px")
        dataset = build_dataset(cfg, say)
    xs = torch.from_numpy(dataset[0]).to(device)
    ss = torch.from_numpy(dataset[1]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict[str, float]] = []
    diverged: dict[str, Any] | None = None
    n = xs.shape[0]
    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        sums: dict[str, float] = {}
        steps = 0
        for start in range(0, n - cfg.batch + 1, cfg.batch):
            idx = torch.from_numpy(order[start:start + cfg.batch]).to(device)
            x, s = xs[idx], ss[idx]
            if rng.random() < 0.5:
                x, s = x.flip(-1), s.flip(-1)
            k = int(rng.integers(0, 4))
            if k:
                x, s = torch.rot90(x, k, (2, 3)), torch.rot90(s, k, (2, 3))
            loss, log = total_loss(model, x, s, cfg)
            if not torch.isfinite(loss):
                # A non-finite loss poisons every weight on the next step. Stop,
                # and record it: a diverged variant is a result, not a model.
                diverged = {"epoch": epoch + 1, "step": steps + 1,
                            "last_noise_metric": sums.get("noise_metric", float("nan")) / max(steps, 1)}
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            for key, v in log.items():
                sums[key] = sums.get(key, 0.0) + v
            steps += 1
        if steps:
            row = {"epoch": epoch + 1, **{k: v / max(steps, 1) for k, v in sums.items()}}
            history.append(row)
        if diverged:
            say(f"[{cfg.name}] DIVERGED at epoch {diverged['epoch']} step {diverged['step']} - no checkpoint saved")
            break
        if (epoch + 1) % 10 == 0 or epoch == 0:
            say(f"[{cfg.name}] epoch {epoch + 1:3d}  total {row['total']:.4f}  noise {row['noise_metric']:.4f}  exp {row['exp']:.4f}")

    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if diverged:
        out.unlink(missing_ok=True)
        digest = None
    else:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out)
        digest = sha256(out)
    meta = {
        "name": cfg.name,
        "sha256": digest,
        "diverged": diverged,
        "in_ch": cfg.in_ch,
        "noise_loss": cfg.noise_loss,
        "config": {k: v for k, v in cfg.__dict__.items() if k != "extra"},
        "final": history[-1] if history else {},
        "seconds": round(time.perf_counter() - started, 1),
        "device": device,
        "torch": torch.__version__,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{cfg.name}_loss.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]) if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    if digest:
        say(f"[{cfg.name}] saved {out} sha256 {digest[:16]} in {meta['seconds']}s")
    return meta


def load_configs(path: str | Path) -> list[TrainConfig]:
    import yaml

    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    common = spec.get("common", {})
    return [TrainConfig(**{**common, **variant}) for variant in spec.get("variants", [])]


def train_all(path: str | Path, only: list[str] | None = None, epochs: int | None = None,
              device: str | None = None, say=print) -> list[dict[str, Any]]:
    """Train every variant in a config. They share one generated dataset."""
    configs = [c for c in load_configs(path) if not only or c.name in only]
    if not configs:
        raise ValueError(f"no variants selected from {path}")
    for c in configs:
        if epochs:
            c.epochs = epochs
        if device:
            c.device = device
    say(f"generating {configs[0].samples} degraded frames at {configs[0].size}px (shared by {len(configs)} variants)")
    dataset = build_dataset(configs[0], say)
    return [train(c, say, dataset=dataset) for c in configs]
