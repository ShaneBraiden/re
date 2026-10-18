"""NAG's method core: the noise map, the curve slope, the gate, back-projection,
the architectures and the training loss.

Each test checks a claim the paper makes. The noise map has to match measured
noise. The closed-form slope has to match a finite difference. The gate has to
close on pure noise and open on an edge. Back-projection has to reduce the
measurement residual at every step.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensorsim.data.synthetic import Scene, scene_to_raw
from sensorsim.degrader.model import DegradationParams, degrade
from sensorsim.enhance.guard import back_project, guarded_sr, make_capture_operator, wiener_gate
from sensorsim.enhance.noise import (
    apply_curves_np,
    blind_sigma_map,
    curve_slope_np,
    propagate,
)
from sensorsim.enhance.vpdenoise import guided_filter_varying
from sensorsim.restore.base import RestoreContext
from sensorsim.restore.learned.nag import front_end


def _flat_frame(level: float, params: DegradationParams, size: int = 192):
    raw = scene_to_raw(Scene("flat", "natural", np.full((size, size, 3), level, np.float32), {}))
    degraded, rec = degrade(raw, params, seed=5)
    return front_end(degraded, RestoreContext("flat", raw.shape, rec.params, rec.sensor))


@pytest.mark.parametrize("level", [0.05, 0.3])
@pytest.mark.parametrize("alpha,sigma_r,binning", [(1.0, 0.0, 1), (0.1, 4.0, 1), (0.03, 10.0, 1), (0.1, 4.0, 2)])
def test_noise_map_matches_measured_noise(level, alpha, sigma_r, binning):
    srgb, sigma = _flat_frame(level, DegradationParams(alpha=alpha, sigma_r_dn=sigma_r, r_psf=0.0,
                                                       binning=binning, bit_depth=12))
    c = (slice(16, -16), slice(16, -16))
    ratio = srgb[c].std(axis=(0, 1)) / sigma[c].mean(axis=(0, 1))
    assert np.all(np.abs(ratio - 1.0) < 0.08), ratio


def test_curve_slope_matches_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.02, 0.98, size=(16, 16, 3)).astype(np.float64)
    r = rng.uniform(-1, 1, size=(16, 16, 24)).astype(np.float64)
    h = 1e-5
    numeric = (apply_curves_np(x + h, r).astype(np.float64) - apply_curves_np(x - h, r)) / (2 * h)
    analytic = curve_slope_np(x, r)
    assert np.allclose(analytic, numeric, rtol=2e-3, atol=2e-3)


def test_secant_propagation_predicts_noise_after_a_curve():
    rng = np.random.default_rng(1)
    x0, s0 = 0.15, 0.02
    noisy = np.clip(x0 + rng.normal(0, s0, size=(256, 256, 3)), 0, 1).astype(np.float32)
    r = np.full((256, 256, 24), -0.3, np.float32)  # moderate brightening, slope > 1
    measured = apply_curves_np(noisy, r).std()
    predicted = propagate(lambda v: apply_curves_np(v, r), np.full_like(noisy, x0), np.full_like(noisy, s0)).mean()
    assert abs(measured / predicted - 1) < 0.10


def test_blind_noise_estimate_recovers_known_noise():
    rng = np.random.default_rng(2)
    ramp = np.tile(np.linspace(0.1, 0.9, 256, dtype=np.float32), (256, 1))
    img = np.repeat(ramp[..., None], 3, axis=-1)
    noisy = np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1).astype(np.float32)
    sigma, nlf = blind_sigma_map(noisy)
    assert nlf["estimated"]
    assert 0.75 < float(np.median(sigma)) / 0.03 < 1.25


def test_gate_closes_on_noise_and_opens_on_edges():
    rng = np.random.default_rng(3)
    s = 0.04
    noise = np.clip(0.5 + rng.normal(0, s, (128, 128, 3)), 0, 1).astype(np.float32)
    gate_noise = wiener_gate(noise, np.full_like(noise, s))
    edge = np.full((128, 128, 3), 0.1, np.float32)
    edge[:, 64:] = 0.9
    gate_edge = wiener_gate(edge, np.full_like(edge, 0.01))
    assert gate_noise[8:-8, 8:-8].mean() < 0.15
    assert gate_edge[:, 60:68].mean() > 0.8


def test_back_projection_reduces_the_residual_every_step():
    rng = np.random.default_rng(4)
    truth = np.clip(rng.random((64, 64, 3)), 0, 1).astype(np.float32)
    import cv2

    truth = cv2.GaussianBlur(truth, (0, 0), 2.0)
    down = make_capture_operator((32, 32))
    y = down(truth)
    x = np.full((64, 64, 3), 0.5, np.float32)
    _, history = back_project(x, y, down, iterations=4, residual_smooth=0.0)
    assert all(b < a for a, b in zip(history, history[1:])), history


def test_guarded_sr_keeps_shape_and_suppresses_invented_detail():
    rng = np.random.default_rng(5)
    flat = np.full((32, 32, 3), 0.5, np.float32)
    s = np.full_like(flat, 0.02)
    # A "generator" that invents strong texture everywhere.
    inventive = lambda img: np.clip(  # noqa: E731
        np.kron(img, np.ones((4, 4, 1), np.float32)) + rng.normal(0, 0.2, (128, 128, 3)), 0, 1)
    out = guarded_sr(flat, s, inventive, (128, 128), bp_iterations=2)
    assert out.image.shape == (128, 128, 3)
    assert out.image.std() < 0.25 * out.unguarded.std()


def test_vp_denoise_is_strong_only_where_noise_is_high():
    rng = np.random.default_rng(6)
    img = np.clip(0.5 + rng.normal(0, 0.05, (96, 96, 3)), 0, 1).astype(np.float32)
    sigma = np.full_like(img, 0.002)
    sigma[:, 48:] = 0.05
    out, residual = guided_filter_varying(img, sigma)
    assert out[:, :40].std() > 0.8 * img[:, :40].std()
    assert out[:, 56:].std() < 0.5 * img[:, 56:].std()
    assert residual[:, 56:].mean() < sigma[:, 56:].mean()


# --------------------------------------------------------------------------
# Networks (need torch)
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_dce_widening_reproduces_the_base_network():
    from sensorsim.restore.learned.arch import build_dce, widen_dce_state

    base = build_dce(3).eval()
    wide = build_dce(4).eval()
    wide.load_state_dict(widen_dce_state(base.state_dict(), 4))
    x = torch.rand(1, 3, 32, 32)
    sigma = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(x), wide(torch.cat([x, sigma], 1)), atol=1e-6)


def test_sr_architectures_upscale_by_four():
    from sensorsim.restore.learned.arch import build_rrdbnet, build_srvgg

    x = torch.rand(1, 3, 12, 16)
    with torch.no_grad():
        assert build_rrdbnet(num_block=1).eval()(x).shape == (1, 3, 48, 64)
        assert build_srvgg(num_conv=2).eval()(x).shape == (1, 3, 48, 64)


def test_tiled_inference_matches_whole_frame_for_sr():
    from sensorsim.restore.learned.arch import build_srvgg
    from sensorsim.restore.learned.torch_pipelines import _tiled_forward

    torch.manual_seed(0)
    model = build_srvgg(num_conv=0).eval()  # receptive-field radius 2 < overlap / 2: tiling is exact
    x = torch.rand(1, 3, 40, 52)
    with torch.no_grad():
        whole = model(x)
        tiled = _tiled_forward(model, x, tile=24, overlap=8)
    assert tiled.shape == whole.shape
    assert torch.allclose(tiled, whole, atol=1e-5)


def test_noise_loss_penalises_amplification_only():
    from sensorsim.train.nc_dce import loss_noise

    x = torch.full((1, 3, 32, 32), 0.2)
    sigma = torch.full((1, 3, 32, 32), 0.03)
    # Mild per-iteration brightening: eight iterations still have slope > 1 at
    # 0.2. (Strong curves saturate toward 1 and *compress* noise instead.)
    brighten = torch.full((1, 24, 32, 32), -0.2)
    identity = torch.zeros((1, 24, 32, 32))
    assert loss_noise(x, sigma, identity, tau=2 / 255).item() == pytest.approx(0.0, abs=1e-8)
    assert loss_noise(x, sigma, brighten, tau=2 / 255).item() > 1.0


def test_training_smoke_writes_a_pinned_checkpoint(tmp_path):
    from sensorsim.train.nc_dce import TrainConfig, train

    cfg = TrainConfig(samples=4, size=64, epochs=2, batch=2, device="cpu", init="",
                      out=str(tmp_path / "m.pth"), log_dir=str(tmp_path / "log"))
    meta = train(cfg, say=lambda *_: None)
    assert (tmp_path / "m.pth").exists()
    assert len(meta["sha256"]) == 64
    assert (tmp_path / "log" / "nc_dce_loss.csv").exists()


def test_diverged_training_saves_no_checkpoint_and_the_arm_says_why(tmp_path, monkeypatch):
    import sensorsim.train.nc_dce as nc
    from sensorsim.restore import build_pipeline

    real = nc.total_loss
    calls = {"n": 0}

    def exploding(model, x, s, cfg):
        calls["n"] += 1
        loss, log = real(model, x, s, cfg)
        return (loss * float("nan"), log) if calls["n"] >= 3 else (loss, log)

    monkeypatch.setattr(nc, "total_loss", exploding)
    out = tmp_path / "bad.pth"
    meta = nc.train(nc.TrainConfig(samples=4, size=64, epochs=3, batch=2, device="cpu", init="",
                                   out=str(out), log_dir=str(tmp_path / "log")), say=lambda *_: None)
    assert meta["diverged"] and meta["sha256"] is None
    assert not out.exists()
    reason = build_pipeline({"kind": "nc_dce", "params": {"weights": str(out)}}).availability().reason
    assert "diverged" in reason


def test_curve_enhancer_refuses_non_finite_weights(tmp_path):
    from sensorsim.enhance.ncdce import CurveEnhancer
    from sensorsim.restore.learned.arch import build_dce

    state = build_dce(4).state_dict()
    state["e_conv3.bias"][0] = float("nan")
    torch.save(state, tmp_path / "nan.pth")
    with pytest.raises(ValueError, match="non-finite"):
        CurveEnhancer(tmp_path / "nan.pth", device="cpu")


def test_nag_arms_report_unavailable_without_weights():
    from sensorsim.restore import build_pipeline

    for spec in ({"kind": "nc_dce", "params": {"weights": "missing.pth"}},
                 {"kind": "g_esrgan", "params": {"weights": None}},
                 {"kind": "nag", "params": {"dce_weights": "missing.pth", "sr_weights": "missing.pth"}}):
        availability = build_pipeline(spec).availability()
        assert not availability
        assert "missing" in availability.reason or "path" in availability.reason


def test_nag_arm_returns_reference_shape_with_tiny_models(tmp_path):
    from sensorsim.restore import build_pipeline, run_pipeline
    from sensorsim.restore.learned.arch import build_dce, build_srvgg

    dce = tmp_path / "dce.pth"
    sr = tmp_path / "sr.pth"
    torch.save(build_dce(4).state_dict(), dce)
    torch.save({"params": build_srvgg().state_dict()}, sr)

    raw = scene_to_raw(Scene("t", "natural", np.full((64, 64, 3), 0.3, np.float32), {}))
    degraded, rec = degrade(raw, DegradationParams(alpha=0.2, sigma_r_dn=2.0, r_psf=0.8, binning=2, bit_depth=12), seed=1)
    ctx = RestoreContext("t", raw.shape, rec.params, rec.sensor)
    arm = build_pipeline({"kind": "nag", "params": {"dce_weights": str(dce), "sr_weights": str(sr),
                                                   "sr_kind": "general", "device": "cpu"}})
    result = run_pipeline(arm, degraded, ctx)
    assert result.ok, result.error
    assert result.image.shape == (64, 64, 3) and result.image.dtype == np.uint8
