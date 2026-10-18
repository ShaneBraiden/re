"""The noise model is the study's foundation, so it gets the strictest tests.

If any test in this file fails, every recovery number downstream is wrong -
not slightly wrong, but wrong in the direction that flatters restoration.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensorsim.degrader.noise import (
    SensorNoiseModel,
    apply_photon_noise,
    estimate_read_noise,
    fit_noise_params,
    photon_transfer_curve,
)
from sensorsim.degrader.validate import self_consistency


@pytest.fixture
def model() -> SensorNoiseModel:
    return SensorNoiseModel(full_well=15000.0, sigma_read_dn=3.0, dark_current_e=0.0, prnu=0.0)


def _stack(model: SensorNoiseModel, level: float, n: int = 32, size: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = np.full((size, size), level)
    return np.stack([apply_photon_noise(flat, model, rng=rng) for _ in range(n)])


def test_variance_scales_with_signal(model):
    """Poisson variance equals its mean. This is the whole model.

    A uniform Gaussian would give equal variance at both levels; the ratio here
    must track the ratio of the signal levels.
    """
    low = _stack(model, 0.1).var(axis=0, ddof=1).mean()
    high = _stack(model, 0.6, seed=1).var(axis=0, ddof=1).mean()
    assert high > low * 3, f"variance barely grew with signal: {low:.3e} -> {high:.3e}"
    assert high / low == pytest.approx(6.0, rel=0.25)


def test_photon_transfer_recovers_full_well(model):
    frames = np.concatenate(
        [_stack(model, level, n=48, seed=i) for i, level in enumerate((0.05, 0.2, 0.4, 0.7))], axis=2
    )
    mean, variance, _ = photon_transfer_curve(frames, n_bins=12)
    fit = fit_noise_params(mean, variance)
    assert fit["r_squared"] > 0.95
    assert 1.0 / fit["slope"] == pytest.approx(model.full_well, rel=0.15)


def test_read_noise_is_not_rectified_at_black(model):
    """Dark frames must keep their negative half.

    Clipping at zero turns a zero-mean Gaussian into a half-normal: the mean
    rises and the measured sigma falls to ~0.58 of the truth. It would do so
    exactly where the study lives, in the darkest frames.
    """
    darks = _stack(model, 0.0, n=64, seed=7)
    expected = model.sigma_read_dn / (model.white_level - model.black_level)
    assert estimate_read_noise(darks) == pytest.approx(expected, rel=0.1)
    assert darks.min() < 0.0, "no sample fell below black level - the pedestal headroom is gone"
    assert abs(float(darks.mean())) < 0.2 * expected


def test_photon_scale_darkens_proportionally(model):
    flat = np.full((96, 96), 0.5)
    rng = np.random.default_rng(0)
    full = apply_photon_noise(flat, model, photon_scale=1.0, rng=rng).mean()
    quarter = apply_photon_noise(flat, model, photon_scale=0.25, rng=rng).mean()
    assert quarter == pytest.approx(full * 0.25, rel=0.05)


def test_binning_improves_snr(model):
    """Binning b x b deepens the well by b^2, so SNR improves by b."""
    flat = np.full((96, 96), 0.2)
    rng = np.random.default_rng(3)

    def snr(well_multiplier: float) -> float:
        frames = np.stack(
            [apply_photon_noise(flat, model, well_multiplier=well_multiplier, rng=rng) for _ in range(32)]
        )
        return float(frames.mean() / frames.std(axis=0, ddof=1).mean())

    assert snr(4.0) / snr(1.0) == pytest.approx(2.0, rel=0.2)


def test_saturation_clips_in_the_charge_domain(model):
    flat = np.full((64, 64), 1.5)  # over full well
    out = apply_photon_noise(flat, model, rng=np.random.default_rng(0))
    assert out.max() <= 1.0


def test_prnu_is_fixed_pattern(model):
    """The same PRNU field must produce the same gain error in every frame."""
    prnu_model = SensorNoiseModel(
        full_well=model.full_well, sigma_read_dn=0.0, dark_current_e=0.0, prnu=0.05
    )
    field = np.random.default_rng(0).normal(1.0, 0.05, size=(64, 64))
    flat = np.full((64, 64), 0.5)
    rng = np.random.default_rng(1)
    frames = np.stack([apply_photon_noise(flat, prnu_model, rng=rng, prnu_field=field) for _ in range(24)])

    # Averaging kills shot noise; a fixed pattern survives and stays correlated
    # with the field that produced it.
    averaged = frames.mean(axis=0)
    correlation = np.corrcoef(averaged.ravel(), field.ravel())[0, 1]
    assert correlation > 0.8


def test_gain_conversion_is_consistent(model):
    assert model.gain_e_per_dn == pytest.approx(15000.0 / (16383.0 - 512.0))
    assert model.sigma_read_e == pytest.approx(model.sigma_read_dn * model.gain_e_per_dn)


def test_white_level_must_exceed_black():
    with pytest.raises(ValueError, match="white_level"):
        SensorNoiseModel(white_level=100.0, black_level=500.0).gain_e_per_dn


@pytest.mark.parametrize("sigma_read", [1.0, 4.0, 10.0])
def test_self_consistency_gate_passes(sigma_read):
    model = SensorNoiseModel(full_well=15000.0, sigma_read_dn=sigma_read, dark_current_e=0.0, prnu=0.0)
    report = self_consistency(model)
    assert report.passed, report.summary()


def test_photon_transfer_needs_a_stack(model):
    with pytest.raises(ValueError, match="stack"):
        photon_transfer_curve(np.zeros((64, 64)))
