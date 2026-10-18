"""Stage 1 as a whole: the forward model, its ordering, and its determinism."""

from __future__ import annotations

import numpy as np
import pytest

from sensorsim.data.synthetic import resolution_scene, scene_to_raw
from sensorsim.degrader import (
    DegradationParams,
    apply_psf_mosaic,
    bin_bayer,
    degrade,
    make_psf,
    mtf50,
    mtf_from_psf,
    quantize,
    slanted_edge_mtf,
    sweep,
)
from sensorsim.degrader.sampling import effective_resolution
from sensorsim.io.raw import RawImage


@pytest.fixture
def clean() -> RawImage:
    return scene_to_raw(resolution_scene(size=(128, 128)))


def test_identity_condition_barely_changes_the_frame(clean):
    out, record = degrade(clean, DegradationParams(), seed=0)
    assert out.shape == clean.shape
    # Shot noise at full exposure is still real, so this is "close", not "equal".
    assert float(np.abs(out.data - clean.data).mean()) < 0.01
    assert record.params["severity"] == 0.0


def test_alpha_darkens_the_frame(clean):
    means = []
    for alpha in (1.0, 0.5, 0.1):
        out, _ = degrade(clean, DegradationParams(alpha=alpha), seed=0)
        means.append(float(np.clip(out.data, 0, 1).mean()))
    assert means[1] == pytest.approx(means[0] * 0.5, rel=0.05)
    assert means[2] == pytest.approx(means[0] * 0.1, rel=0.08)


def test_binning_reduces_resolution(clean):
    out, record = degrade(clean, DegradationParams(binning=2), seed=0)
    assert out.shape == effective_resolution(clean.shape, 2)
    assert tuple(record.output_shape) == out.shape
    assert out.shape[0] * 2 == clean.shape[0]


def test_binning_preserves_the_cfa_phase():
    """A binned mosaic must still start with the same colour in the corner."""
    raw = np.zeros((16, 16), dtype=np.float32)
    raw[0::2, 0::2] = 1.0  # R sites only, for an RGGB pattern
    binned = bin_bayer(raw, 2, "RGGB")
    assert binned[0::2, 0::2].mean() == pytest.approx(1.0)
    assert binned[1::2, 1::2].mean() == pytest.approx(0.0)


def test_quantization_lands_on_the_code_grid():
    values = np.linspace(0, 1, 500)
    out = quantize(values, bit_depth=8)
    assert len(np.unique(out)) <= 256
    assert np.all(np.abs(out - values) <= 1.0 / 255 / 2 + 1e-6)


def test_quantize_keeps_negative_headroom():
    values = np.array([-0.02, 0.0, 0.5])
    out = quantize(values, bit_depth=12, floor=-0.03)
    assert out[0] < 0.0


@pytest.mark.parametrize("kind", ["gaussian", "airy"])
def test_psf_is_normalised(kind):
    psf = make_psf(kind, 1.5)
    assert psf.sum() == pytest.approx(1.0)
    assert psf.shape[0] % 2 == 1


def test_larger_psf_means_lower_mtf50():
    resolutions = [mtf50(*mtf_from_psf(make_psf("gaussian", r))) for r in (0.5, 1.0, 2.0)]
    assert resolutions[0] > resolutions[1] > resolutions[2]


def test_psf_does_not_mix_colour_channels():
    """Blur must act per channel; a red-only mosaic must stay red-only."""
    raw = np.zeros((32, 32), dtype=np.float32)
    raw[0::2, 0::2] = 1.0
    blurred = apply_psf_mosaic(raw, make_psf("gaussian", 1.5), "RGGB")
    assert blurred[0::2, 0::2].mean() == pytest.approx(1.0, abs=1e-4)
    assert blurred[1::2, 1::2].max() == pytest.approx(0.0, abs=1e-6)


def test_psf_preserves_energy_on_a_flat_field():
    raw = np.full((32, 32), 0.4, dtype=np.float32)
    blurred = apply_psf_mosaic(raw, make_psf("gaussian", 2.0), "RGGB")
    assert blurred.mean() == pytest.approx(0.4, rel=1e-4)


def test_slanted_edge_mtf_tracks_blur():
    scene = resolution_scene(size=(256, 256))
    gray = scene.linear_rgb[..., 1]
    x, y, w, h = scene.crops["slanted_edge"]
    sharp = gray[y : y + h, x : x + w]
    blurred = apply_psf_mosaic(gray, make_psf("gaussian", 2.0), "RGGB")[y : y + h, x : x + w]

    sharp_mtf50 = mtf50(*slanted_edge_mtf(sharp))
    blurred_mtf50 = mtf50(*slanted_edge_mtf(blurred))
    assert sharp_mtf50 > blurred_mtf50


def test_degradation_is_deterministic(clean):
    params = DegradationParams(alpha=0.2, sigma_r_dn=5.0, r_psf=1.0, binning=2, bit_depth=10)
    first, _ = degrade(clean, params, seed=42)
    second, _ = degrade(clean, params, seed=42)
    assert np.array_equal(first.data, second.data)

    other, _ = degrade(clean, params, seed=43)
    assert not np.array_equal(first.data, other.data)


def test_conditions_do_not_share_a_noise_realisation(clean):
    a, _ = degrade(clean, DegradationParams(alpha=0.5), seed=1)
    b, _ = degrade(clean, DegradationParams(alpha=0.5, sigma_r_dn=1.0), seed=1)
    assert not np.array_equal(a.data, b.data)


def test_ordering_matters_and_is_recorded(clean):
    """Blur after noise smooths the noise; the record must say which was used."""
    params = {"alpha": 0.1, "sigma_r_dn": 6.0, "r_psf": 2.0}
    optical, rec_o = degrade(clean, DegradationParams(**params, order="optical"), seed=0)
    listed, rec_l = degrade(clean, DegradationParams(**params, order="listed"), seed=0)

    assert rec_o.order == "optical"
    assert rec_l.order == "listed"

    def high_frequency_energy(frame):
        return float(np.abs(np.diff(frame.data, axis=1)).mean())

    assert high_frequency_energy(listed) < high_frequency_energy(optical)


def test_params_hash_is_stable_and_discriminating():
    a = DegradationParams(alpha=0.5, sigma_r_dn=2.0)
    b = DegradationParams(alpha=0.5, sigma_r_dn=2.0)
    c = DegradationParams(alpha=0.5, sigma_r_dn=2.5)
    assert a.params_hash == b.params_hash
    assert a.params_hash != c.params_hash


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"sigma_r_dn": -1.0}, "sigma_r_dn"),
        ({"binning": 0}, "binning"),
        ({"bit_depth": 20}, "bit_depth"),
        ({"order": "sideways"}, "order"),
    ],
)
def test_invalid_params_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DegradationParams(**kwargs)


def test_sweep_labels_alpha_severity_backwards():
    """Less light is more severe, so alpha's severity axis runs in reverse."""
    conditions = sweep(DegradationParams(), "alpha", [1.0, 0.5, 0.1])
    severities = [c.severity for c in conditions]
    assert severities[0] == 0.0
    assert severities[-1] == 1.0
    assert severities == sorted(severities)


def test_record_carries_the_measurable_consequences(clean):
    _, record = degrade(clean, DegradationParams(alpha=0.1, r_psf=1.5), seed=0)
    assert record.measured["mtf50_psf"] < 0.5
    assert record.measured["shot_limited_snr"] > 0
    assert record.sensor["gain_e_per_dn"] > 0
    assert record.image_id == clean.image_id
