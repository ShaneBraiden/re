"""Stage 2: the interface contract, the registry, the cache, and the gap rule."""

from __future__ import annotations

import numpy as np
import pytest

from sensorsim.data.synthetic import mosaic, natural_scene, scene_to_raw
from sensorsim.degrader import DegradationParams, degrade
from sensorsim.io.raw import RawImage
from sensorsim.restore import (
    Pipeline,
    RestoreContext,
    build_bank,
    build_pipeline,
    cache_key,
    registered_kinds,
    run_pipeline,
)
from sensorsim.restore.classical import BAYER_CODES, demosaic

CLASSICAL = ["classical_isp", "nlm_denoise", "curve_tone", "rl_deconv"]
LEARNED = ["zero_dce", "nafnet", "restormer", "real_esrgan", "learned_isp"]


@pytest.fixture
def frame():
    clean = scene_to_raw(natural_scene(size=(128, 128)))
    params = DegradationParams(alpha=0.15, sigma_r_dn=5.0, r_psf=1.2, binning=2, bit_depth=10)
    degraded, record = degrade(clean, params, seed=0)
    context = RestoreContext(
        image_id=clean.image_id,
        target_shape=clean.shape,
        degradation=record.params,
        sensor=record.sensor,
    )
    return clean, degraded, context


def test_every_expected_kind_is_registered():
    assert set(CLASSICAL + LEARNED) <= set(registered_kinds())


@pytest.mark.parametrize("pattern, code", list(BAYER_CODES.items()))
def test_demosaic_code_maps_the_right_colour(pattern, code):
    """OpenCV names Bayer codes from a different corner than the sensor
    convention, so RGGB is COLOR_BayerBG2*. A wrong mapping swaps red and blue
    in every output and no other test would notice."""
    rgb = np.zeros((32, 32, 3), np.float32)
    rgb[..., 0] = 1.0  # pure red
    raw = RawImage(mosaic(rgb, pattern), cfa_pattern=pattern, wb_gains=(1.0, 1.0, 1.0))
    out = demosaic(raw)[4:-4, 4:-4]
    assert out[..., 0].mean() == pytest.approx(1.0, abs=0.02)
    assert out[..., 1].mean() == pytest.approx(0.0, abs=0.02)
    assert out[..., 2].mean() == pytest.approx(0.0, abs=0.02)


@pytest.mark.parametrize("kind", CLASSICAL)
def test_classical_pipelines_honour_the_interface(kind, frame):
    clean, degraded, context = frame
    pipeline = build_pipeline({"kind": kind})
    assert pipeline.availability(), pipeline.availability().reason

    result = run_pipeline(pipeline, degraded, context)
    assert result.ok, result.error
    assert result.image.dtype == np.uint8
    assert result.image.shape == (*clean.shape, 3)


def test_a_binned_frame_is_resampled_to_reference_resolution(frame):
    clean, degraded, context = frame
    assert degraded.shape != clean.shape  # the input really is smaller
    result = run_pipeline(build_pipeline({"kind": "classical_isp"}), degraded, context)
    assert result.image.shape[:2] == clean.shape


def test_a_pipeline_returning_the_wrong_shape_is_rejected(frame):
    clean, degraded, context = frame

    class Wrong(Pipeline):
        kind = "wrong_shape"

        def restore(self, degraded_raw, meta):
            return np.zeros((8, 8, 3), np.uint8)

    result = run_pipeline(Wrong(name="wrong"), degraded, context)
    assert not result.ok
    assert "reference" in result.error


def test_a_pipeline_returning_floats_is_rejected(frame):
    _, degraded, context = frame

    class Floaty(Pipeline):
        kind = "floaty"

        def restore(self, degraded_raw, meta):
            return np.zeros((*meta.target_shape, 3), np.float32)

    result = run_pipeline(Floaty(name="floaty"), degraded, context)
    assert not result.ok
    assert "rgb_uint8" in result.error


def test_a_raising_pipeline_does_not_end_the_sweep(frame):
    _, degraded, context = frame

    class Broken(Pipeline):
        kind = "broken"

        def restore(self, degraded_raw, meta):
            raise RuntimeError("checkpoint is corrupt")

    result = run_pipeline(Broken(name="broken"), degraded, context)
    assert not result.ok
    assert "corrupt" in result.error


@pytest.mark.parametrize("kind", LEARNED)
def test_learned_arms_report_unavailable_and_produce_nothing(kind, frame):
    """The rule that keeps the table honest: a missing checkpoint must never
    be quietly replaced by a different algorithm."""
    _, degraded, context = frame
    pipeline = build_pipeline({"kind": kind})
    availability = pipeline.availability()
    assert not availability
    assert "weights" in availability.reason or "torch" in availability.reason

    result = run_pipeline(pipeline, degraded, context)
    assert not result.ok
    assert result.error.startswith("unavailable:")
    assert result.image.max() == 0  # a placeholder, never a substituted result


def test_cache_key_changes_with_every_input(frame):
    clean, degraded, context = frame
    pipeline = build_pipeline({"kind": "classical_isp"})
    base = cache_key(degraded, pipeline, context)

    assert cache_key(degraded, pipeline, context) == base

    other_params = build_pipeline({"kind": "classical_isp", "params": {"denoise_strength": 0.9}})
    assert cache_key(degraded, other_params, context) != base

    nudged = degraded.replace(degraded.data + 0.01)
    assert cache_key(nudged, pipeline, context) != base


def test_cache_round_trips(tmp_path, frame):
    _, degraded, context = frame
    pipeline = build_pipeline({"kind": "classical_isp"})

    first = run_pipeline(pipeline, degraded, context, cache_dir=tmp_path)
    assert not first.cached
    second = run_pipeline(pipeline, degraded, context, cache_dir=tmp_path)
    assert second.cached
    assert np.array_equal(first.image, second.image)


def test_duplicate_pipeline_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build_bank([{"kind": "classical_isp"}, {"kind": "classical_isp"}])


def test_unknown_kind_names_what_is_available():
    with pytest.raises(ValueError, match="unknown pipeline kind"):
        build_pipeline({"kind": "definitely_not_a_pipeline"})


def test_gain_modes(frame):
    """Recorded gain matches brightness across arms; auto estimates it."""
    _, degraded, context = frame
    from sensorsim.restore.classical import estimate_gain

    assert estimate_gain(degraded, context, "recorded") == pytest.approx(1 / context.alpha)
    assert estimate_gain(degraded, context, "none") == 1.0
    assert estimate_gain(degraded, context, "auto") > 1.0
    with pytest.raises(ValueError, match="gain_mode"):
        estimate_gain(degraded, context, "telepathy")


def test_denoising_arm_actually_denoises():
    """A sanity floor on the metric the arm exists to move.

    Measured at binning 1: binning averages noise away in the sensor, so a
    binned frame arrives nearly noiseless and there is nothing left to test.
    """
    from sensorsim.evaluate.perceptual import noise_sigma

    clean = scene_to_raw(natural_scene(size=(192, 192)))
    params = DegradationParams(alpha=0.05, sigma_r_dn=5.0, r_psf=0.6, binning=1, bit_depth=12)
    degraded, record = degrade(clean, params, seed=0)
    context = RestoreContext(clean.image_id, clean.shape, record.params, record.sensor)

    plain = run_pipeline(build_pipeline({"kind": "classical_isp", "params": {"denoise_strength": 0.0}}),
                         degraded, context)
    denoised = run_pipeline(build_pipeline({"kind": "nlm_denoise"}), degraded, context)
    assert noise_sigma(denoised.image) < noise_sigma(plain.image) * 0.7


def test_auto_denoise_strength_backs_off_when_binning_did_the_work():
    """Binning improves SNR by b, so the filter must ease off by the same factor."""
    from sensorsim.restore.classical import NonLocalMeansDenoise

    def strength(binning: int) -> float:
        meta = RestoreContext("x", (64, 64), {"alpha": 0.1, "sigma_r_dn": 5.0, "binning": binning})
        return NonLocalMeansDenoise._auto_strength(meta)

    assert strength(2) == pytest.approx(strength(1) / 2, rel=1e-6)
    assert strength(1) > strength(2) > strength(3)


def test_deconvolution_recovers_resolution_lost_to_blur():
    """The arm that inverts known optics must actually raise measured MTF50."""
    from sensorsim.data.synthetic import resolution_scene
    from sensorsim.evaluate.machine_task import mtf50_on_crop

    scene = resolution_scene(size=(256, 256))
    clean = scene_to_raw(scene)
    crop = tuple(scene.crops["slanted_edge"])
    params = DegradationParams(alpha=0.8, sigma_r_dn=1.0, r_psf=1.6, binning=1, bit_depth=12)
    degraded, record = degrade(clean, params, seed=0)
    context = RestoreContext(clean.image_id, clean.shape, record.params, record.sensor)

    control = run_pipeline(build_pipeline({"kind": "classical_isp"}), degraded, context)
    deconv = run_pipeline(build_pipeline({"kind": "rl_deconv"}), degraded, context)
    assert mtf50_on_crop(deconv.image, crop) > mtf50_on_crop(control.image, crop)
