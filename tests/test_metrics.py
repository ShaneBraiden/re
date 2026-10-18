"""Stage 3: every metric family, plus the honesty rules around gaps."""

from __future__ import annotations

import numpy as np
import pytest

from sensorsim.data.synthetic import natural_scene, resolution_scene, linear_to_srgb
from sensorsim.evaluate import (
    DIRECTION,
    FAMILY_OF,
    evaluate_frame,
    fidelity,
    machine_task,
    missing_metrics,
    perceptual,
    to_dataframe,
    unavailable_rows,
)


@pytest.fixture
def image() -> np.ndarray:
    scene = natural_scene(size=(192, 192))
    return (linear_to_srgb(scene.linear_rgb) * 255).round().astype(np.uint8)


# ---------------------------------------------------------------- fidelity ---

def test_psnr_of_identical_images_is_infinite(image):
    assert fidelity.psnr(image, image) == float("inf")


def test_psnr_falls_as_noise_rises(image):
    rng = np.random.default_rng(0)
    scores = []
    for sigma in (2, 8, 20):
        noisy = np.clip(image + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)
        scores.append(fidelity.psnr(noisy, image))
    assert scores == sorted(scores, reverse=True)


def test_ssim_of_identical_images_is_one(image):
    assert fidelity.ssim(image, image) == pytest.approx(1.0, abs=1e-6)


def test_ssim_is_symmetric(image):
    rng = np.random.default_rng(1)
    other = np.clip(image + rng.normal(0, 10, image.shape), 0, 255).astype(np.uint8)
    assert fidelity.ssim(image, other) == pytest.approx(fidelity.ssim(other, image), abs=1e-9)


def test_delta_e_catches_a_colour_cast_that_ssim_shrugs_at(image):
    cast = image.astype(np.float32).copy()
    cast[..., 2] *= 0.75  # pull the blues - a white balance failure
    cast = cast.astype(np.uint8)
    assert fidelity.delta_e_76(cast, image) > 3.0
    assert fidelity.ssim(cast, image) > 0.85


def test_mismatched_shapes_are_an_error(image):
    with pytest.raises(ValueError, match="match in shape"):
        fidelity.psnr(image, image[:-2])


# -------------------------------------------------------------- perceptual ---

def test_noise_sigma_recovers_a_known_sigma_on_grayscale():
    flat = np.full((256, 256), 128.0)
    rng = np.random.default_rng(2)
    for sigma in (3.0, 10.0):
        noisy = np.clip(flat + rng.normal(0, sigma, flat.shape), 0, 255).astype(np.uint8)
        assert perceptual.noise_sigma(noisy) == pytest.approx(sigma, rel=0.12)


def test_noise_sigma_on_rgb_measures_luma_noise():
    """Independent per-channel noise combines to 0.75x on luma. That is the
    quantity a viewer perceives, and the docstring says so - this pins it."""
    luma_weight = np.sqrt(0.2126**2 + 0.7152**2 + 0.0722**2)
    flat = np.full((256, 256, 3), 128.0)
    rng = np.random.default_rng(2)
    for sigma in (3.0, 10.0):
        noisy = np.clip(flat + rng.normal(0, sigma, flat.shape), 0, 255).astype(np.uint8)
        assert perceptual.noise_sigma(noisy) == pytest.approx(sigma * luma_weight, rel=0.12)


def test_sharpness_rises_with_detail(image):
    import cv2

    blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
    assert perceptual.sharpness(image) > perceptual.sharpness(blurred)


def test_nr_quality_punishes_noise_and_rewards_detail(image):
    import cv2

    rng = np.random.default_rng(3)
    blurred = cv2.GaussianBlur(image, (0, 0), 2.5)
    noisy = np.clip(image + rng.normal(0, 18, image.shape), 0, 255).astype(np.uint8)
    assert perceptual.nr_quality(image) > perceptual.nr_quality(blurred)
    assert perceptual.nr_quality(image) > perceptual.nr_quality(noisy)


def test_optional_metrics_report_why_they_cannot_run():
    reasons = perceptual.unavailable()
    for name, reason in reasons.items():
        assert reason, f"{name} is unavailable with no explanation"
        assert name in reason.lower() or "needs" in reason.lower()


# ------------------------------------------------------------ machine task ---

def test_sift_matches_an_image_to_itself(image):
    result = machine_task.sift_correspondence(image, image)
    assert result["sift_matches"] > 20
    assert result["sift_inlier_ratio"] > 0.9


def test_sift_inlier_ratio_falls_when_structure_is_destroyed(image):
    import cv2

    smeared = cv2.GaussianBlur(image, (0, 0), 6.0)
    intact = machine_task.sift_correspondence(image, image)["sift_inlier_ratio"]
    damaged = machine_task.sift_correspondence(smeared, image)["sift_inlier_ratio"]
    assert damaged < intact


def test_sift_on_a_blank_frame_returns_zeros_not_an_error(image):
    blank = np.full_like(image, 128)
    result = machine_task.sift_correspondence(blank, image)
    assert result["sift_matches"] == 0.0
    assert result["sift_inlier_ratio"] == 0.0


def test_edge_acutance_is_one_for_an_identical_image(image):
    assert machine_task.edge_acutance(image, image) == pytest.approx(1.0, abs=1e-5)


def test_edge_acutance_exceeds_one_when_oversharpened(image):
    import cv2

    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    sharpened = np.clip(image.astype(np.float32) * 1.8 - blurred.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    assert machine_task.edge_acutance(sharpened, image) > 1.0


def test_mtf50_on_a_crop_measures_resolution():
    import cv2

    scene = resolution_scene(size=(256, 256))
    rgb = (linear_to_srgb(scene.linear_rgb) * 255).round().astype(np.uint8)
    crop = tuple(scene.crops["slanted_edge"])
    blurred = cv2.GaussianBlur(rgb, (0, 0), 2.0)
    assert machine_task.mtf50_on_crop(rgb, crop) > machine_task.mtf50_on_crop(blurred, crop)


def test_mtf50_on_a_flat_crop_is_nan(image):
    flat = np.full_like(image, 200)
    assert np.isnan(machine_task.mtf50_on_crop(flat, (0, 0, 64, 64)))


def test_levenshtein_basics():
    assert machine_task._levenshtein("kitten", "sitting") == 3
    assert machine_task._levenshtein("same", "same") == 0
    assert machine_task._levenshtein("", "abc") == 3


# ----------------------------------------------------------------- tables ---

def test_evaluate_frame_emits_long_format(image):
    rows = evaluate_frame(
        image, image,
        image_id="probe", degradation_params_hash="deadbeef", pipeline="classical_isp",
        severity=0.4,
    )
    frame = to_dataframe(rows)
    assert set(frame.columns) >= {
        "image_id", "degradation_params_hash", "pipeline",
        "metric_family", "metric_name", "value", "severity", "status",
    }
    assert set(frame["metric_family"]) <= {"fidelity", "perceptual", "machine_task"}
    assert (frame["severity"] == 0.4).all()
    # Long format: adding a metric adds rows, never columns.
    assert len(frame) == frame["metric_name"].nunique()


def test_unavailable_pipelines_get_explicit_rows():
    rows = unavailable_rows(
        image_id="probe", degradation_params_hash="deadbeef", pipeline="nafnet",
        reason="unavailable: needs torch",
    )
    assert len(rows) == 3
    assert all(row.status == "unavailable" for row in rows)
    assert all("torch" in row.note for row in rows)
    assert all(np.isnan(row.value) for row in rows)


def test_every_metric_declares_a_family_and_direction(image):
    rows = evaluate_frame(image, image, image_id="p", degradation_params_hash="h", pipeline="x")
    for row in rows:
        assert row.metric_name in DIRECTION, f"{row.metric_name} has no declared direction"
        assert FAMILY_OF[row.metric_name] == row.metric_family


def test_missing_metrics_are_enumerable():
    reasons = missing_metrics()
    assert isinstance(reasons, dict)
    for name, reason in reasons.items():
        assert reason
