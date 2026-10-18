"""The validation gate - calibration of the forward model against real hardware.

ARCHITECTURE.md section 3 makes this a hard precondition: before any experiment
runs, the degraded output must be checked against real captures from a low-end
sensor, on two axes - the noise variance versus intensity curve, and the MTF
slope. If they disagree beyond tolerance the model is recalibrated, not
excused. This module is that gate, and it is wired into the experiment runner
so a run cannot skip it by accident.

It does two jobs:

``calibrate_from_stack``
    Fit the sensor constants from a stack of real captures, so the model is
    *measured* rather than guessed.
``validate``
    Compare synthetic against measured and return a pass/fail report.

The distinction the paper's method section rests on: a simulation invents its
own noise; a study measures a real sensor and reproduces it within a stated
tolerance. Only the second is defensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from sensorsim.degrader.noise import (
    SensorNoiseModel,
    fit_noise_params,
    photon_transfer_curve,
)
from sensorsim.degrader.optics import mtf50, slanted_edge_mtf


@dataclass
class Tolerance:
    """How far synthetic may drift from measured before the gate fails.

    Defaults are deliberately tight enough to catch a wrong noise model and
    loose enough to survive honest measurement scatter. Loosen them in a config
    if your calibration captures are noisy - but say so in the paper.
    """

    ptc_slope_rel: float = 0.15       # inverse gain, within 15%
    read_noise_rel: float = 0.25      # read noise sigma, within 25%
    ptc_r_squared_min: float = 0.90   # the measured curve must actually be a line
    mtf50_rel: float = 0.15           # MTF50 within 15%


@dataclass
class CheckResult:
    name: str
    passed: bool
    measured: float
    synthetic: float
    tolerance: float
    detail: str = ""

    @property
    def relative_error(self) -> float:
        if self.measured == 0:
            return float("inf") if self.synthetic != 0 else 0.0
        return abs(self.synthetic - self.measured) / abs(self.measured)


@dataclass
class ValidationReport:
    """Outcome of the calibration gate."""

    checks: list[CheckResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [asdict(c) | {"relative_error": c.relative_error} for c in self.checks],
            "notes": self.notes,
        }

    def summary(self) -> str:
        if not self.checks:
            return "no checks run - the gate is not satisfied by an empty report"
        lines = [f"validation: {'PASS' if self.passed else 'FAIL'}"]
        for c in self.checks:
            mark = "ok  " if c.passed else "FAIL"
            lines.append(
                f"  [{mark}] {c.name:<18} measured={c.measured:.5g} synthetic={c.synthetic:.5g} "
                f"rel_err={c.relative_error:.1%} tol={c.tolerance:.0%}"
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)

    def require_pass(self) -> None:
        """Raise unless every check passed. Phase 1's gate, in one call."""
        if not self.passed:
            raise ValidationError(
                "forward sensor model does not match the measured sensor within tolerance.\n"
                + self.summary()
                + "\n\nRecalibrate with calibrate_from_stack() before running experiments, "
                "or state the mismatch explicitly and rerun with --skip-validation."
            )

    def write(self, path: str | Path) -> Path:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


class ValidationError(RuntimeError):
    """Raised when the forward model fails calibration against real captures."""


def calibrate_from_stack(
    frames: np.ndarray,
    black_level: float,
    white_level: float,
    dark_frames: np.ndarray | None = None,
    full_well_hint: float | None = None,
) -> tuple[SensorNoiseModel, dict[str, float]]:
    """Fit sensor constants from repeated captures of one static scene.

    Parameters
    ----------
    frames:
        Shape (N, H, W), N >= 8 captures of an identical scene, normalised to
        [0, 1] against ``black_level``/``white_level``. Tripod, fixed exposure.
    dark_frames:
        Optional lens-cap captures. When present they give a direct read-noise
        estimate that does not lean on extrapolating the PTC to zero.

    Returns
    -------
    (model, fit) - the calibrated sensor and the raw fit numbers behind it.
    """
    means, variances, _ = photon_transfer_curve(frames)
    fit = fit_noise_params(means, variances)

    span = white_level - black_level
    if span <= 0:
        raise ValueError("white_level must exceed black_level")

    # var_norm = mu_norm / full_well  =>  full_well = 1 / slope, in normalised units.
    slope = fit["slope"]
    if slope <= 0:
        raise ValueError(
            f"photon transfer slope is {slope:.4g} (expected > 0). Variance that does not "
            "grow with intensity means these frames are not shot-noise limited - check that "
            "the stack is really N captures of the same scene and not N crops of one."
        )
    full_well = full_well_hint if full_well_hint is not None else 1.0 / slope

    if dark_frames is not None:
        sigma_read_norm = float(np.sqrt(np.asarray(dark_frames, dtype=np.float64).var(axis=0, ddof=1).mean()))
    else:
        sigma_read_norm = fit["read_noise_sigma"]

    model = SensorNoiseModel(
        full_well=float(full_well),
        black_level=float(black_level),
        white_level=float(white_level),
        sigma_read_dn=float(sigma_read_norm * span),
        dark_current_e=0.0,
        prnu=0.0,  # temporal PTC cannot see fixed-pattern gain; fit it separately
    )
    return model, fit


def check_noise_curve(
    synthetic_frames: np.ndarray,
    measured_frames: np.ndarray,
    tol: Tolerance | None = None,
) -> list[CheckResult]:
    """Compare photon transfer curves: does synthetic noise grow like measured?"""
    tol = tol or Tolerance()
    s_mean, s_var, _ = photon_transfer_curve(synthetic_frames)
    m_mean, m_var, _ = photon_transfer_curve(measured_frames)
    s_fit = fit_noise_params(s_mean, s_var)
    m_fit = fit_noise_params(m_mean, m_var)

    results = [
        CheckResult(
            name="ptc_linearity",
            passed=m_fit["r_squared"] >= tol.ptc_r_squared_min,
            measured=m_fit["r_squared"],
            synthetic=s_fit["r_squared"],
            tolerance=tol.ptc_r_squared_min,
            detail="measured variance must be linear in intensity for the fit to mean anything",
        )
    ]
    for name, key, rel in (
        ("ptc_slope", "slope", tol.ptc_slope_rel),
        ("read_noise", "read_noise_sigma", tol.read_noise_rel),
    ):
        c = CheckResult(
            name=name,
            passed=False,
            measured=m_fit[key],
            synthetic=s_fit[key],
            tolerance=rel,
        )
        c.passed = c.relative_error <= rel
        results.append(c)
    return results


def check_mtf(
    synthetic_edge: np.ndarray,
    measured_edge: np.ndarray,
    tol: Tolerance | None = None,
) -> CheckResult:
    """Compare MTF50 from a slanted-edge patch in each frame."""
    tol = tol or Tolerance()
    s_f, s_m = slanted_edge_mtf(synthetic_edge)
    m_f, m_m = slanted_edge_mtf(measured_edge)
    c = CheckResult(
        name="mtf50",
        passed=False,
        measured=mtf50(m_f, m_m),
        synthetic=mtf50(s_f, s_m),
        tolerance=tol.mtf50_rel,
        detail="slanted-edge MTF50, cycles/pixel",
    )
    c.passed = c.relative_error <= tol.mtf50_rel
    return c


def validate(
    synthetic_frames: np.ndarray,
    measured_frames: np.ndarray,
    synthetic_edge: np.ndarray | None = None,
    measured_edge: np.ndarray | None = None,
    tol: Tolerance | None = None,
) -> ValidationReport:
    """Run the full gate: noise curves, and MTF when edge patches are supplied."""
    tol = tol or Tolerance()
    report = ValidationReport()
    for c in check_noise_curve(synthetic_frames, measured_frames, tol):
        report.add(c)

    if synthetic_edge is not None and measured_edge is not None:
        report.add(check_mtf(synthetic_edge, measured_edge, tol))
    else:
        report.notes.append(
            "MTF not checked - no slanted-edge patches supplied. Section 5 asks for a "
            "resolution chart in the calibration captures; without it the optical half "
            "of the model is unvalidated and the paper must say so."
        )

    report.notes.append(
        f"tolerances: ptc_slope {tol.ptc_slope_rel:.0%}, read_noise {tol.read_noise_rel:.0%}, "
        f"mtf50 {tol.mtf50_rel:.0%}, ptc R^2 >= {tol.ptc_r_squared_min}"
    )
    return report


def synthetic_photon_transfer(
    model: SensorNoiseModel,
    levels: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7),
    n_frames: int = 32,
    patch: int = 96,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Photon transfer curve of the model itself, for the method-section figure.

    Returns ``(mean, variance)`` per intensity bin, in normalised units - the
    same shape :func:`sensorsim.analyze.figures.photon_transfer_figure` plots,
    so the synthetic arm of that figure is always regenerable even before real
    calibration captures exist.
    """
    from sensorsim.degrader.noise import apply_photon_noise

    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_frames):
        flat = np.concatenate([np.full((patch, patch), lvl) for lvl in levels], axis=1)
        frames.append(apply_photon_noise(flat, model, rng=rng))
    mean, variance, _ = photon_transfer_curve(np.stack(frames), n_bins=len(levels) * 2)
    return mean, variance


def self_consistency(
    model: SensorNoiseModel,
    levels: tuple[float, ...] = (0.05, 0.15, 0.3, 0.5, 0.7),
    n_frames: int = 24,
    patch: int = 96,
    seed: int = 0,
) -> ValidationReport:
    """Check the noise generator against its own theory - no hardware needed.

    This is *not* a substitute for the hardware gate. It catches implementation
    bugs (a mis-scaled gain, noise applied in the wrong domain) by confirming
    that frames drawn from the model recover the constants the model was given.
    Passing here and failing :func:`validate` means the code is right and the
    calibration is wrong; failing here means the code is wrong.
    """
    from sensorsim.degrader.noise import apply_photon_noise, estimate_read_noise

    rng = np.random.default_rng(seed)
    frames, darks = [], []
    for _ in range(n_frames):
        flat = np.concatenate([np.full((patch, patch), lvl) for lvl in levels], axis=1)
        frames.append(apply_photon_noise(flat, model, rng=rng))
        darks.append(apply_photon_noise(np.zeros((patch, patch)), model, rng=rng))
    stack = np.stack(frames)

    means, variances, _ = photon_transfer_curve(stack, n_bins=len(levels) * 2)
    fit = fit_noise_params(means, variances)

    span = model.white_level - model.black_level
    expected_slope = 1.0 / model.full_well
    expected_read = model.sigma_read_dn / span
    # Read noise comes from the dark stack, not the PTC intercept. At any level
    # bright enough to fit a slope on, read variance is a rounding error next to
    # shot variance, so the intercept carries almost no information about it -
    # which is exactly why real calibration protocols shoot dark frames.
    measured_read = estimate_read_noise(np.stack(darks))

    report = ValidationReport()
    slope_check = CheckResult(
        name="ptc_slope_self",
        passed=False,
        measured=expected_slope,
        synthetic=fit["slope"],
        tolerance=0.10,
        detail="1/full_well recovered from generated frames",
    )
    slope_check.passed = slope_check.relative_error <= 0.10
    report.add(slope_check)

    read_check = CheckResult(
        name="read_noise_self",
        passed=False,
        measured=expected_read,
        synthetic=measured_read,
        tolerance=0.20,
        detail="read noise recovered from a generated dark stack",
    )
    read_check.passed = read_check.relative_error <= 0.20
    report.add(read_check)

    report.notes.append(
        f"PTC intercept for reference: {fit['read_noise_sigma']:.3g} (expected {expected_read:.3g}) "
        "- unreliable by construction at these levels; the dark stack is the estimator"
    )
    report.notes.append("self-consistency only - the hardware gate in validate() still applies")
    return report
