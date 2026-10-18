"""Stage 4: recovery ratios, curve fitting, knee detection and significance.

These tests use hand-built tables with known answers. That is deliberate: an
analysis tested only against real experiment output can only tell you that it
did not crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sensorsim.analyze.curves import (
    aggregate,
    exp_saturation,
    fit_curve,
    index_to_base,
    recovery_ratio,
    to_distance_from_target,
)
from sensorsim.analyze.knee import find_knee, kneedle, piecewise_breakpoint
from sensorsim.analyze.stats import (
    bootstrap_ci,
    compare_pipelines,
    divergence_test,
    holm_bonferroni,
    power_note,
)


def make_table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["status"] = frame.get("status", "ok")
    frame["metric_family"] = "fidelity"
    frame["scene_kind"] = "natural"
    return frame


@pytest.fixture
def simple_table() -> pd.DataFrame:
    """Clean = 30 dB, degraded falls to 20 dB, one arm recovers exactly half."""
    rows = []
    for severity, degraded in [(0.0, 30.0), (0.5, 25.0), (1.0, 20.0)]:
        rows.append({"image_id": "a", "degradation_params_hash": f"h{severity}",
                     "pipeline": "classical_isp", "metric_name": "psnr",
                     "severity": severity, "value": degraded})
        halfway = degraded + (30.0 - degraded) * 0.5
        rows.append({"image_id": "a", "degradation_params_hash": f"h{severity}",
                     "pipeline": "half_recovery", "metric_name": "psnr",
                     "severity": severity, "value": halfway})
    return make_table(rows)


# ------------------------------------------------------------- recovery ---

def test_recovery_ratio_is_zero_for_the_baseline(simple_table):
    recovery = recovery_ratio(simple_table)
    baseline = recovery[recovery["pipeline"] == "classical_isp"]
    assert np.allclose(baseline["recovery"], 0.0)


def test_recovery_ratio_recovers_a_known_fraction(simple_table):
    recovery = recovery_ratio(simple_table)
    arm = recovery[recovery["pipeline"] == "half_recovery"]
    assert np.allclose(arm["recovery"], 0.5)


def test_conditions_with_no_measurable_deficit_are_dropped():
    """The clean condition has zero deficit, so R there is 0/0 - it must not
    appear as a data point, and it must not appear as a huge one either."""
    rows = []
    for severity, degraded in [(0.0, 30.0), (0.02, 29.98), (1.0, 20.0)]:
        rows.append({"image_id": "a", "degradation_params_hash": f"h{severity}",
                     "pipeline": "classical_isp", "metric_name": "psnr",
                     "severity": severity, "value": degraded})
        rows.append({"image_id": "a", "degradation_params_hash": f"h{severity}",
                     "pipeline": "arm", "metric_name": "psnr",
                     "severity": severity, "value": degraded + 0.5})
    recovery = recovery_ratio(make_table(rows))
    severities = sorted(set(recovery["severity"]))
    assert 0.0 not in severities
    assert 0.02 not in severities, "a near-zero deficit produced a ratio anyway"
    assert recovery["recovery"].abs().max() < 10


def test_missing_clean_condition_is_an_error():
    rows = [{"image_id": "a", "degradation_params_hash": "h", "pipeline": "classical_isp",
             "metric_name": "psnr", "severity": 0.5, "value": 25.0}]
    with pytest.raises(ValueError, match="clean condition"):
        recovery_ratio(make_table(rows))


def test_missing_baseline_is_an_error(simple_table):
    with pytest.raises(ValueError, match="baseline"):
        recovery_ratio(simple_table, baseline="not_a_pipeline")


def test_target_metrics_become_distance_from_target():
    frame = make_table([{"image_id": "a", "degradation_params_hash": "h", "pipeline": "p",
                         "metric_name": "edge_acutance", "severity": 0.5, "value": 1.4}])
    out = to_distance_from_target(frame)
    assert out["metric_name"].iloc[0] == "edge_acutance_error"
    assert out["value"].iloc[0] == pytest.approx(0.4)


def test_index_to_base_flips_lower_is_better_metrics():
    rows = []
    for severity, value in [(0.0, 10.0), (1.0, 20.0)]:  # delta_e rising = worse
        rows.append({"image_id": "a", "degradation_params_hash": "h", "pipeline": "p",
                     "metric_name": "delta_e", "severity": severity, "value": value})
    agg = aggregate(make_table(rows), value_column="value")
    indexed = index_to_base(agg)
    worse = indexed[indexed["severity"] == 1.0]["indexed"].iloc[0]
    assert worse < 100, "a worsening lower-is-better metric must index downward"


def test_aggregate_reports_bootstrap_intervals():
    rows = [
        {"image_id": f"img{i}", "degradation_params_hash": "h", "pipeline": "p",
         "metric_name": "psnr", "severity": 0.5, "value": 20.0 + i}
        for i in range(8)
    ]
    agg = aggregate(make_table(rows), value_column="value")
    row = agg.iloc[0]
    assert row["n"] == 8
    assert row["ci_low"] < row["mean"] < row["ci_high"]


# ---------------------------------------------------------------- curves ---

def test_fit_curve_recovers_a_known_saturation():
    x = np.linspace(0.05, 1.0, 20)
    y = exp_saturation(x, 0.8, 4.0, 0.1)
    fit = fit_curve(x, y, model="exp_saturation")
    assert fit.r_squared > 0.999
    assert fit.params[0] == pytest.approx(0.8, rel=0.05)
    assert fit.params[1] == pytest.approx(4.0, rel=0.05)


def test_fit_curve_picks_a_model_by_aic():
    x = np.linspace(0.05, 1.0, 24)
    y = exp_saturation(x, 1.0, 6.0, 0.0) + np.random.default_rng(0).normal(0, 0.005, x.size)
    assert fit_curve(x, y, model="auto").model == "exp_saturation"


def test_fit_curve_refuses_a_constant_series():
    x = np.linspace(0, 1, 10)
    with pytest.raises(RuntimeError, match="constant"):
        fit_curve(x, np.full_like(x, 0.3))


def test_fit_curve_needs_enough_points():
    with pytest.raises(ValueError, match="at least 4"):
        fit_curve(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


# ------------------------------------------------------------------ knee ---

def test_kneedle_finds_an_obvious_elbow():
    x = np.linspace(0, 1, 41)
    y = np.where(x < 0.5, x * 2, 1.0)  # rises then flattens hard at 0.5
    assert kneedle(x, y) == pytest.approx(0.5, abs=0.1)


def test_kneedle_returns_nan_for_a_straight_line():
    x = np.linspace(0, 1, 20)
    assert np.isnan(kneedle(x, 2.0 * x + 1.0))


def test_piecewise_finds_a_breakpoint():
    x = np.linspace(0, 1, 41)
    y = np.where(x < 0.6, x, 0.6 + (x - 0.6) * 0.05)
    breakpoint_, _ = piecewise_breakpoint(x, y)
    assert breakpoint_ == pytest.approx(0.6, abs=0.08)


def test_piecewise_returns_nan_for_a_flat_line():
    x = np.linspace(0, 1, 20)
    breakpoint_, _ = piecewise_breakpoint(x, np.full_like(x, 0.5))
    assert np.isnan(breakpoint_)


def test_find_knee_agrees_with_itself_on_a_clean_elbow():
    x = np.linspace(0.02, 1.0, 40)
    y = exp_saturation(x, 1.0, 8.0, 0.0)
    result = find_knee(x, y)
    assert np.isfinite(result.consensus)
    assert result.is_identifiable(), result.summary()


def test_find_knee_reports_disagreement_rather_than_inventing_one():
    """A curve with no knee must come back as 'not identifiable', not as a
    number. 'We could not identify a knee' has to stay available as a finding."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 40)
    y = rng.normal(0.5, 0.3, x.size)
    result = find_knee(x, y)
    assert not result.is_identifiable() or result.agreement > 0.2


# ----------------------------------------------------------------- stats ---

def test_holm_bonferroni_is_monotone_and_conservative():
    adjusted, rejected = holm_bonferroni([0.001, 0.02, 0.04, 0.5])
    assert adjusted == sorted(adjusted)
    assert all(a >= p for a, p in zip(adjusted, [0.001, 0.02, 0.04, 0.5]))
    assert rejected[0] is True
    assert rejected[-1] is False


def test_holm_bonferroni_handles_the_empty_case():
    assert holm_bonferroni([]) == ([], [])


def test_compare_pipelines_finds_a_real_difference():
    rows = []
    rng = np.random.default_rng(0)
    for i in range(12):
        base = 20.0 + rng.normal(0, 0.5)
        rows.append({"image_id": f"img{i}", "degradation_params_hash": "h",
                     "pipeline": "classical_isp", "metric_name": "psnr",
                     "severity": 0.5, "value": base})
        rows.append({"image_id": f"img{i}", "degradation_params_hash": "h",
                     "pipeline": "better", "metric_name": "psnr",
                     "severity": 0.5, "value": base + 2.0})
    tests = compare_pipelines(make_table(rows), metric="psnr", baseline="classical_isp")
    assert len(tests) == 1
    assert tests["significant"].iloc[0]
    assert tests["mean_difference"].iloc[0] == pytest.approx(2.0, abs=0.1)


def test_compare_pipelines_reports_too_few_pairs_rather_than_a_p_value():
    rows = [
        {"image_id": "a", "degradation_params_hash": "h", "pipeline": p,
         "metric_name": "psnr", "severity": 0.5, "value": v}
        for p, v in [("classical_isp", 20.0), ("other", 22.0)]
    ]
    tests = compare_pipelines(make_table(rows), metric="psnr", baseline="classical_isp")
    assert "fewer than 3" in tests["note"].iloc[0]
    assert not tests["significant"].iloc[0]


def test_divergence_test_detects_opposing_directions():
    rows = []
    for i, severity in enumerate(np.linspace(0, 1, 12)):
        rows.append({"image_id": f"img{i}", "degradation_params_hash": f"h{i}", "pipeline": "p",
                     "metric_name": "psnr", "severity": severity, "value": 30 - 10 * severity})
        rows.append({"image_id": f"img{i}", "degradation_params_hash": f"h{i}", "pipeline": "p",
                     "metric_name": "nr_quality", "severity": severity, "value": 1 + 2 * severity})
    result = divergence_test(make_table(rows))
    assert result["spearman_rho"] < 0
    assert result["diverges"]


def test_divergence_test_says_so_when_there_is_not_enough_data():
    rows = [{"image_id": "a", "degradation_params_hash": "h", "pipeline": "p",
             "metric_name": "psnr", "severity": 0.5, "value": 20.0}]
    assert "note" in divergence_test(make_table(rows))


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(10, 2, 60)
    point, low, high = bootstrap_ci(values)
    assert low < point < high
    assert point == pytest.approx(10, abs=1.0)


def test_bootstrap_ci_degenerate_cases():
    assert all(np.isnan(v) for v in bootstrap_ci(np.array([])))
    assert bootstrap_ci(np.array([4.0])) == (4.0, 4.0, 4.0)


def test_power_note_states_the_detectable_effect():
    note = power_note(6)
    assert "dz" in note
    assert "does not rule out" in note
    assert "too few" in power_note(2)
