"""Stage 4 - recovery curves, knee detection, divergence, significance, figures."""

from sensorsim.analyze.curves import (
    CurveFit,
    aggregate,
    curve_table,
    fit_curve,
    index_to_base,
    recovery_ratio,
)
from sensorsim.analyze.knee import KneeResult, find_knee, kneedle, knee_table, piecewise_breakpoint
from sensorsim.analyze.stats import (
    bootstrap_ci,
    compare_pipelines,
    divergence_test,
    holm_bonferroni,
    power_note,
)

__all__ = [
    "CurveFit",
    "KneeResult",
    "aggregate",
    "bootstrap_ci",
    "compare_pipelines",
    "curve_table",
    "divergence_test",
    "find_knee",
    "fit_curve",
    "holm_bonferroni",
    "index_to_base",
    "knee_table",
    "kneedle",
    "piecewise_breakpoint",
    "power_note",
    "recovery_ratio",
]
