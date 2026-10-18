"""Knee detection - where recovery stops paying.

H1 claims the recovery curve has "an identifiable knee". Identifiable is the
load-bearing word, so this module locates it three independent ways and reports
whether they agree:

``kneedle``
    Satopaa et al., *Finding a "Kneedle" in a Haystack* (ICDCS 2011). Works on
    the raw points, no model assumed.
``max_curvature``
    Maximum curvature of the fitted analytic curve. Needs a model, and gives a
    continuous answer rather than one snapped to a sampled severity.
``piecewise``
    The breakpoint of the best two-segment linear fit. The most robust of the
    three when the curve is closer to two lines than to a smooth bend.

A knee the three methods disagree about is a knee the paper should not claim.
:attr:`KneeResult.agreement` is the spread between them, and
:meth:`KneeResult.is_identifiable` applies a stated threshold - so "we could not
identify a knee" stays available as a finding rather than being fitted away.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from sensorsim.analyze.curves import CurveFit, fit_curve


@dataclass
class KneeResult:
    """Where each method put the knee, and whether they agree."""

    kneedle: float = float("nan")
    max_curvature: float = float("nan")
    piecewise: float = float("nan")
    consensus: float = float("nan")
    agreement: float = float("nan")
    fit: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    #: Methods must agree within this fraction of the swept severity range.
    identifiability_threshold: float = 0.20

    def is_identifiable(self) -> bool:
        return bool(np.isfinite(self.agreement) and self.agreement <= self.identifiability_threshold)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["identifiable"] = self.is_identifiable()
        return d

    def summary(self) -> str:
        if not np.isfinite(self.consensus):
            return "no knee located by any method"
        verdict = "identifiable" if self.is_identifiable() else "NOT identifiable (methods disagree)"
        return (
            f"knee at severity {self.consensus:.3f} [{verdict}] - "
            f"kneedle {self.kneedle:.3f}, curvature {self.max_curvature:.3f}, "
            f"piecewise {self.piecewise:.3f}, spread {self.agreement:.1%} of range"
        )


def kneedle(x: np.ndarray, y: np.ndarray, sensitivity: float = 1.0) -> float:
    """Point of maximum distance from the chord joining the endpoints.

    Normalises both axes, subtracts the straight line between first and last
    point, and takes the extreme of the difference curve. Sign-agnostic, so it
    handles both a rising saturation and a falling collapse.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return float("nan")

    order = np.argsort(x)
    x, y = x[order], y[order]
    x_span = x[-1] - x[0]
    y_span = y.max() - y.min()
    if x_span <= 0 or y_span <= 0:
        return float("nan")

    xn = (x - x[0]) / x_span
    yn = (y - y.min()) / y_span
    difference = yn - (yn[0] + (yn[-1] - yn[0]) * xn)

    index = int(np.argmax(np.abs(difference)))
    if abs(difference[index]) < 0.01 * sensitivity:
        return float("nan")  # essentially a straight line - no knee to find
    return float(x[index])


def max_curvature(fit: CurveFit, n: int = 512) -> float:
    """Severity of maximum curvature of a fitted curve.

    Curvature is ``|y''| / (1 + y'^2)^{3/2}``, evaluated numerically on a dense
    grid so it works for any model in the bank without hand-derived derivatives.
    Endpoints are excluded: the finite-difference second derivative is unreliable
    there and would otherwise win by artefact.
    """
    lo, hi = fit.x_range
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    x = np.linspace(lo, hi, n)
    y = fit.predict(x)
    if not np.all(np.isfinite(y)):
        return float("nan")

    d1 = np.gradient(y, x)
    d2 = np.gradient(d1, x)
    curvature = np.abs(d2) / np.power(1.0 + d1**2, 1.5)

    margin = max(2, n // 50)
    interior = curvature[margin:-margin]
    if interior.size == 0 or not np.any(np.isfinite(interior)):
        return float("nan")
    return float(x[margin + int(np.nanargmax(interior))])


def piecewise_breakpoint(x: np.ndarray, y: np.ndarray, min_points: int = 2) -> tuple[float, float]:
    """Breakpoint of the best two-segment continuous-free linear fit.

    Grid-searches every interior split, fitting an independent line to each
    side, and keeps the split with the lowest total squared error.

    Returns ``(breakpoint, sse)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    order = np.argsort(x)
    x, y = x[order], y[order]

    if x.size < 2 * min_points + 1:
        return float("nan"), float("nan")
    if np.ptp(y) == 0:
        # A flat line has no breakpoint. Every split fits it perfectly, so the
        # grid search would return whichever one it tried first - a number with
        # no meaning, which is worse than a gap.
        return float("nan"), 0.0

    best_sse = np.inf
    best_x = float("nan")
    for split in range(min_points, x.size - min_points):
        sse = _line_sse(x[:split], y[:split]) + _line_sse(x[split:], y[split:])
        if sse < best_sse:
            best_sse, best_x = sse, float(x[split])
    return best_x, float(best_sse)


def _line_sse(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    if np.ptp(x) == 0:
        return float(np.sum((y - y.mean()) ** 2))
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.sum((y - (slope * x + intercept)) ** 2))


def find_knee(
    x: np.ndarray,
    y: np.ndarray,
    model: str = "auto",
    identifiability_threshold: float = 0.20,
) -> KneeResult:
    """Locate the knee three ways and report the consensus."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = KneeResult(identifiability_threshold=identifiability_threshold)

    result.kneedle = kneedle(x, y)

    try:
        fit = fit_curve(x, y, model)
        result.fit = fit.to_dict()
        result.max_curvature = max_curvature(fit)
        if fit.r_squared < 0.5:
            result.notes.append(
                f"curve fit is poor (R^2 = {fit.r_squared:.2f}); the curvature knee is unreliable"
            )
    except (ValueError, RuntimeError) as exc:
        result.notes.append(f"no curve fit: {exc}")

    result.piecewise, _ = piecewise_breakpoint(x, y)

    estimates = np.array([result.kneedle, result.max_curvature, result.piecewise], dtype=float)
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        result.notes.append("no method located a knee - the curve may have none")
        return result

    result.consensus = float(np.median(finite))
    span = float(np.nanmax(x) - np.nanmin(x))
    result.agreement = float((finite.max() - finite.min()) / span) if span > 0 else float("nan")

    if finite.size < 3:
        result.notes.append(f"only {finite.size} of 3 methods located a knee")
    return result


def knee_table(agg, metric: str, value_column: str = "mean", model: str = "auto"):
    """One knee per pipeline for a metric, as a tidy table."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    subset = agg[agg["metric_name"] == metric]
    for pipeline, chunk in subset.groupby("pipeline"):
        chunk = chunk.sort_values("severity")
        result = find_knee(chunk["severity"].to_numpy(), chunk[value_column].to_numpy(), model)
        rows.append(
            {
                "pipeline": pipeline,
                "metric_name": metric,
                "knee": result.consensus,
                "knee_kneedle": result.kneedle,
                "knee_curvature": result.max_curvature,
                "knee_piecewise": result.piecewise,
                "agreement": result.agreement,
                "identifiable": result.is_identifiable(),
                "curve_model": result.fit.get("model", ""),
                "curve_r2": result.fit.get("r_squared", float("nan")),
                "notes": "; ".join(result.notes),
            }
        )
    return pd.DataFrame(rows)
