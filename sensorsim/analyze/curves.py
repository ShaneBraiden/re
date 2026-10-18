"""Recovery curves - how much of the hardware deficit the algorithm buys back.

The recovery ratio from ARCHITECTURE.md section 3::

    R = (M_restored - M_degraded) / (M_clean - M_degraded)

with the three terms pinned down as:

``M_restored``
    Pipeline ``p`` at condition ``c``.
``M_degraded``
    The *baseline* pipeline at condition ``c`` - what the same hardware gives
    with a plain ISP and no restoration. This is the floor.
``M_clean``
    The baseline pipeline at the clean condition - what that ISP achieves with
    perfect hardware. This is the ceiling.

So ``R = 0`` means the pipeline did nothing the plain ISP did not already do,
and ``R = 1`` means it fully closed the gap to perfect hardware. Above 1 means
it beat the clean-hardware baseline, which happens and is interesting: it is
usually sharpening or invented detail, and the machine-task columns are where
you check which.

Normalising against the baseline's own ceiling matters because that ceiling is
not the same for every scene. A Bayer sensor cannot resolve a two-pixel-period
pattern at all, so a resolution chart caps far below a natural scene. Comparing
raw PSNR across scenes would read that ceiling as a difference in recovery.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import optimize

from sensorsim.evaluate import DIRECTION

#: Metrics whose ideal is a target value rather than a direction, and the target.
TARGET_VALUES: dict[str, float] = {"edge_acutance": 1.0}

DEFAULT_BASELINE = "classical_isp"


def to_distance_from_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite target-valued metrics as ``|value - target|``, lower better.

    ``edge_acutance`` is 1.0 when the output carries the reference's edge energy;
    0.6 (soft) and 1.4 (oversharpened) are equally wrong. Left as a raw value it
    would enter a recovery ratio as if more were better, which would score
    overshoot as recovery.
    """
    out = df.copy()
    for metric, target in TARGET_VALUES.items():
        sel = out["metric_name"] == metric
        if sel.any():
            out.loc[sel, "value"] = (out.loc[sel, "value"] - target).abs()
            out.loc[sel, "metric_name"] = f"{metric}_error"
    return out


def _finite(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def recovery_ratio(
    df: pd.DataFrame,
    baseline: str = DEFAULT_BASELINE,
    clean_severity: float = 0.0,
    min_deficit_frac: float = 0.05,
) -> pd.DataFrame:
    """Recovery ratio per image, pipeline, metric and severity.

    Conditions whose deficit is smaller than ``min_deficit_frac`` of the largest
    deficit that image and metric see across the sweep are dropped.

    This threshold is not tidying - it is the difference between a curve and
    noise. At low severity the denominator ``M_clean - M_degraded`` approaches
    zero, so R is a small difference divided by a smaller one and swings to
    hundreds. Those points are not "high recovery"; they are a ratio evaluated
    where the ratio is undefined. Scaling the threshold to each image's own
    maximum deficit keeps it meaningful across scenes whose ceilings differ by
    10 dB.
    """
    required = {"image_id", "pipeline", "metric_name", "severity", "value", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics table is missing columns: {sorted(missing)}")

    work = to_distance_from_target(df)
    work = work[work["status"] == "ok"].copy()
    work["value"] = _finite(work["value"])

    if baseline not in set(work["pipeline"]):
        raise ValueError(
            f"baseline pipeline {baseline!r} is not in the table; present: "
            f"{sorted(set(work['pipeline']))}. The recovery ratio is defined against it."
        )

    keys = ["image_id", "metric_name"]
    base = work[work["pipeline"] == baseline]

    floor = (
        base.groupby(keys + ["severity"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "m_degraded"})
    )
    ceiling = (
        base[np.isclose(base["severity"], clean_severity)]
        .groupby(keys, as_index=False)["value"]
        .mean()
        .rename(columns={"value": "m_clean"})
    )
    if ceiling.empty:
        raise ValueError(
            f"no rows at the clean condition (severity == {clean_severity}). Every sweep must "
            "include the identity condition - it is the ceiling the ratio is measured against."
        )

    merged = work.merge(floor, on=keys + ["severity"], how="left").merge(ceiling, on=keys, how="left")
    gap = merged["m_clean"] - merged["m_degraded"]
    merged["deficit"] = gap
    merged["recovery"] = (merged["value"] - merged["m_degraded"]) / gap

    largest = merged.groupby(keys)["deficit"].transform(lambda s: s.abs().max())
    measurable = gap.abs() >= np.maximum(min_deficit_frac * largest, 1e-9)
    merged.loc[~measurable, "recovery"] = np.nan
    merged["deficit_measurable"] = measurable

    return merged.dropna(subset=["recovery"])


def aggregate(
    df: pd.DataFrame,
    value_column: str = "recovery",
    group: tuple[str, ...] = ("pipeline", "metric_name", "severity"),
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean, median and a bootstrap 95% interval per group.

    Bootstrap rather than a normal interval because these samples are small
    (a handful of images) and recovery ratios are not remotely Gaussian.
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for key, chunk in df.groupby(list(group)):
        values = _finite(chunk[value_column]).dropna().to_numpy()
        if values.size == 0:
            continue
        row = dict(zip(group, key if isinstance(key, tuple) else (key,)))
        row["n"] = int(values.size)
        row["mean"] = float(values.mean())
        row["median"] = float(np.median(values))
        row["std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        if values.size > 1:
            draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
            row["ci_low"], row["ci_high"] = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
        else:
            row["ci_low"] = row["ci_high"] = row["mean"]
        records.append(row)
    return pd.DataFrame.from_records(records).sort_values(list(group)).reset_index(drop=True)


# --------------------------------------------------------------------------
# Curve models
# --------------------------------------------------------------------------

def exp_saturation(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """``c + a * (1 - exp(-b * x))`` - rises fast, then flattens."""
    return c + a * (1.0 - np.exp(-b * np.asarray(x, dtype=float)))


def logistic(x: np.ndarray, a: float, b: float, x0: float, c: float) -> np.ndarray:
    """``c + a / (1 + exp(-b * (x - x0)))`` - an S with an explicit midpoint."""
    z = np.clip(-b * (np.asarray(x, dtype=float) - x0), -500, 500)
    return c + a / (1.0 + np.exp(z))


def power_law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """``c + a * x^b`` - scale-free decay, no characteristic knee."""
    return c + a * np.power(np.clip(np.asarray(x, dtype=float), 1e-9, None), b)


MODELS: dict[str, tuple[Callable[..., np.ndarray], tuple[float, ...]]] = {
    "exp_saturation": (exp_saturation, (1.0, 3.0, 0.0)),
    "logistic": (logistic, (1.0, 8.0, 0.5, 0.0)),
    "power_law": (power_law, (1.0, 0.5, 0.0)),
}


@dataclass
class CurveFit:
    """One fitted recovery curve."""

    model: str
    params: list[float]
    r_squared: float
    aic: float
    n: int
    x_range: tuple[float, float]
    converged: bool = True
    note: str = ""

    def predict(self, x: np.ndarray) -> np.ndarray:
        fn, _ = MODELS[self.model]
        return fn(np.asarray(x, dtype=float), *self.params)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["x_range"] = list(self.x_range)
        return d


def fit_curve(x: np.ndarray, y: np.ndarray, model: str = "auto") -> CurveFit:
    """Fit a saturating curve, choosing the family by AIC when ``model='auto'``.

    AIC rather than R-squared: these models do not all have the same parameter
    count, and R-squared always prefers the bigger one.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 4:
        raise ValueError(f"need at least 4 points to fit a curve, got {x.size}")

    candidates = list(MODELS) if model == "auto" else [model]
    if model != "auto" and model not in MODELS:
        raise ValueError(f"unknown curve model {model!r}; known: {sorted(MODELS)}")

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # Relative, not `<= 0`: subtracting the mean from identical values leaves
    # float residue around 1e-32, which is not zero and would sail past an
    # exact test - and then a "fit" to that residue reports R^2 of -6.6e12.
    if np.ptp(y) <= 1e-9 * max(1.0, abs(float(y.mean()))):
        # A constant series - the baseline's own recovery ratio is exactly this.
        # Every model fits it perfectly and none of them means anything, so say
        # so rather than reporting a curve and a knee for a flat line.
        raise RuntimeError(
            "series is constant; there is no curve to fit and no knee to find "
            "(the baseline pipeline is flat at R = 0 by construction)"
        )

    best: CurveFit | None = None
    for name in candidates:
        fn, p0 = MODELS[name]
        try:
            with warnings.catch_warnings():
                # A singular Jacobian means "no covariance estimate", not "bad
                # fit". The AIC comparison below is what actually picks a model.
                warnings.simplefilter("ignore", optimize.OptimizeWarning)
                params, _ = optimize.curve_fit(fn, x, y, p0=p0, maxfev=20000)
            residual = y - fn(x, *params)
            ss_res = float(np.sum(residual**2))
            k = len(params)
            aic = x.size * np.log(max(ss_res, 1e-300) / x.size) + 2 * k
            fit = CurveFit(
                model=name,
                params=[float(p) for p in params],
                r_squared=float(1.0 - ss_res / ss_tot),
                aic=float(aic),
                n=int(x.size),
                x_range=(float(x.min()), float(x.max())),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            fit = CurveFit(
                model=name, params=[], r_squared=float("nan"), aic=float("inf"),
                n=int(x.size), x_range=(float(x.min()), float(x.max())),
                converged=False, note=str(exc),
            )
        if fit.converged and (best is None or fit.aic < best.aic):
            best = fit

    if best is None:
        raise RuntimeError(
            f"no curve model converged on {x.size} points. Curve shape is a finding too - "
            "report the raw points rather than forcing a fit."
        )
    return best


def curve_table(agg: pd.DataFrame, metric: str, model: str = "auto") -> pd.DataFrame:
    """Fit one curve per pipeline for a metric. Returns a tidy fit table."""
    rows: list[dict[str, Any]] = []
    subset = agg[agg["metric_name"] == metric]
    for pipeline, chunk in subset.groupby("pipeline"):
        chunk = chunk.sort_values("severity")
        try:
            fit = fit_curve(chunk["severity"].to_numpy(), chunk["mean"].to_numpy(), model)
            rows.append({"pipeline": pipeline, "metric_name": metric, **fit.to_dict()})
        except (ValueError, RuntimeError) as exc:
            rows.append(
                {"pipeline": pipeline, "metric_name": metric, "model": model,
                 "params": [], "r_squared": float("nan"), "aic": float("nan"),
                 "n": int(len(chunk)), "x_range": [], "converged": False, "note": str(exc)}
            )
    return pd.DataFrame(rows)


def index_to_base(
    agg: pd.DataFrame,
    base_severity: float = 0.0,
    value_column: str = "mean",
) -> pd.DataFrame:
    """Index every metric to 100 at the clean condition.

    This is what makes a fidelity series and a perceptual series shareable on
    one axis. Two y-scales on one plot would let the alignment of the scales
    invent a crossing point that is not in the data - and the crossing point is
    precisely the claim H2 makes, so it has to come from the numbers.
    """
    out = []
    for (pipeline, metric), chunk in agg.groupby(["pipeline", "metric_name"]):
        chunk = chunk.sort_values("severity").copy()
        at_base = chunk[np.isclose(chunk["severity"], base_severity)][value_column]
        if at_base.empty or at_base.iloc[0] == 0:
            continue
        chunk["indexed"] = chunk[value_column] / at_base.iloc[0] * 100.0
        # Flip lower-is-better metrics so "up" means "better" on every series.
        if DIRECTION.get(metric) is False:
            chunk["indexed"] = 200.0 - chunk["indexed"]
        out.append(chunk)
    return pd.concat(out, ignore_index=True) if out else agg.head(0).assign(indexed=[])
