"""Significance testing across pipelines, with multiple-comparison correction.

Two things this module refuses to let happen:

**Unpaired comparison of paired data.** Every pipeline sees the same images at
the same conditions, so the comparison is paired. Treating the samples as
independent throws away the pairing and inflates the variance, which usually
means declaring nothing significant.

**Uncorrected pairwise tests.** A bank of six pipelines gives fifteen pairs.
At alpha = 0.05 that is a 54% chance of at least one false positive if nothing
differs at all. Holm-Bonferroni is the default: uniformly more powerful than
plain Bonferroni, and it makes no assumption about the dependence between tests,
which matters here because the tests share data by construction.

The divergence test for H2 and H3 is :func:`divergence_test`, which asks
directly whether two metric families move in opposite directions across a
severity range - that is the claim, so it is what gets tested.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

Method = Literal["wilcoxon", "ttest"]


@dataclass
class TestResult:
    """One pairwise comparison."""

    metric_name: str
    pipeline_a: str
    pipeline_b: str
    n_pairs: int
    statistic: float
    p_value: float
    p_adjusted: float
    effect_size: float
    mean_difference: float
    method: str
    significant: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down correction.

    Returns adjusted p-values and rejection flags. Adjusted values are made
    monotone so a later test never reports a smaller adjusted p than an earlier,
    more significant one.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return [], []

    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return [float(v) for v in adjusted], [bool(v <= alpha) for v in adjusted]


def paired_deltas(
    df: pd.DataFrame,
    metric: str,
    pipeline_a: str,
    pipeline_b: str,
    value_column: str = "value",
    keys: tuple[str, ...] = ("image_id", "degradation_params_hash"),
) -> tuple[np.ndarray, np.ndarray]:
    """Values for two pipelines aligned on the conditions they share."""
    subset = df[(df["metric_name"] == metric) & (df["pipeline"].isin([pipeline_a, pipeline_b]))]
    wide = subset.pivot_table(index=list(keys), columns="pipeline", values=value_column, aggfunc="mean")
    if pipeline_a not in wide.columns or pipeline_b not in wide.columns:
        return np.array([]), np.array([])
    wide = wide[[pipeline_a, pipeline_b]].replace([np.inf, -np.inf], np.nan).dropna()
    return wide[pipeline_a].to_numpy(), wide[pipeline_b].to_numpy()


def compare_pipelines(
    df: pd.DataFrame,
    metric: str,
    baseline: str | None = None,
    method: Method = "wilcoxon",
    alpha: float = 0.05,
    value_column: str = "value",
) -> pd.DataFrame:
    """Pairwise paired tests across the bank, Holm-corrected.

    ``baseline`` restricts the comparisons to "everything against the control",
    which is usually the right family: it is n-1 tests instead of n(n-1)/2, so
    the correction costs far less power, and it matches the question the study
    actually asks.
    """
    pipelines = sorted(set(df[df["metric_name"] == metric]["pipeline"]))
    if baseline is not None and baseline not in pipelines:
        raise ValueError(f"baseline {baseline!r} not present for metric {metric!r}: {pipelines}")

    pairs = (
        [(baseline, p) for p in pipelines if p != baseline]
        if baseline
        else [(a, b) for i, a in enumerate(pipelines) for b in pipelines[i + 1 :]]
    )

    results: list[TestResult] = []
    for a, b in pairs:
        va, vb = paired_deltas(df, metric, a, b, value_column)
        if va.size < 3:
            results.append(
                TestResult(metric, a, b, int(va.size), float("nan"), 1.0, 1.0, float("nan"),
                           float("nan"), method, note="fewer than 3 paired observations")
            )
            continue

        differences = vb - va
        if np.allclose(differences, 0):
            results.append(
                TestResult(metric, a, b, int(va.size), 0.0, 1.0, 1.0, 0.0, 0.0, method,
                           note="identical values - no difference to test")
            )
            continue

        if method == "wilcoxon":
            statistic, p_value = stats.wilcoxon(va, vb)
            # Matched-pairs rank-biserial correlation.
            ranks = stats.rankdata(np.abs(differences))
            positive = float(ranks[differences > 0].sum())
            negative = float(ranks[differences < 0].sum())
            total = positive + negative
            effect = (positive - negative) / total if total > 0 else 0.0
        else:
            statistic, p_value = stats.ttest_rel(va, vb)
            sd = differences.std(ddof=1)
            effect = float(differences.mean() / sd) if sd > 0 else 0.0  # Cohen's dz

        results.append(
            TestResult(metric, a, b, int(va.size), float(statistic), float(p_value), float("nan"),
                       float(effect), float(differences.mean()), method)
        )

    adjusted, rejected = holm_bonferroni([r.p_value for r in results], alpha)
    for result, p_adj, reject in zip(results, adjusted, rejected):
        result.p_adjusted = p_adj
        result.significant = reject

    return pd.DataFrame([r.to_dict() for r in results])


def divergence_test(
    df: pd.DataFrame,
    fidelity_metric: str = "psnr",
    perceptual_metric: str = "nr_quality",
    pipeline: str | None = None,
    value_column: str = "value",
) -> dict[str, Any]:
    """Do fidelity and perceptual scores move in opposite directions?

    Spearman rather than Pearson: the claim is about direction across severity,
    not about linearity, and these metrics are on incomparable scales.

    A significantly *negative* correlation across the severity sweep is direct
    evidence for H2 - the two axes disagree, which is what "perceptual improves
    while fidelity plateaus or declines" means when stated testably.
    """
    subset = df if pipeline is None else df[df["pipeline"] == pipeline]
    keys = ["image_id", "degradation_params_hash", "pipeline"]
    wide = subset[subset["metric_name"].isin([fidelity_metric, perceptual_metric])].pivot_table(
        index=keys, columns="metric_name", values=value_column, aggfunc="mean"
    )
    if fidelity_metric not in wide.columns or perceptual_metric not in wide.columns:
        return {"n": 0, "note": f"need both {fidelity_metric} and {perceptual_metric} in the table"}

    wide = wide[[fidelity_metric, perceptual_metric]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(wide) < 4:
        return {"n": int(len(wide)), "note": "fewer than 4 paired observations"}

    rho, p_value = stats.spearmanr(wide[fidelity_metric], wide[perceptual_metric])
    return {
        "n": int(len(wide)),
        "fidelity_metric": fidelity_metric,
        "perceptual_metric": perceptual_metric,
        "pipeline": pipeline or "all",
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "diverges": bool(rho < 0 and p_value < 0.05),
        "note": (
            "negative rho with p < 0.05 supports H2: the two axes disagree across the sweep"
            if rho < 0 and p_value < 0.05
            else "no significant divergence in this slice"
        ),
    }


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval. Returns ``(point, low, high)``."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    estimates = statistic(draws, axis=1)
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(values)), float(low), float(high)


def power_note(n_pairs: int, alpha: float = 0.05, target_power: float = 0.8) -> str:
    """The smallest effect this sample size can detect, as a plain sentence.

    Worth printing next to a null result: "not significant" with six images
    usually means the study could not have seen anything smaller than a large
    effect, which is a different statement from "there is no effect".
    """
    if n_pairs < 3:
        return f"n = {n_pairs}: too few pairs for any meaningful test"
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(target_power)
    detectable = (z_alpha + z_power) / np.sqrt(n_pairs)
    return (
        f"n = {n_pairs} pairs: {target_power:.0%} power at alpha = {alpha} reaches down to "
        f"dz = {detectable:.2f}. A null result here does not rule out a smaller effect."
    )
