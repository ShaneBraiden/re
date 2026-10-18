"""Paper-ready figures, regenerated from the metrics table.

The rule from ARCHITECTURE.md section 4: no figure in ``paper/figures/`` is ever
hand-made. Every one comes from here, from the tidy table, and every one is
written next to a CSV holding the exact numbers it plots - so a reviewer who
asks "what changed?" gets a diff of numbers rather than a diff of pixels.

Chart conventions, applied uniformly:

* One y-axis, always. Two measures on incomparable scales are indexed to 100 at
  the clean condition instead - a second y-scale would let the arbitrary
  alignment of the two scales invent the crossing point that H2 is about.
* Colour follows the pipeline, fixed by name in :data:`SERIES_ORDER`. Filtering
  the bank never repaints the survivors.
* 2px lines, >=8px markers with a 2px surface ring, hairline solid gridlines,
  a legend whenever there are two or more series, and endpoint labels in ink
  rather than in the series colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown interactively

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sensorsim.analyze.curves import fit_curve  # noqa: E402
from sensorsim.analyze.knee import find_knee  # noqa: E402


# --------------------------------------------------------------------------
# Palette - validated for CVD separation and contrast on the light surface
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Theme:
    surface: str = "#fcfcfb"
    page: str = "#f9f9f7"
    ink: str = "#0b0b0b"
    ink_secondary: str = "#52514e"
    ink_muted: str = "#898781"
    grid: str = "#e1e0d9"
    axis: str = "#c3c2b7"


THEME = Theme()

#: Categorical slots in fixed order. Validated as a set: worst adjacent CVD
#: separation dE 9.1, worst adjacent normal-vision dE 19.6 on this surface.
#: Three of these sit below 3:1 contrast, so every chart also carries visible
#: direct labels and a CSV table twin - that is the relief, not an extra.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

#: Canonical pipeline order. A pipeline's colour comes from its position here,
#: not from its position in whatever subset is being plotted.
SERIES_ORDER = [
    "classical_isp",
    "nlm_denoise",
    "curve_tone",
    "rl_deconv",
    "zero_dce",
    "nafnet",
    "restormer",
    "real_esrgan",
    "learned_isp",
]

MARKER_SIZE = 5.0       # radius in points; >= 8px diameter at 100 dpi
LINE_WIDTH = 2.0
RING_WIDTH = 2.0


def color_for(name: str, extra_names: Iterable[str] = ()) -> str:
    """Stable colour for a series name."""
    if name in SERIES_ORDER:
        index = SERIES_ORDER.index(name)
    else:
        unknown = sorted(set(extra_names) - set(SERIES_ORDER))
        index = len(SERIES_ORDER) + (unknown.index(name) if name in unknown else 0)
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def apply_style() -> None:
    """Recessive chrome: the data is the only thing allowed to be loud."""
    plt.rcParams.update(
        {
            "figure.facecolor": THEME.page,
            "axes.facecolor": THEME.surface,
            "axes.edgecolor": THEME.axis,
            "axes.linewidth": 1.0,
            "axes.labelcolor": THEME.ink_secondary,
            "axes.titlecolor": THEME.ink,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": THEME.grid,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",   # solid: dashing reads as "threshold"
            "xtick.color": THEME.ink_muted,
            "ytick.color": THEME.ink_muted,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": THEME.ink,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _finish_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    subtitle: str = "",
    has_legend: bool = False,
) -> None:
    """Title, subtitle and legend stack upward from the axes, never over data."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)

    legend_band = 22 if has_legend else 0
    if subtitle:
        ax.set_title(title, loc="left", pad=20 + legend_band)
        ax.text(
            0.0, 1.02 + (0.075 if has_legend else 0.0), subtitle, transform=ax.transAxes,
            fontsize=8.5, color=THEME.ink_secondary, va="bottom", ha="left",
        )
    else:
        ax.set_title(title, loc="left", pad=8 + legend_band)


def _plot_series(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    color: str,
    yerr: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """One line: 2px stroke, ringed markers, an optional confidence wash."""
    if yerr is not None:
        ax.fill_between(x, yerr[0], yerr[1], color=color, alpha=0.10, linewidth=0, zorder=2)
    ax.plot(x, y, color=color, linewidth=LINE_WIDTH, solid_capstyle="round",
            solid_joinstyle="round", label=label, zorder=3)
    ax.plot(x, y, "o", markersize=MARKER_SIZE * 2, markerfacecolor=color,
            markeredgecolor=THEME.surface, markeredgewidth=RING_WIDTH, zorder=4)


def _end_labels(ax: plt.Axes, points: list[tuple[float, float, str]], min_separation: float = 0.055) -> None:
    """Endpoint labels in ink, dropped wholesale when the lines converge.

    Direct labels work when series separate at the right edge. When they do not,
    nudging them apart detaches each label from its line and reads as noise, so
    the legend carries identity instead - it is always present for two or more
    series precisely so this fallback exists.

    ``min_separation`` is a fraction of the y-axis span.
    """
    if not points:
        return
    lo, hi = ax.get_ylim()
    span = hi - lo
    if span <= 0:
        return
    ordered = sorted(points, key=lambda p: p[1])
    for (_, y_prev, _), (_, y_next, _) in zip(ordered, ordered[1:]):
        if abs(y_next - y_prev) < min_separation * span:
            return  # converged - let the legend do the work

    for x, y, text in points:
        ax.annotate(
            text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
            color=THEME.ink_secondary, fontsize=8, va="center", ha="left",
        )


def _focus_ylim(ax: plt.Axes, values: np.ndarray, must_include: tuple[float, ...] = (0.0, 1.0)) -> int:
    """Scale the y-axis to the bulk of the data, and say what fell outside.

    One pathological scene can send a ratio to -50 and flatten every other
    series into a single line. Clipping the view fixes that; clipping it
    *silently* would hide the pathology, so the count of excluded points is
    annotated on the chart and every value stays in the CSV twin.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0

    lo = min(float(np.percentile(values, 5)), *must_include)
    hi = max(float(np.percentile(values, 95)), *must_include)
    pad = max((hi - lo) * 0.18, 0.1)
    lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, hi)

    outside = int(np.sum((values < lo) | (values > hi)))
    if outside:
        ax.text(
            1.0, -0.155, f"{outside} point(s) outside this range - full values in the CSV twin",
            transform=ax.transAxes, fontsize=7.5, color=THEME.ink_muted, ha="right", va="top",
        )
    return outside


def _legend_above(ax: plt.Axes, n_series: int) -> None:
    """A legend in one row above the plot, where it can never cover a mark."""
    if n_series < 2:
        return  # one series needs no legend box; the title already names it
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=min(n_series, 4),
        handlelength=1.6, columnspacing=1.6, borderaxespad=0.0,
    )


def save_figure(fig: plt.Figure, path: str | Path, data: pd.DataFrame | None = None, provenance: str = "") -> Path:
    """Write a figure plus its table twin, stamped with where it came from."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if provenance:
        fig.text(0.005, 0.005, provenance, fontsize=5.5, color=THEME.ink_muted, ha="left", va="bottom")
    fig.savefig(path)
    plt.close(fig)
    if data is not None:
        data.to_csv(path.with_suffix(".csv"), index=False)
    return path


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def recovery_curves(
    agg: pd.DataFrame,
    metric: str = "psnr",
    value_column: str = "median",
    baseline: str | None = "classical_isp",
    title: str | None = None,
    provenance: str = "",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Recovery ratio against degradation severity, one line per pipeline.

    The baseline sits at R = 0 by construction, so it is drawn as a reference
    rule rather than as a series - a flat line at zero in the categorical
    palette would read as a result.
    """
    apply_style()
    data = agg[agg["metric_name"] == metric].copy()
    if data.empty:
        raise ValueError(f"no rows for metric {metric!r} in the aggregate table")

    pipelines = [p for p in data["pipeline"].unique() if p != baseline]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.axhline(0.0, color=THEME.axis, linewidth=1.0, zorder=1)
    ax.axhline(1.0, color=THEME.axis, linewidth=1.0, zorder=1)

    labels: list[tuple[float, float, str]] = []
    for name in sorted(pipelines, key=lambda p: SERIES_ORDER.index(p) if p in SERIES_ORDER else 99):
        chunk = data[data["pipeline"] == name].sort_values("severity")
        color = color_for(name, data["pipeline"].unique())
        bounds = None
        if {"ci_low", "ci_high"} <= set(chunk.columns):
            bounds = (chunk["ci_low"].to_numpy(), chunk["ci_high"].to_numpy())
        _plot_series(ax, chunk["severity"].to_numpy(), chunk[value_column].to_numpy(), name, color, bounds)
        labels.append((float(chunk["severity"].iloc[-1]), float(chunk[value_column].iloc[-1]), name))

    ax.margins(x=0.18)
    _focus_ylim(ax, data[value_column].to_numpy())
    # va bottom / top so the two reference labels separate by their own height
    # even when R = 0 and R = 1 sit close together on the axis.
    ax.text(0.004, 1.0, "fully recovered", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=THEME.ink_muted, va="bottom", ha="left")
    ax.text(0.004, 0.0, f"no gain over {baseline}", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=THEME.ink_muted, va="top", ha="left")
    _end_labels(ax, labels)
    _legend_above(ax, len(pipelines))
    _finish_axes(
        ax,
        "Degradation severity",
        f"Recovery ratio ({metric})",
        title or f"Algorithmic recovery of the hardware deficit - {metric}",
        f"R = (restored - degraded) / (clean - degraded); {value_column} across images, 95% bootstrap band",
        has_legend=len(pipelines) >= 2,
    )
    return fig, data


def knee_figure(
    agg: pd.DataFrame,
    metric: str = "psnr",
    pipeline: str | None = None,
    value_column: str = "mean",
    provenance: str = "",
) -> tuple[plt.Figure, pd.DataFrame]:
    """One pipeline's recovery curve with its fit and the located knee.

    Emphasis form: the subject is one curve and where it bends, so the other
    pipelines are context in grey rather than eight competing hues.
    """
    apply_style()
    data = agg[agg["metric_name"] == metric].copy()
    if data.empty:
        raise ValueError(f"no rows for metric {metric!r}")
    subject = pipeline or data["pipeline"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for name in data["pipeline"].unique():
        if name == subject:
            continue
        chunk = data[data["pipeline"] == name].sort_values("severity")
        ax.plot(chunk["severity"], chunk[value_column], color=THEME.grid, linewidth=1.5, zorder=2)

    chunk = data[data["pipeline"] == subject].sort_values("severity")
    x = chunk["severity"].to_numpy()
    y = chunk[value_column].to_numpy()
    color = color_for(subject, data["pipeline"].unique())
    _plot_series(ax, x, y, subject, color)

    knee = find_knee(x, y)
    rows = chunk.copy()
    rows["knee_consensus"] = knee.consensus
    rows["knee_identifiable"] = knee.is_identifiable()

    if np.isfinite(knee.consensus):
        try:
            fit = fit_curve(x, y)
            dense = np.linspace(x.min(), x.max(), 200)
            ax.plot(dense, fit.predict(dense), color=color, linewidth=1.0, alpha=0.55, zorder=3)
            rows["curve_model"] = fit.model
        except (ValueError, RuntimeError):
            pass

    ax.margins(x=0.12)
    # Scale to the *subject*, not to the context lines. A pathological arm in the
    # grey background would otherwise squash the one curve the figure is about.
    _focus_ylim(ax, y, must_include=(0.0, 1.0))

    if np.isfinite(knee.consensus):
        knee_y = float(np.interp(knee.consensus, x, y))
        ax.axvline(knee.consensus, color=THEME.axis, linewidth=1.0, zorder=1)
        ax.plot([knee.consensus], [knee_y], "o", markersize=MARKER_SIZE * 2.4,
                markerfacecolor=THEME.surface, markeredgecolor=color, markeredgewidth=2.0, zorder=5)
        # Annotate below the marker: above it is where the subtitle band lives.
        lo, hi = ax.get_ylim()
        below = knee_y - (hi - lo) * 0.10 > lo
        ax.annotate(
            f"knee {knee.consensus:.2f}",
            xy=(knee.consensus, knee_y), xytext=(9, -16 if below else 14),
            textcoords="offset points", color=THEME.ink_secondary, fontsize=8,
        )

    # The three estimates go in a footnote rather than the subtitle: they are
    # what a reader checks *after* deciding whether to believe the knee, and a
    # subtitle long enough to hold them runs off the figure.
    ax.text(
        0.0, -0.20,
        f"kneedle {knee.kneedle:.2f} · curvature {knee.max_curvature:.2f} · "
        f"piecewise {knee.piecewise:.2f} · spread {knee.agreement:.0%} of range",
        transform=ax.transAxes, fontsize=7.5, color=THEME.ink_muted, va="top", ha="left",
    )

    verdict = "identifiable" if knee.is_identifiable() else "NOT identifiable - the three methods disagree"
    _finish_axes(
        ax, "Degradation severity", f"Recovery ratio ({metric})",
        f"Saturation point - {subject}",
        f"knee at severity {knee.consensus:.2f}, {verdict}",
    )
    return fig, rows


def divergence_figure(
    indexed: pd.DataFrame,
    pipeline: str,
    metrics: tuple[str, ...] = ("psnr", "nr_quality", "sift_inlier_ratio"),
    provenance: str = "",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Fidelity, perceptual and machine-task on one axis, indexed to 100.

    Indexing rather than a second y-scale is the whole point. H2 and H3 are
    claims about where these lines cross; a dual-axis chart would place that
    crossing wherever the two scales happened to be aligned, so the crossing has
    to be produced by the numbers instead.

    Every series is oriented so up means better, including the
    lower-is-better ones - otherwise "diverging" would be a statement about
    metric conventions rather than about the images.
    """
    apply_style()
    data = indexed[(indexed["pipeline"] == pipeline) & (indexed["metric_name"].isin(metrics))].copy()
    if data.empty:
        raise ValueError(f"no indexed rows for {pipeline!r} and metrics {metrics}")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(100.0, color=THEME.axis, linewidth=1.0, zorder=1)

    labels: list[tuple[float, float, str]] = []
    drawn = 0
    for i, metric in enumerate(metrics):
        chunk = data[data["metric_name"] == metric].sort_values("severity")
        if chunk.empty:
            continue
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        _plot_series(ax, chunk["severity"].to_numpy(), chunk["indexed"].to_numpy(), metric, color)
        labels.append((float(chunk["severity"].iloc[-1]), float(chunk["indexed"].iloc[-1]), metric))
        drawn += 1

    ax.margins(x=0.20)
    _focus_ylim(ax, data["indexed"].to_numpy(), must_include=(100.0,))
    # Right-anchored: at severity 0 every series starts at 100, so a left-hand
    # note would sit directly on the marks it is annotating.
    ax.text(0.995, 100.0, "clean-hardware baseline", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=THEME.ink_muted, va="bottom", ha="right")
    _end_labels(ax, labels)
    _legend_above(ax, drawn)
    _finish_axes(
        ax, "Degradation severity", "Indexed to clean condition (= 100, up is better)",
        f"Fidelity, perception and machine vision diverge - {pipeline}",
        "one axis: every series indexed to its own clean-condition value",
        has_legend=drawn >= 2,
    )
    return fig, data


def photon_transfer_figure(
    synthetic: tuple[np.ndarray, np.ndarray],
    measured: tuple[np.ndarray, np.ndarray] | None = None,
    provenance: str = "",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Noise variance against intensity - the calibration gate, as a figure.

    This is the plot that separates a study from a simulation, so it belongs in
    the method section whether or not it is flattering.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    frames = []
    s_mean, s_var = synthetic
    _plot_series(ax, s_mean, s_var, "synthetic", SERIES_COLORS[0])
    frames.append(pd.DataFrame({"source": "synthetic", "mean": s_mean, "variance": s_var}))

    if measured is not None:
        m_mean, m_var = measured
        _plot_series(ax, m_mean, m_var, "measured", SERIES_COLORS[1])
        frames.append(pd.DataFrame({"source": "measured", "mean": m_mean, "variance": m_var}))

    # Plain tick labels: matplotlib's scientific-notation offset ("1e-3") is
    # parked in the top-left corner, where it lands on the subtitle.
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    _legend_above(ax, 2 if measured is not None else 1)
    _finish_axes(
        ax, "Mean signal (normalised)", "Temporal variance",
        "Photon transfer curve",
        "variance rising linearly with signal is the Poisson signature",
        has_legend=measured is not None,
    )
    return fig, pd.concat(frames, ignore_index=True)


def crop_grid(
    crops: dict[str, dict[str, np.ndarray]],
    crop_name: str,
    provenance: str = "",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Qualitative crops at fixed coordinates: rows are conditions, columns pipelines.

    ``crops`` is ``{condition_label: {pipeline: image_crop}}``. Fixed
    coordinates, chosen once per scene, so nobody picks the flattering crop
    after seeing the results.
    """
    apply_style()
    conditions = list(crops)
    pipelines = sorted({p for row in crops.values() for p in row},
                       key=lambda p: SERIES_ORDER.index(p) if p in SERIES_ORDER else 99)
    if not conditions or not pipelines:
        raise ValueError("crop grid needs at least one condition and one pipeline")

    fig, axes = plt.subplots(
        len(conditions), len(pipelines),
        figsize=(1.6 * len(pipelines), 1.7 * len(conditions)),
        squeeze=False,
    )
    for r, condition in enumerate(conditions):
        for c, pipeline in enumerate(pipelines):
            ax = axes[r][c]
            image = crops[condition].get(pipeline)
            if image is not None:
                ax.imshow(np.asarray(image))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(THEME.grid)
            if r == 0:
                ax.set_title(pipeline, fontsize=8, color=THEME.ink_secondary, pad=4)
            if c == 0:
                ax.set_ylabel(condition, fontsize=8, color=THEME.ink_secondary, rotation=0,
                              ha="right", va="center", labelpad=8)

    fig.suptitle(f"Qualitative comparison - {crop_name}", x=0.02, ha="left", fontsize=11, color=THEME.ink)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    index = pd.DataFrame(
        [{"condition": c, "pipeline": p, "crop": crop_name} for c in conditions for p in pipelines]
    )
    return fig, index


def generate_all(
    metrics: pd.DataFrame,
    out_dir: str | Path = "paper/figures",
    metric: str = "psnr",
    baseline: str = "classical_isp",
    provenance: str = "",
) -> list[Path]:
    """Regenerate the paper's figure set from a metrics table.

    Every figure the analysis chapter needs, in one call, so ``make reproduce``
    has a single entry point and no figure can drift from the table it claims
    to show.
    """
    from sensorsim.analyze.curves import aggregate, index_to_base, recovery_ratio

    out_dir = Path(out_dir)
    written: list[Path] = []

    recovery = recovery_ratio(metrics, baseline=baseline)
    agg_recovery = aggregate(recovery, value_column="recovery")

    fig, data = recovery_curves(agg_recovery, metric=metric, baseline=baseline)
    written.append(save_figure(fig, out_dir / f"recovery_{metric}.png", data, provenance))

    # A knee figure per arm, not just the first one: the paper cites them by
    # name, and a figure that exists only when a bank happens to be ordered a
    # certain way is a broken \includegraphics waiting to happen.
    for name in sorted(p for p in agg_recovery["pipeline"].unique() if p != baseline):
        try:
            fig, data = knee_figure(agg_recovery, metric=metric, pipeline=name)
        except ValueError:
            continue
        written.append(save_figure(fig, out_dir / f"knee_{metric}_{name}.png", data, provenance))

    agg_raw = aggregate(metrics[metrics["status"] == "ok"], value_column="value")
    indexed = index_to_base(agg_raw)
    for pipeline in agg_raw["pipeline"].unique():
        try:
            fig, data = divergence_figure(indexed, pipeline=pipeline)
        except ValueError:
            continue
        written.append(save_figure(fig, out_dir / f"divergence_{pipeline}.png", data, provenance))

    return written
