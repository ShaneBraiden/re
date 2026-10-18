"""Tables, figures, macros and generated prose for the NAG section of the paper.

    sensorsim paper-nag

Reads ``experiments/enhance_study/metrics.csv`` and writes the following:

``paper/overleaf/generated/proposed_numbers.tex``
    One ``\\newcommand`` per number quoted in prose.
``paper/overleaf/generated/proposed_results.tex``
    The noise-map validation table, the main table, the ablation table, the
    figures, and the findings paragraph.
``paper/overleaf/generated/nag_abstract.tex``, ``nag_conclusion.tex``
    The sentences inserted into the abstract and the conclusion.
``paper/figures/enhance_study/*.png`` (+ CSV twins), copied to
``paper/overleaf/figures/``
    The figures themselves.

Every sentence that states a direction ("raised", "lowered", "significant") is
built from the number it describes. A result that goes against NAG is
reported in the same words as one that goes for it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sensorsim.analyze.stats import holm_bonferroni, paired_deltas

GENERATED = Path("paper/overleaf/generated")
OVERLEAF_FIGS = Path("paper/overleaf/figures")
FIG_DIR = Path("paper/figures/enhance_study")

MAIN_ARMS = ["classical_isp", "nlm_denoise", "curve_tone", "rl_deconv",
             "zero_dce", "real_esrgan", "nc_dce", "g_esrgan", "nag"]
ABLATION_ARMS = ["nag", "nag_nodenoise", "nag_nogate", "nag_nobp",
                 "nc_dce", "nc_dce_nosigma", "nc_dce_nonoise", "zero_dce_ft"]
LABELS = {
    "classical_isp": r"\ISP", "nlm_denoise": r"\NLM", "curve_tone": r"\TONE", "rl_deconv": r"\RL",
    "zero_dce": r"\ZDCE", "real_esrgan": r"\RESR", "nc_dce": r"\NCDCE", "g_esrgan": r"\GESR",
    "nag": r"\NAGarm", "nag_nodenoise": r"\NAGarm\ $-$ VP-denoise", "nag_nogate": r"\NAGarm\ $-$ gate",
    "nag_nobp": r"\NAGarm\ $-$ back-proj.", "nc_dce_nosigma": r"\NCDCE\ $-$ $\sigma$ input",
    "nc_dce_nonoise": r"\NCDCE\ $-$ $\mathcal{L}_{\mathrm{noise}}$",
    "zero_dce_ft": r"\ZDCE\ fine-tuned",
}
PLAIN = {
    "classical_isp": "control ISP", "nlm_denoise": "non-local means", "curve_tone": "tone curve",
    "rl_deconv": "RL deconvolution", "zero_dce": "Zero-DCE", "real_esrgan": "Real-ESRGAN",
    "nc_dce": "NC-DCE", "g_esrgan": "G-ESRGAN", "nag": "NAG", "zero_dce_ft": "Zero-DCE (fine-tuned)",
}
TABLE_METRICS = [("psnr", "PSNR", 2), ("ssim", "SSIM", 3), ("delta_e", r"$\Delta E$", 2),
                 ("noise_sigma", r"$\hat\sigma$", 2), ("nr_quality", r"\NRQ", 2), ("sift_inlier_ratio", "SIFT", 3)]

#: Planned paired comparisons: (arm, against). Holm-corrected as one family.
PLANNED = [
    ("g_esrgan", "real_esrgan"),
    ("nc_dce", "zero_dce"),
    ("nc_dce", "zero_dce_ft"),
    ("nag", "classical_isp"),
    ("nag", "real_esrgan"),
    ("nag", "zero_dce"),
    ("nag_nodenoise", "nag"),
    ("nag_nogate", "nag"),
    ("nag_nobp", "nag"),
    ("nc_dce_nosigma", "nc_dce"),
    ("nc_dce_nonoise", "nc_dce"),
]


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

def arm_means(ok: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    sub = ok[ok["pipeline"].isin(arms)]
    table = sub.pivot_table(index="pipeline", columns="metric_name", values="value", aggfunc="mean")
    return table.reindex([a for a in arms if a in table.index])


def planned_tests(ok: pd.DataFrame, metrics: tuple[str, ...] = ("psnr", "ssim", "noise_sigma", "sift_inlier_ratio")) -> pd.DataFrame:
    """Paired Wilcoxon tests for the planned comparisons, Holm-corrected per metric."""
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        family: list[dict[str, Any]] = []
        for arm, ref in PLANNED:
            va, vb = paired_deltas(ok, metric, ref, arm)
            row = {"metric": metric, "arm": arm, "against": ref, "n": int(va.size),
                   "mean_delta": float(np.mean(vb - va)) if va.size else float("nan"),
                   "median_delta": float(np.median(vb - va)) if va.size else float("nan"),
                   "wins": int(np.sum(vb > va)), "p": 1.0}
            if va.size >= 3 and not np.allclose(vb, va):
                row["p"] = float(stats.wilcoxon(va, vb).pvalue)
            family.append(row)
        # A comparison with an arm that never ran was not performed, so it is
        # not part of the family: counting it as p = 1 would inflate every
        # other test's correction.
        tested = [r for r in family if r["n"] >= 3]
        adjusted, rejected = holm_bonferroni([r["p"] for r in tested])
        for r, p_adj, rej in zip(tested, adjusted, rejected):
            r["p_holm"], r["significant"] = float(p_adj), bool(rej)
        for r in family:
            if r["n"] < 3:
                r["p"], r["p_holm"], r["significant"] = float("nan"), float("nan"), False
        rows.extend(family)
    return pd.DataFrame(rows)


def noise_validation() -> pd.DataFrame:
    """Predicted vs measured noise std on flat fields across the sweep's range."""
    from sensorsim.data.synthetic import Scene, scene_to_raw
    from sensorsim.degrader.model import DegradationParams, degrade
    from sensorsim.restore.base import RestoreContext
    from sensorsim.restore.learned.nag import front_end

    rows = []
    conditions = [(1.0, 0.0, 1, 14), (0.2, 3.0, 1, 12), (0.03, 10.0, 1, 10), (0.1, 4.0, 2, 12)]
    for alpha, sr, b, bits in conditions:
        for level in (0.05, 0.2, 0.5):
            raw = scene_to_raw(Scene("flat", "natural", np.full((256, 256, 3), level, np.float32), {}))
            params = DegradationParams(alpha=alpha, sigma_r_dn=sr, r_psf=0.0, binning=b, bit_depth=bits)
            degraded, rec = degrade(raw, params, seed=11)
            srgb, sigma = front_end(degraded, RestoreContext("flat", raw.shape, rec.params, rec.sensor))
            c = (slice(16, -16), slice(16, -16))
            measured, predicted = srgb[c].std(axis=(0, 1)), sigma[c].mean(axis=(0, 1))
            for ch, name in enumerate("RGB"):
                rows.append({"alpha": alpha, "sigma_r_dn": sr, "binning": b, "bit_depth": bits, "level": level,
                             "channel": name, "measured": float(measured[ch]), "predicted": float(predicted[ch]),
                             "ratio": float(measured[ch] / predicted[ch])})
    return pd.DataFrame(rows)


def _signed(v: float, digits: int = 2) -> str:
    if not np.isfinite(v):
        return "n/a"
    return rf"\ensuremath{{{'+' if v >= 0 else '-'}{abs(v):.{digits}f}}}"


def _p(p: float) -> str:
    """A p-value for a table cell."""
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _pexpr(p: float) -> str:
    """A p-value for running text, relation included: ``$p < 0.001$`` or ``$p = 0.042$``."""
    if not np.isfinite(p):
        return "$p$ n/a"
    return "$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$"


def _verb(delta: float, up: str = "raised", down: str = "lowered") -> str:
    return up if delta >= 0 else down


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

NAG_COLOURS = {
    # Pilot slots, so an arm keeps its colour across every figure in the paper.
    "classical_isp": 0, "nlm_denoise": 1, "rl_deconv": 3, "zero_dce": 4, "real_esrgan": 7,
    # Slots 5 and 6 belong to nafnet and restormer, which never run in this paper.
    "g_esrgan": 5, "nag": 6,
}


def curves_figure(ok: pd.DataFrame, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sensorsim.analyze.figures import SERIES_COLORS, THEME, _plot_series, apply_style, save_figure

    apply_style()
    arms = [a for a in NAG_COLOURS if a in set(ok["pipeline"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1))
    rows = []
    for ax, (metric, label) in zip(axes, (("psnr", "PSNR (dB)"), ("sift_inlier_ratio", "SIFT inlier ratio"))):
        for arm in arms:
            series = ok[(ok["pipeline"] == arm) & (ok["metric_name"] == metric)].groupby("severity")["value"].mean()
            x, y = series.index.to_numpy(float), series.to_numpy(float)
            _plot_series(ax, x, y, PLAIN.get(arm, arm), SERIES_COLORS[NAG_COLOURS[arm]])
            rows += [{"metric": metric, "pipeline": arm, "severity": a, "mean": b} for a, b in zip(x, y)]
        ax.set_xlabel("Degradation severity s")
        ax.set_ylabel(label)
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.03, 1.03)
    # One legend for both panels, above them: the series are the same arms.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(arms), 7), frameon=False,
               fontsize=7.5, handlelength=1.6, columnspacing=1.2, labelcolor=THEME.ink_secondary)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save_figure(fig, out, pd.DataFrame(rows), "sensorsim paper-nag")


def crops_figure(study_dir: Path, metrics: pd.DataFrame, out: Path) -> Path:
    """Matched crops at the most severe condition: reference and four arms."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sensorsim.analyze.figures import THEME, apply_style

    apply_style()
    worst = metrics.loc[metrics["severity"].idxmax(), "degradation_params_hash"]
    corpus_index = json.loads(Path("data/synthetic/index.json").read_text(encoding="utf-8"))
    crops = {e["image_id"]: e["crops"] for e in corpus_index}
    picks = [("txt_00", "paragraph"), ("nat_00", "distant_windows")]
    arms = [("reference", "Reference"), ("classical_isp", "Control ISP"), ("zero_dce", "Zero-DCE"),
            ("real_esrgan", "Real-ESRGAN"), ("nag", "NAG (ours)")]
    fig, axes = plt.subplots(len(picks), len(arms), figsize=(7.16, 3.1))
    for r, (image_id, crop_name) in enumerate(picks):
        x, y, w, h = crops[image_id][crop_name]
        w, h = min(w, 140), min(h, 100)
        for c, (arm, title) in enumerate(arms):
            name = "reference.jpg" if arm == "reference" else f"{worst}__{arm}.jpg"
            img = cv2.imread(str(study_dir / "previews" / image_id / name))
            ax = axes[r, c]
            ax.axis("off")
            if img is None:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", fontsize=8, color=THEME.ink_muted)
                continue
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[y:y + h, x:x + w], interpolation="nearest")
            if r == 0:
                ax.set_title(title, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=THEME.surface)
    plt.close(fig)
    return out


def training_figure(training_dir: Path, out: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sensorsim.analyze.figures import SERIES_COLORS, THEME, apply_style, save_figure

    logs = {p.stem.replace("_loss", ""): pd.read_csv(p) for p in sorted(training_dir.glob("*_loss.csv"))}
    logs = {k: v for k, v in logs.items() if "total" in v.columns}
    if not logs:
        return None
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    rows = []
    for i, (name, df) in enumerate(logs.items()):
        colour = SERIES_COLORS[i]
        for ax, col in zip(axes, ("total", "noise_metric")):
            ax.plot(df["epoch"], df[col], color=colour, linewidth=2.0, label=name, solid_capstyle="round")
            if len(df) < 10:  # a run that stopped early: mark where it stopped
                ax.plot(df["epoch"].iloc[-1:], df[col].iloc[-1:], "X", color=colour, markersize=8,
                        markeredgecolor=THEME.surface)
        rows += [{"variant": name, **r} for r in df.to_dict("records")]
    for ax, label in zip(axes, ("Training loss (log)", r"$\mathcal{L}_{\mathrm{noise}}$, measured (log)")):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(True, axis="y", which="major", zorder=0)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(logs), frameon=False, fontsize=7.5,
               labelcolor=THEME.ink_secondary)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save_figure(fig, out, pd.DataFrame(rows), "sensorsim train nc-dce")


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------

def _table(means: pd.DataFrame, caption: str, label: str, extra: dict[str, list[str]] | None = None) -> str:
    cols = [(m, h, d) for m, h, d in TABLE_METRICS if m in means.columns]
    extra = extra or {}
    head = " & ".join(["Arm"] + [h for _, h, _ in cols] + list(extra))
    lines = []
    for i, arm in enumerate(means.index):
        cells = [LABELS.get(arm, arm)]
        for m, _, d in cols:
            v = means.loc[arm, m]
            cells.append("--" if not np.isfinite(v) else f"{v:.{d}f}")
        cells += [extra[k][i] for k in extra]
        lines.append(" & ".join(cells) + r" \\")
    spec = "l" + "r" * (len(cols) + len(extra))
    body = "\n    ".join(lines)
    return rf"""\begin{{table*}}[!t]
  \centering
  \caption{{{caption}}}
  \label{{{label}}}
  \small
  \begin{{tabular}}{{@{{}}{spec}@{{}}}}
    \toprule
    {head} \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""


def build_paper_assets(
    metrics_path: str | Path = "experiments/enhance_study/metrics.csv",
    training_dir: str | Path = "experiments/enhance_study/training",
    generated: str | Path = GENERATED,
    fig_dir: str | Path = FIG_DIR,
    overleaf_figs: str | Path = OVERLEAF_FIGS,
) -> dict[str, Any]:
    metrics_path, training_dir = Path(metrics_path), Path(training_dir)
    generated, fig_dir, overleaf_figs = Path(generated), Path(fig_dir), Path(overleaf_figs)
    generated.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_path)
    ok = df[df["status"] == "ok"]
    unavailable = sorted(set(df.loc[df["status"] != "ok", "pipeline"]))
    written: list[str] = []

    # -- numbers ---------------------------------------------------------
    main = arm_means(ok, MAIN_ARMS)
    abl = arm_means(ok, ABLATION_ARMS + ["zero_dce"])
    tests = planned_tests(ok)
    noise = noise_validation()
    fig_dir.mkdir(parents=True, exist_ok=True)
    tests.to_csv(fig_dir / "planned_tests.csv", index=False)
    noise.to_csv(fig_dir / "noise_validation.csv", index=False)
    main.to_csv(fig_dir / "main_means.csv")
    abl.to_csv(fig_dir / "ablation_means.csv")

    def t(arm: str, ref: str, metric: str = "psnr") -> dict[str, Any]:
        row = tests[(tests["arm"] == arm) & (tests["against"] == ref) & (tests["metric"] == metric)]
        return row.iloc[0].to_dict() if len(row) else {"mean_delta": float("nan"), "p_holm": float("nan"),
                                                        "significant": False, "wins": 0, "n": 0}

    training = {}
    for p in sorted(Path("models").glob("*.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        training[meta["name"]] = meta
    main_train = training.get("nc_dce", {})
    cfg = main_train.get("config", {})

    manifest_path = metrics_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    worst_err = float(np.abs(noise["ratio"] - 1.0).max())
    observations = int(len(ok))
    gesr, ncd, nag_isp = t("g_esrgan", "real_esrgan"), t("nc_dce", "zero_dce"), t("nag", "classical_isp")
    ncd_ft, ncd_ft_noise = t("nc_dce", "zero_dce_ft"), t("nc_dce", "zero_dce_ft", "noise_sigma")
    diverged = (training.get("nc_dce_nonoise") or {}).get("diverged")
    nag_resr, nag_zdce = t("nag", "real_esrgan"), t("nag", "zero_dce")
    gesr_sift, ncd_noise = t("g_esrgan", "real_esrgan", "sift_inlier_ratio"), t("nc_dce", "zero_dce", "noise_sigma")
    nag_sift = t("nag", "classical_isp", "sift_inlier_ratio")
    # G-ESRGAN against the control is not a planned test; it is the main table's
    # Delta-PSNR entry, quoted to say where NAG's PSNR difference comes from.
    va, vb = paired_deltas(ok, "psnr", "classical_isp", "g_esrgan")
    gesr_isp = float(np.mean(vb - va)) if va.size else float("nan")
    nag_nodenoise = t("nag_nodenoise", "nag")

    macros = {
        "nagNoiseCases": str(len(noise)),
        "nagNoiseMaxErr": f"{worst_err * 100:.1f}\\%",
        "nagTrainSamples": str(cfg.get("samples", "n/a")),
        "nagTrainEpochs": str(cfg.get("epochs", "n/a")),
        "nagTrainSeconds": f"{main_train.get('seconds', float('nan')):.0f}",
        "nagObservations": f"{observations:,}",
        "gesrDpsnrVsResr": _signed(gesr["mean_delta"]),
        "gesrPvsResr": _pexpr(gesr["p_holm"]),
        "gesrDsiftVsResr": _signed(gesr_sift["mean_delta"], 3),
        "ncdceDpsnrVsZdce": _signed(ncd["mean_delta"]),
        "ncdcePvsZdce": _pexpr(ncd["p_holm"]),
        "ncdceDnoiseVsZdce": _signed(ncd_noise["mean_delta"], 2),
        "ncdceDpsnrVsZdceFt": _signed(ncd_ft["mean_delta"]),
        "ncdcePvsZdceFt": _pexpr(ncd_ft["p_holm"]),
        "ncdceDnoiseVsZdceFt": _signed(ncd_ft_noise["mean_delta"], 2),
        "ncdcePnoiseVsZdceFt": _pexpr(ncd_ft_noise["p_holm"]),
        "gesrDpsnrVsIsp": _signed(gesr_isp),
        "nagDpsnrVsIsp": _signed(nag_isp["mean_delta"]),
        "nagPvsIsp": _pexpr(nag_isp["p_holm"]),
        "nagDpsnrVsResr": _signed(nag_resr["mean_delta"]),
        "nagDpsnrVsZdce": _signed(nag_zdce["mean_delta"]),
        "nagDsiftVsIsp": _signed(nag_sift["mean_delta"], 3),
        # Provenance of the study run, for the methodology text.
        "nagConfigHash": r"\texttt{" + str(manifest.get("config_hash", "n/a")) + "}",
        "nagTorch": str(manifest.get("packages", {}).get("torch", "n/a")).split("+")[0],
        "nagCuda": str(manifest.get("hardware_info", {}).get("cuda", "n/a")),
        "nagGpu": str(manifest.get("hardware_info", {}).get("gpu", "n/a")).replace("NVIDIA ", ""),
    }
    numbers = ["% Generated by sensorsim paper-nag from " + metrics_path.as_posix() + ". Do not edit."]
    numbers += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()]
    (generated / "proposed_numbers.tex").write_text("\n".join(numbers) + "\n", encoding="utf-8")
    written.append((generated / "proposed_numbers.tex").as_posix())

    # -- figures ---------------------------------------------------------
    figs = {
        "nag_curves.png": curves_figure(ok, fig_dir / "nag_curves.png"),
        "nag_crops.png": crops_figure(metrics_path.parent, df, fig_dir / "nag_crops.png"),
    }
    train_fig = training_figure(training_dir, fig_dir / "nag_training.png")
    if train_fig:
        figs["nag_training.png"] = train_fig
    overleaf_figs.mkdir(parents=True, exist_ok=True)
    for name, path in figs.items():
        shutil.copy2(path, overleaf_figs / name)
        written.append((overleaf_figs / name).as_posix())

    # -- tables ----------------------------------------------------------
    p_col = []
    d_col = []
    for arm in main.index:
        if arm == "classical_isp":
            p_col.append("--")
            d_col.append("--")
            continue
        va, vb = paired_deltas(ok, "psnr", "classical_isp", arm)
        d_col.append(f"{np.mean(vb - va):+.2f}" if va.size else "--")
        p_col.append(_p(float(stats.wilcoxon(va, vb).pvalue)) if va.size >= 3 and not np.allclose(va, vb) else "--")
    main_tab = _table(
        main,
        "NAG study: means over three scenes and seven severities (21 frames per arm). "
        r"$\Delta$PSNR and $p$ (Wilcoxon signed-rank, uncorrected) are paired against the control \ISP. "
        r"$\hat\sigma$ is the luma noise estimate (lower is better); SIFT is the inlier ratio.",
        "tab:nag", {r"$\Delta$PSNR": d_col, "$p$": p_col},
    )

    abl_delta = []
    for arm in abl.index:
        ref = "nag" if arm.startswith("nag") else ("nc_dce" if arm.startswith("nc_dce") else None)
        if ref is None or arm == ref:
            abl_delta.append("--")
            continue
        va, vb = paired_deltas(ok, "psnr", ref, arm)
        abl_delta.append(f"{np.mean(vb - va):+.2f}" if va.size else "--")
    abl_tab = _table(
        abl,
        r"Ablations. Each row removes one component; $\Delta$PSNR is paired against the full arm it ablates "
        r"(\NAGarm\ or \NCDCE). \ZDCE\ is the released checkpoint of the base method; \ZDCE\ fine-tuned is "
        r"its published recipe retrained on the same data and schedule as \NCDCE, the control for the "
        r"training data." + (r" A missing row is an arm that could not run (see text)." if unavailable else ""),
        "tab:nagabl", {r"$\Delta$PSNR": abl_delta},
    )

    noise_rows = []
    for (alpha, sr, b, bits), g in noise.groupby(["alpha", "sigma_r_dn", "binning", "bit_depth"], sort=False):
        noise_rows.append(rf"{alpha:g} & {sr:g} & {b} & {bits} & {g['ratio'].mean():.3f} & "
                          rf"{np.abs(g['ratio'] - 1).max() * 100:.1f}\% \\")
    worst = noise.loc[(noise["ratio"] - 1).abs().idxmax()]
    gain_step = 1.0 / (2.0 ** worst["bit_depth"] - 1.0)
    read_norm = worst["sigma_r_dn"] / (16383.0 - 512.0)
    worst_note = (
        rf" The worst case ($\alpha = {worst['alpha']:g}$, {int(worst['bit_depth'])}-bit, level {worst['level']:g}, "
        rf"{'under' if worst['ratio'] > 1 else 'over'}-predicted by {abs(worst['ratio'] - 1) * 100:.1f}\%) is where the "
        r"ADC step exceeds the read noise, so quantisation is no longer noise-like and the $q^2/12$ term "
        r"underestimates it." if gain_step > read_norm else ""
    )
    noise_tab = rf"""\begin{{table}}[!t]
  \centering
  \caption{{Propagated noise map against measurement on flat fields at three levels and three channels
  per condition: mean ratio of measured to predicted std, and worst absolute error.{worst_note}}}
  \label{{tab:nagnoise}}
  \footnotesize
  \begin{{tabular}}{{@{{}}rrrrrr@{{}}}}
    \toprule
    $\alpha$ & $\sigma_r$ & $b$ & bits & meas./pred. & worst \\
    \midrule
    {chr(10).join('    ' + r for r in noise_rows).strip()}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""

    # -- generated prose -------------------------------------------------
    def sig(row: dict[str, Any]) -> str:
        if row.get("n", 0) < 3:
            return "(too few pairs to test)"
        word = "significant" if row["significant"] else "not significant"
        return f"(Wilcoxon, Holm-corrected {_pexpr(row['p_holm'])}, {word}; {row['wins']} of {row['n']} frames higher)"

    findings = [
        rf"Relative to the generator it wraps, \GESR\ {_verb(gesr['mean_delta'])} PSNR by "
        rf"{abs(gesr['mean_delta']):.2f}~dB {sig(gesr)} and {_verb(gesr_sift['mean_delta'], 'raised', 'lowered')} "
        rf"the SIFT inlier ratio by {abs(gesr_sift['mean_delta']):.3f}.",
        rf"Relative to the published curve estimator, \NCDCE\ {_verb(ncd['mean_delta'])} PSNR by "
        rf"{abs(ncd['mean_delta']):.2f}~dB {sig(ncd)} and {_verb(ncd_noise['mean_delta'])} the noise estimate "
        rf"$\hat\sigma$ by {abs(ncd_noise['mean_delta']):.2f} {sig(ncd_noise)}. Against the published recipe "
        rf"fine-tuned on the same data and schedule, the difference was {_signed(ncd_ft['mean_delta'])}~dB "
        rf"{sig(ncd_ft)} in PSNR and {_signed(ncd_ft_noise['mean_delta'])} in $\hat\sigma$ {sig(ncd_ft_noise)}; "
        r"this comparison separates the method's contribution from that of the fine-tuning data.",
        rf"Against the control ISP, the full \NAGarm\ arm {_verb(nag_isp['mean_delta'])} PSNR by "
        rf"{abs(nag_isp['mean_delta']):.2f}~dB {sig(nag_isp)}; against \RESR\ and \ZDCE\ the paired "
        rf"differences were {_signed(nag_resr['mean_delta'])}~dB and {_signed(nag_zdce['mean_delta'])}~dB.",
    ]
    for arm, ref, what in (("nag_nodenoise", "nag", "VP-denoise"), ("nag_nogate", "nag", "the gate"),
                           ("nag_nobp", "nag", "back-projection"), ("nc_dce_nosigma", "nc_dce", r"the $\sigma$ input"),
                           ("nc_dce_nonoise", "nc_dce", r"$\mathcal{L}_{\mathrm{noise}}$")):
        row = t(arm, ref)
        if row["n"]:
            findings.append(rf"Removing {what} changed PSNR by {_signed(row['mean_delta'])}~dB {sig(row)}.")
    if diverged:
        findings.append(
            r"Without $\mathcal{L}_{\mathrm{noise}}$, training diverged (non-finite loss at epoch "
            rf"{diverged['epoch']}). With a one-sided exposure term, nothing else limits how far the curves "
            r"brighten, so the noise term is also what keeps NC-DCE's training stable. That variant is recorded "
            r"as a gap rather than evaluated.")
    if unavailable:
        findings.append("Arms that could not run and are recorded as gaps: " + ", ".join(
            r"\texttt{" + a.replace("_", r"\_") + "}" for a in unavailable) + ".")

    results = [
        "% Generated by sensorsim paper-nag. Do not edit; rerun the command instead.",
        noise_tab,
        main_tab,
        abl_tab,
        r"""\begin{figure*}[!t]
  \centering
  \includegraphics[width=\textwidth]{nag_curves.png}
  \caption{Fidelity (left) and machine-vision correspondence (right) across the sweep for the control,
  non-local means, RL deconvolution, the two published learned arms and the two proposed arms that
  include G-ESRGAN.
  Means over three scenes; the CSV twin of the figure holds every plotted value.}
  \label{fig:nagcurves}
\end{figure*}

\begin{figure*}[!t]
  \centering
  \includegraphics[width=\textwidth]{nag_crops.png}
  \caption{Matched crops at the most severe condition ($s = 1$): text (top) and the distant-windows
  region of the natural scene (bottom).}
  \label{fig:nagcrops}
\end{figure*}
""",
        " ".join(findings),
    ]
    (generated / "proposed_results.tex").write_text("\n".join(results) + "\n", encoding="utf-8")
    written.append((generated / "proposed_results.tex").as_posix())

    abstract = (
        r"We then use the testbench to develop and test a new method, NAG (noise-aware guarded enhancement). "
        r"NAG extends Zero-DCE and Real-ESRGAN with a per-pixel noise map that is derived from the sensor model "
        rf"and carried through every stage in closed form; the map matched measured noise to within {worst_err * 100:.1f}\%. "
        rf"Guarding Real-ESRGAN {_verb(gesr['mean_delta'])} its PSNR by {abs(gesr['mean_delta']):.2f}~dB "
        rf"(Holm-corrected {_pexpr(gesr['p_holm'])}). "
        + (rf"Against Zero-DCE fine-tuned on the same data, noise conditioning "
           rf"{_verb(ncd_ft_noise['mean_delta'])} the noise estimate by {abs(ncd_ft_noise['mean_delta']):.2f} "
           rf"({_pexpr(ncd_ft_noise['p_holm'])}) and changed PSNR by {_signed(ncd_ft['mean_delta'])}~dB "
           rf"({_pexpr(ncd_ft['p_holm'])}). " if ncd_ft["n"] else "")
        + rf"The full method {_verb(nag_isp['mean_delta'])} PSNR relative to the control ISP by "
        rf"{abs(nag_isp['mean_delta']):.2f}~dB"
        # Attribute a loss to the curves only when the arm without the curve and
        # denoising stages does not share it and the denoiser alone barely moves it.
        + (rf", a loss that comes from the curve stage (G-ESRGAN alone differed from the control by "
           rf"{_signed(gesr_isp)}~dB, and removing the denoiser changed NAG's PSNR by "
           rf"{_signed(nag_nodenoise['mean_delta'])}~dB), whose brightening a full-reference metric scores as error."
           if nag_isp["mean_delta"] < 0 <= gesr_isp
           and abs(nag_nodenoise["mean_delta"]) < 0.1 * abs(nag_isp["mean_delta"]) else ".")
    )
    (generated / "nag_abstract.tex").write_text(abstract + "\n", encoding="utf-8")
    conclusion = (
        r"The same instrument was used to develop a new method as well as to measure existing ones. Carrying an "
        r"exact noise map through each stage gave a curve estimator, a denoiser and a generative upsampler a "
        rf"common, physically grounded signal. Across the sweep, the guard {_verb(gesr['mean_delta'])} "
        rf"Real-ESRGAN's PSNR by {abs(gesr['mean_delta']):.2f}~dB, and noise conditioning "
        rf"{_verb(ncd['mean_delta'])} Zero-DCE's by {abs(ncd['mean_delta']):.2f}~dB "
        rf"({_signed(ncd_ft['mean_delta'])}~dB against Zero-DCE fine-tuned on the same data, against which it "
        rf"{_verb(ncd_ft_noise['mean_delta'])} the noise estimate by {abs(ncd_ft_noise['mean_delta']):.2f}, "
        rf"Holm-corrected {_pexpr(ncd_ft_noise['p_holm'])}). Against the control ISP, "
        rf"the full method's PSNR difference was {_signed(nag_isp['mean_delta'])}~dB. "
        r"Section~\ref{sec:nag} gives the per-component ablations."
    )
    (generated / "nag_conclusion.tex").write_text(conclusion + "\n", encoding="utf-8")
    written += [(generated / "nag_abstract.tex").as_posix(), (generated / "nag_conclusion.tex").as_posix()]

    return {
        "written": written,
        "macros": macros,
        "main": main,
        "ablation": abl,
        "tests": tests,
        "noise": noise,
        "training": training,
        "figures": {k: Path(v).as_posix() for k, v in figs.items()},
        "findings": findings,
        "unavailable": unavailable,
        "abstract": abstract,
        "conclusion": conclusion,
        "observations": observations,
    }
