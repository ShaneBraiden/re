/* Analysis: what `sensorsim analyze` and `sensorsim paper-nag` concluded from
 * this run - recovery curves, knees, significance, the NAG tables, training
 * curves, the noise-model check - plus every figure they rendered. Only the
 * sections whose tables exist are shown; nothing here is recomputed. */

import { api } from "../api.js";
import { lineChart } from "../chart.js";
import {
  h, clear, card, table, select, segmented, field, fmt, fmtP, fmtSigned, metricLabel, metricUnit,
  directionNote, pipelineInfo, pipelineName, badge, emptyState, openLightbox,
} from "../ui.js";
import { armPresets } from "./overview.js";

export const title = "Analysis";

const BASELINE = "classical_isp";
const MEAN_COLUMNS = ["psnr", "ssim", "delta_e", "nr_quality", "sharpness", "noise_sigma", "mtf50_edge", "sift_inlier_ratio"];

export async function render(main, ctx, stale) {
  ctx.header("Analysis", `${ctx.index.experiment_id} · recovery, knees, significance and the figures the paper is built from`);
  const data = await api.analysis(ctx.exp);
  if (stale()) return;

  if (!data.figures_dir) {
    clear(main, emptyState(
      "This run has not been analysed yet",
      "The analysis stage writes its tables and figures next to the paper. Run it, then reload this page.",
      `python -m sensorsim.cli analyze experiments/${ctx.exp}/metrics.csv --figures paper/figures/${ctx.exp}`));
    return;
  }

  const t = data.tables;
  const directions = ctx.index.directions ?? {};
  const sections = [
    ["recovery", "Recovery", t.recovery_aggregate?.length ? () => recoverySection(t.recovery_aggregate, ctx) : null],
    ["knees", "Knees", t.knees?.length ? () => kneesSection(t.knees) : null],
    ["significance", "Significance", t.significance?.length ? () => significanceSection(t.significance) : null],
    ["planned", "Planned tests", t.planned_tests?.length ? () => plannedSection(t.planned_tests) : null],
    ["arms", "Arm means", t.main_means?.length || t.ablation_means?.length ? () => meansSection(t.main_means, t.ablation_means, directions) : null],
    ["divergence", "Divergence", data.divergence ? () => divergenceSection(data.divergence) : null],
    ["training", "Training", t.nag_training?.length ? () => trainingSection(t.nag_training) : null],
    ["noise", "Noise model", t.noise_validation?.length || t.photon_transfer?.length ? () => noiseSection(t.noise_validation, t.photon_transfer) : null],
    ["figures", "Figures", data.figures.length ? () => figuresSection(data.figures) : null],
  ].filter(([, , build]) => build);

  const subnav = h("nav.subnav", { "aria-label": "Sections" }, sections.map(([id, label]) => {
    const link = h("button.subnav-link", { type: "button" }, label);
    link.addEventListener("click", () => document.getElementById(`sec-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" }));
    return link;
  }));

  clear(main,
    h("p.source", null, "Read from ", h("code", null, data.figures_dir), " — regenerate with ", h("code", null, "sensorsim analyze"), "."),
    subnav,
    sections.map(([id, , build]) => h("div.section", { id: `sec-${id}` }, build())));
}

/* --------------------------------------------------------------- recovery --- */

function recoverySection(rows, ctx) {
  const metrics = [...new Set(rows.map((r) => r.metric_name))];
  let metric = metrics.includes("psnr") ? "psnr" : metrics[0];
  const names = [...new Set(rows.map((r) => r.pipeline))].filter((n) => n !== BASELINE);
  const presets = armPresets(names);
  let preset = presets ? "main" : "all";
  const slot = h("div");

  const draw = () => {
    const shown = presets ? presets[preset].filter((n) => n !== BASELINE) : names;
    const chart = lineChart({
      series: shown.map((name) => {
        const info = pipelineInfo(name);
        return {
          name, label: info.label, color: info.color, dash: info.dash,
          points: rows.filter((r) => r.pipeline === name && r.metric_name === metric)
            .map((r) => ({ x: r.severity, y: r.median, lo: r.ci_low, hi: r.ci_high, n: r.n })),
        };
      }).filter((ser) => ser.points.length),
      height: 340,
      xTitle: "Degradation severity",
      yTitle: `Recovery ratio R · ${metricLabel(metric)} · median over scenes`,
      xFormat: (x) => fmt(x, 2),
      yFormat: (y) => fmt(y, 2),
      refLines: [{ y: 0, label: "R = 0 · baseline" }, { y: 1, label: "R = 1 · clean sensor" }],
      ariaLabel: `Recovery ratio for ${metricLabel(metric)}`,
    });
    clear(slot, card("Recovery ratio across the sweep", {
      sub: "R = (arm − baseline on degraded) / (baseline on clean − baseline on degraded): the share of the hardware loss an arm wins back. Hover a legend entry for its 95% interval.",
      tools: chart.controls,
      cls: "chart-card",
    },
    h("div.card-filters", null,
      field("Metric", select(metrics.map((m) => [m, metricLabel(m.replace(/_error$/, "")) + (m.endsWith("_error") ? " (distance to target)" : "")]), metric, (v) => { metric = v; draw(); }, { label: "Recovery metric" })),
      presets ? field("Arms", segmented([["main", "Main arms"], ["ablations", "Ablations"], ["all", "All"]], preset, (v) => { preset = v; draw(); })) : null),
    chart.el,
    h("p.card-foot", null, "Low severities are dropped where the hardware loss is under 5% of the sweep's largest: there the ratio divides a small difference by a smaller one.")));
  };
  draw();
  return slot;
}

/* ------------------------------------------------------------------ knees --- */

function kneesSection(rows) {
  return card("Where each arm stops keeping up", {
    sub: "The knee of each recovery curve, located three independent ways. When the methods disagree, the knee is reported as not identifiable rather than averaged into a number.",
  }, table([
    { key: "pipeline", label: "Pipeline", align: "text", render: (r) => pipelineName(r.pipeline) },
    { key: "metric_name", label: "Metric", align: "text", render: (r) => metricLabel(r.metric_name) },
    { key: "knee", label: "Knee", render: (r) => fmt(r.knee, 2) },
    { key: "knee_kneedle", label: "Kneedle", render: (r) => fmt(r.knee_kneedle, 2) },
    { key: "knee_curvature", label: "Curvature", render: (r) => fmt(r.knee_curvature, 2) },
    { key: "knee_piecewise", label: "Piecewise", render: (r) => fmt(r.knee_piecewise, 2) },
    { key: "agreement", label: "Spread", title: "how far apart the three methods land", render: (r) => fmt(r.agreement, 2) },
    { key: "identifiable", label: "Verdict", align: "text", render: (r) => r.identifiable ? badge("good", "identifiable") : badge("warn", "not identifiable") },
    { key: "curve_model", label: "Fit", align: "text", render: (r) => h("span", null, r.curve_model ?? "—", h("span.muted", null, r.curve_r2 !== null ? ` R² ${fmt(r.curve_r2, 3)}` : "")) },
  ], rows));
}

/* ----------------------------------------------------------- significance --- */

function significanceSection(rows) {
  return card("Pairwise significance", {
    sub: "Paired tests over every scene × condition, p adjusted for multiple comparisons.",
  }, table([
    { key: "metric_name", label: "Metric", align: "text", render: (r) => metricLabel(r.metric_name) },
    { key: "pair", label: "Comparison", align: "text", render: (r) => h("span.pair", null, pipelineName(r.pipeline_a), h("span.muted", null, " vs "), pipelineName(r.pipeline_b)) },
    { key: "n_pairs", label: "n", render: (r) => fmt(r.n_pairs, 0) },
    { key: "mean_difference", label: "Mean diff.", title: "A − B", render: (r) => fmtSigned(r.mean_difference) },
    { key: "effect_size", label: "Effect", title: "rank-biserial effect size", render: (r) => fmtSigned(r.effect_size, 2) },
    { key: "p_adjusted", label: "p (adj.)", render: (r) => fmtP(r.p_adjusted) },
    { key: "significant", label: "", align: "text", render: (r) => r.significant ? badge("good", "significant") : badge("neutral", "n.s.") },
  ], rows));
}

function plannedSection(rows) {
  const metrics = [...new Set(rows.map((r) => r.metric))];
  let metric = metrics.includes("psnr") ? "psnr" : metrics[0];
  const slot = h("div");
  const draw = () => {
    const shown = rows.filter((r) => r.metric === metric);
    clear(slot, card("Planned comparisons", {
      sub: "Each proposed component against the arm it extends, Wilcoxon signed-rank over scenes × conditions, Holm-corrected.",
      tools: select(metrics.map((m) => [m, metricLabel(m)]), metric, (v) => { metric = v; draw(); }, { label: "Metric" }),
    }, table([
      { key: "arm", label: "Arm", align: "text", render: (r) => pipelineName(r.arm) },
      { key: "against", label: "Against", align: "text", render: (r) => pipelineName(r.against) },
      { key: "n", label: "n", render: (r) => fmt(r.n, 0) },
      { key: "mean_delta", label: "Mean Δ", render: (r) => fmtSigned(r.mean_delta) },
      { key: "median_delta", label: "Median Δ", render: (r) => fmtSigned(r.median_delta) },
      { key: "wins", label: "Wins", title: "pairs where the arm beat its comparator", render: (r) => `${fmt(r.wins, 0)} / ${fmt(r.n, 0)}` },
      { key: "p_holm", label: "p (Holm)", render: (r) => fmtP(r.p_holm) },
      { key: "significant", label: "", align: "text", render: (r) => r.significant ? badge("good", "significant") : badge("neutral", "n.s.") },
    ], shown)));
  };
  draw();
  return slot;
}

/* ------------------------------------------------------------- arm means --- */

function meansSection(mainRows, ablationRows, directions) {
  const sets = [["main", "Main arms", mainRows], ["ablation", "Ablations", ablationRows]].filter(([, , rows]) => rows?.length);
  let which = sets[0][0];
  const slot = h("div");
  const draw = () => {
    const rows = sets.find(([key]) => key === which)[2];
    const columns = MEAN_COLUMNS.filter((m) => m in rows[0]);
    clear(slot, card("Arm means over the whole sweep", {
      sub: "Each value averages every scene and every condition. The best value in each column is marked.",
      tools: sets.length > 1 ? segmented(sets.map(([key, label]) => [key, label]), which, (v) => { which = v; draw(); }) : null,
    }, table([
      { key: "pipeline", label: "Pipeline", align: "text", render: (r) => pipelineName(r.pipeline) },
      ...columns.map((m) => ({
        key: m,
        label: metricLabel(m),
        title: `${m}${metricUnit(m) ? ` (${metricUnit(m)})` : ""} · ${directionNote(directions[m] ?? true)}`,
        best: directions[m] === undefined ? undefined : directions[m],
        render: (r) => fmt(r[m]),
      })),
    ], rows)));
  };
  draw();
  return slot;
}

/* ------------------------------------------------------------- divergence --- */

function divergenceSection(divergence) {
  const pooled = divergence.pooled ?? {};
  const rows = Object.values(divergence.per_pipeline ?? {});
  return card("Fidelity vs perception", {
    sub: `Does ${metricLabel(pooled.fidelity_metric ?? "psnr")} move with ${metricLabel(pooled.perceptual_metric ?? "nr_quality")}? Spearman ρ across scenes × conditions; a significant negative ρ means an arm trades one for the other.`,
  },
  h("div.callout", null,
    pooled.diverges ? badge("warn", "diverges") : badge("good", "no divergence"),
    h("span", null, ` All arms pooled: ρ = ${fmt(pooled.spearman_rho, 2)}, p = ${fmtP(pooled.p_value)}, n = ${fmt(pooled.n, 0)}. ${pooled.note ?? ""}`)),
  table([
    { key: "pipeline", label: "Pipeline", align: "text", render: (r) => pipelineName(r.pipeline) },
    { key: "n", label: "n", render: (r) => fmt(r.n, 0) },
    { key: "spearman_rho", label: "ρ", render: (r) => rhoBar(r.spearman_rho) },
    { key: "p_value", label: "p", render: (r) => fmtP(r.p_value) },
    { key: "diverges", label: "", align: "text", render: (r) => r.diverges ? badge("warn", "diverges") : badge("neutral", "no") },
  ], rows));
}

function rhoBar(rho) {
  if (rho === null || rho === undefined) return "—";
  const width = Math.min(50, Math.abs(rho) * 50);
  return h("span.rho", null,
    h("span.rho-track", { "aria-hidden": "true" },
      h("span.rho-fill", { style: { left: rho >= 0 ? "50%" : `${50 - width}%`, width: `${width}%` } })),
    h("span.rho-value", null, fmtSigned(rho, 2)));
}

/* --------------------------------------------------------------- training --- */

function trainingSection(rows) {
  const terms = ["total", "noise", "exp", "col", "spa", "tv"].filter((k) => k in rows[0]);
  const termLabels = { total: "Total", noise: "Noise", exp: "Exposure", col: "Colour", spa: "Spatial", tv: "Smoothness" };
  let term = "total";
  const variants = [...new Set(rows.map((r) => r.variant))];
  const slot = h("div");
  const draw = () => {
    const chart = lineChart({
      series: variants.map((variant) => {
        const info = pipelineInfo(variant);
        return {
          name: variant, label: info.label, color: info.color, dash: info.dash,
          points: rows.filter((r) => r.variant === variant).map((r) => ({ x: r.epoch, y: r[term] })),
        };
      }),
      height: 300,
      xTitle: "Epoch",
      yTitle: `${termLabels[term] ?? term} loss`,
      xFormat: (x) => fmt(x, 0),
      yFormat: (y) => fmt(y),
      ariaLabel: `${termLabels[term] ?? term} training loss per epoch`,
    });
    const stopped = variants.filter((v) => rows.filter((r) => r.variant === v).length < Math.max(...variants.map((w) => rows.filter((r) => r.variant === w).length)));
    clear(slot, card("Training curves", {
      sub: "Zero-reference training of NC-DCE and its ablations, one point per epoch.",
      tools: chart.controls,
      cls: "chart-card",
    },
    h("div.card-filters", null, field("Loss term", segmented(terms.map((k) => [k, termLabels[k] ?? k]), term, (v) => { term = v; draw(); }))),
    chart.el,
    stopped.length ? h("p.card-foot", null, `Stopped early: ${stopped.map((v) => pipelineInfo(v).label).join(", ")} — see the training record in models/.`) : null));
  };
  draw();
  return slot;
}

/* ------------------------------------------------------------ noise model --- */

function noiseSection(validation, photon) {
  const ratios = (validation ?? []).map((r) => r.ratio).filter((x) => x !== null);
  const sorted = [...ratios].sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
  return card("Noise model check", {
    sub: "Noise σ measured on degraded flat fields against the Poisson–Gaussian prediction, var = μ/g + σ_read². A ratio of 1 means the model is exact.",
  },
  ratios.length ? h("div.callout", null, badge(Math.abs(median - 1) < 0.05 ? "good" : "warn", "measured / predicted"),
    h("span", null, ` ${ratios.length} checks · median ${fmt(median, 3)} · range ${fmt(sorted[0], 3)}–${fmt(sorted[sorted.length - 1], 3)}`)) : null,
  validation?.length ? h("div.scroll-table", null, table([
    { key: "alpha", label: "Light", render: (r) => `${fmt(r.alpha * 100, r.alpha < 0.1 ? 1 : 0)}%` },
    { key: "sigma_r_dn", label: "Read noise", render: (r) => `${fmt(r.sigma_r_dn, 1)} DN` },
    { key: "binning", label: "Bin", render: (r) => `${fmt(r.binning, 0)}×` },
    { key: "bit_depth", label: "ADC", render: (r) => `${fmt(r.bit_depth, 0)}-bit` },
    { key: "level", label: "Level", render: (r) => fmt(r.level, 2) },
    { key: "channel", label: "Ch.", align: "text", render: (r) => r.channel },
    { key: "measured", label: "Measured σ", render: (r) => fmt(r.measured, 5) },
    { key: "predicted", label: "Predicted σ", render: (r) => fmt(r.predicted, 5) },
    { key: "ratio", label: "Ratio", render: (r) => fmt(r.ratio, 3) },
  ], validation)) : null,
  photon?.length ? h("details.more", null, h("summary", null, `Photon transfer points (${photon.length})`), table([
    { key: "source", label: "Source", align: "text", render: (r) => r.source },
    { key: "mean", label: "Mean", render: (r) => fmt(r.mean, 4) },
    { key: "variance", label: "Variance", render: (r) => r.variance?.toExponential(3) ?? "—" },
  ], photon)) : null);
}

/* ---------------------------------------------------------------- figures --- */

function figuresSection(figures) {
  return card(`Figures (${figures.length})`, {
    sub: "Rendered by the analysis stage for the paper. Click one to view it full size.",
  }, h("div.gallery", null, figures.map((figure) => {
    const name = figure.name.replaceAll("_", " ");
    const tile = h("button.figure", { type: "button", title: figure.name },
      h("img", { src: figure.url, alt: name, loading: "lazy" }),
      h("span.figure-name", null, name));
    tile.addEventListener("click", () => openLightbox(figure.url, figure.name));
    return tile;
  })));
}

