/* Overview: the run at a glance - what hardware was emulated, what each arm
 * recovers on average, which arm holds up best on the worst hardware, and
 * whether the run can be trusted (calibration gate, gaps, provenance). */

import { api } from "../api.js";
import { lineChart } from "../chart.js";
import {
  h, clear, card, stat, badge, table, select, segmented, field, fmt, fmtMetric, fmtSigned,
  metricLabel, metricUnit, directionNote, pipelineInfo, pipelineName, isBetter, bestIndex,
  MAIN_ARMS, metricOptions, defaultMetric, sortedConditions, describeCondition, emptyState, loading,
} from "../ui.js";

export const title = "Overview";

const BASELINE = "classical_isp";
const ABLATION_SET = ["zero_dce", "zero_dce_ft", "nc_dce", "nc_dce_nosigma", "nc_dce_nonoise",
  "nag", "nag_nodenoise", "nag_nogate", "nag_nobp"];

export function armPresets(available) {
  const hasNag = available.some((name) => ["nag", "ablation"].includes(pipelineInfo(name).group));
  if (!hasNag) return null;
  const pick = (names) => names.filter((name) => available.includes(name));
  return {
    main: pick(MAIN_ARMS),
    ablations: pick(ABLATION_SET),
    all: available,
  };
}

export async function render(main, ctx, stale) {
  const index = ctx.index;
  const conditions = sortedConditions(index);
  const clean = conditions[0];
  const worst = conditions[conditions.length - 1];
  const available = index.pipelines.filter((p) => p.available).map((p) => p.name);
  const presets = armPresets(available);
  const directions = index.directions ?? {};

  let metric = defaultMetric(index, ctx.params.metric);
  let preset = presets && presets[ctx.params.arms] ? ctx.params.arms : "main";

  ctx.header("Overview",
    `${index.experiment_id} · ${index.images.length} scenes × ${conditions.length} hardware conditions × ${available.length} pipelines`);

  const run = await api.run(ctx.exp).catch(() => null);
  if (stale()) return;

  const kpis = h("div.stats");
  const chartSlot = h("div");
  const leaderSlot = h("div");
  const arms = () => (presets ? presets[preset] : available);

  const filters = h("div.filters",
    null,
    field("Metric", select(metricOptions(index), metric, (value) => {
      metric = value;
      ctx.setParams({ metric });
      refresh();
    }, { label: "Metric" })),
    presets ? field("Arms", segmented(
      [["main", "Main arms"], ["ablations", "Ablations"], ["all", "All"]], preset, (value) => {
        preset = value;
        ctx.setParams({ arms: value === "main" ? null : value });
        refresh();
      }, { label: "Arms" })) : null,
    h("p.filters-note", null, "Every value is the mean over the run's scenes. Hover a legend entry to see the range across scenes."));

  clear(main,
    kpis,
    filters,
    chartSlot,
    h("div.grid-2", null, leaderSlot, sweepCard(index, conditions, ctx)),
    h("div.grid-2", null, calibrationCard(run), provenanceCard(run, index)),
    gapsCard(run));

  async function refresh() {
    const names = [...new Set([...(available.includes(BASELINE) ? [BASELINE] : []), ...arms()])];
    clear(chartSlot, card(null, {}, loading()));
    let data;
    try {
      data = await api.aggregate(ctx.exp, metric, names);
    } catch (err) {
      clear(chartSlot, card("Across the sweep", {}, emptyState("Could not load the sweep", String(err.message || err))));
      return;
    }
    if (stale()) return;
    renderKpis(data, names);
    renderChart(data, names);
    renderLeaderboard(data, names);
  }

  function valueAt(data, name, condition) {
    return data.series[name]?.find((p) => p.hash === condition.hash)?.mean ?? null;
  }

  function renderKpis(data, names) {
    const direction = directions[metric];
    const unit = metricUnit(metric);
    const contenders = names.filter((name) => name !== BASELINE || names.length === 1);
    const worstValues = contenders.map((name) => valueAt(data, name, worst));
    const bestAt = bestIndex(worstValues, direction);
    const bestName = bestAt >= 0 ? contenders[bestAt] : null;

    const baseClean = valueAt(data, BASELINE, clean);
    const baseWorst = valueAt(data, BASELINE, worst);
    const lost = baseClean !== null && baseWorst !== null ? baseWorst - baseClean : null;

    const passed = run?.validation?.passed;
    clear(kpis,
      h("div.stat.hero", null,
        h("span.stat-label", null, `Best ${metricLabel(metric)} on the worst hardware`),
        h("span.stat-value", null, bestName ? fmtMetric(metric, worstValues[bestAt]) : "—", unit ? h("span.stat-unit", null, unit) : null),
        h("span.stat-note", null, bestName ? pipelineName(bestName) : "no finite values",
          h("span.muted", null, ` · severity ${fmt(worst.severity, 2)} · ${directionNote(direction)}`))),
      stat(`${metricLabel(metric)} lost to hardware`, lost === null ? "—" : fmtSigned(lost, null), {
        unit,
        note: available.includes(BASELINE) ? `baseline, clean → worst (${fmtMetric(metric, baseClean)} → ${fmtMetric(metric, baseWorst)})` : "no baseline arm in this run",
      }),
      stat("Hardware sweep", `${conditions.length}`, {
        unit: "conditions",
        note: `severity ${fmt(clean.severity, 2)} → ${fmt(worst.severity, 2)} · ${index.images.length} scenes`,
      }),
      stat("Pipelines that ran", `${available.length}`, {
        unit: `of ${index.pipelines.length}`,
        note: available.length === index.pipelines.length ? "the whole bank ran" : `${index.pipelines.length - available.length} recorded as gaps`,
        noteState: available.length === index.pipelines.length ? null : "warn",
      }),
      h("div.stat", null,
        h("span.stat-label", null, "Calibration gate"),
        h("span.stat-value.small", null, passed === undefined || passed === null ? badge("neutral", "not recorded")
          : passed ? badge("good", "Passed") : badge("bad", "Failed")),
        h("span.stat-note", null, run?.validation ? `${run.validation.checks?.length ?? 0} checks · noise model vs measurement` : "no validation.json in this run")));
  }

  function renderChart(data, names) {
    const direction = directions[metric];
    const unit = metricUnit(metric);
    const chart = lineChart({
      series: names.map((name) => {
        const info = pipelineInfo(name);
        return {
          name, label: info.label, color: info.color, dash: info.dash,
          points: data.series[name].map((p) => ({ x: p.severity, y: p.mean, lo: p.min, hi: p.max, n: p.n })),
        };
      }),
      height: 340,
      xTitle: "Degradation severity",
      yTitle: `${metricLabel(metric)}${unit ? ` (${unit})` : ""} · mean of ${data.n_images} scenes`,
      xFormat: (x) => fmt(x, 2),
      yFormat: (y) => fmtMetric(metric, y),
      ariaLabel: `${metricLabel(metric)} across the degradation sweep`,
      onPick: (severity) => ctx.go("explore", { sev: conditions.findIndex((c) => c.severity === severity), metric }),
      tooltipNote: () => "Click to open this condition in the Explorer",
    });
    clear(chartSlot, card(`${metricLabel(metric)} across the sweep`, {
      sub: `${directionNote(direction)} · severity 0 is the clean sensor, 1 the weakest hardware emulated`,
      tools: chart.controls,
      cls: "chart-card",
    }, chart.el));
  }

  function renderLeaderboard(data, names) {
    const direction = directions[metric];
    const baseWorst = valueAt(data, BASELINE, worst);
    const rows = names.map((name) => ({
      name,
      clean: valueAt(data, name, clean),
      worst: valueAt(data, name, worst),
    }));
    rows.forEach((row) => {
      row.change = row.clean !== null && row.worst !== null ? row.worst - row.clean : null;
      row.vsBase = baseWorst !== null && row.worst !== null && row.name !== BASELINE ? row.worst - baseWorst : null;
    });
    rows.sort((a, b) => {
      const better = isBetter(direction, a.worst, b.worst);
      if (better === null) return a.worst === null ? 1 : -1;
      return a.worst === b.worst ? 0 : better ? -1 : 1;
    });
    const signed = (value) => {
      if (value === null) return "—";
      // A target-valued metric has no "better" sign for a difference.
      const state = value === 0 || direction === null ? "" : (direction ? value > 0 : value < 0) ? ".good" : ".bad";
      return h(`span.delta${state}`, null, fmtSigned(value, null));
    };
    clear(leaderSlot, card("Worst-hardware leaderboard", {
      sub: `${metricLabel(metric)} at severity ${fmt(worst.severity, 2)}, mean over scenes · click a row to inspect it`,
    }, table([
      { key: "rank", label: "#", align: "num", render: (row) => String(rows.indexOf(row) + 1) },
      { key: "name", label: "Pipeline", align: "text", render: (row) => pipelineName(row.name) },
      { key: "clean", label: "Clean", render: (row) => fmtMetric(metric, row.clean) },
      { key: "worst", label: "Worst", best: direction, render: (row) => fmtMetric(metric, row.worst) },
      { key: "vsBase", label: "vs baseline", title: "difference from the baseline on the same hardware", render: (row) => signed(row.vsBase) },
    ], rows, {
      onRowClick: (row) => ctx.go("explore", { sev: conditions.length - 1, right: row.name, metric }),
    })));
  }

  await refresh();
}

function sweepCard(index, conditions, ctx) {
  const rows = conditions.map((c, i) => ({ ...c, i, ...(c.params ?? {}) }));
  return card("The hardware sweep", {
    sub: "What each severity emulates, applied to linear RAW before demosaicing · click a row to view it",
  }, table([
    { key: "severity", label: "Severity", align: "text", render: (r) => h("span.mono", null, fmt(r.severity, 2)) },
    { key: "alpha", label: "Light", title: "photon scale α: aperture, sensor area, exposure", render: (r) => r.alpha === undefined ? "—" : `${fmt(r.alpha * 100, r.alpha < 0.1 ? 1 : 0)}%` },
    { key: "sigma_r_dn", label: "Read noise", title: "read noise σ_r in DN", render: (r) => r.sigma_r_dn === undefined ? "—" : `${fmt(r.sigma_r_dn, 1)} DN` },
    { key: "r_psf", label: "Blur", title: "PSF radius in pixels", render: (r) => r.r_psf === undefined ? "—" : `${fmt(r.r_psf, 2)} px` },
    { key: "binning", label: "Binning", render: (r) => r.binning === undefined ? "—" : `${r.binning}×` },
    { key: "bit_depth", label: "ADC", render: (r) => r.bit_depth === undefined ? "—" : `${r.bit_depth}-bit` },
  ], rows, {
    onRowClick: (row) => ctx.go("explore", { sev: row.i }),
  }), h("p.card-foot", null, `Severity ${fmt(conditions[conditions.length - 1].severity, 2)}: ${describeCondition(conditions[conditions.length - 1])}.`));
}

function calibrationCard(run) {
  const validation = run?.validation;
  if (!validation) {
    return card("Calibration gate", {}, emptyState("No calibration record", "This run wrote no validation.json.", "python -m sensorsim.cli validate configs/experiments/pilot.yaml"));
  }
  const checks = validation.checks ?? [];
  return card("Calibration gate", {
    sub: "The degrader's noise model checked against measurement before any experiment ran",
    tools: validation.passed ? badge("good", "Passed") : badge("bad", "Failed"),
  },
  checks.length ? table([
    { key: "name", label: "Check", align: "text", render: (c) => h("span", { title: c.detail ?? "" }, c.name.replaceAll("_", " ")) },
    { key: "measured", label: "Measured", render: (c) => fmtSci(c.measured) },
    { key: "synthetic", label: "Model", render: (c) => fmtSci(c.synthetic) },
    { key: "relative_error", label: "Error", render: (c) => c.relative_error === undefined ? "—" : `${fmt(c.relative_error * 100, 2)}%` },
    { key: "tolerance", label: "Tolerance", render: (c) => c.tolerance === undefined ? "—" : `${fmt(c.tolerance * 100, 0)}%` },
    { key: "passed", label: "", align: "text", render: (c) => c.passed ? badge("good", "pass") : badge("bad", "fail") },
  ], checks) : null,
  validation.notes?.length ? h("ul.notes", null, validation.notes.map((note) => h("li", null, note))) : null);
}

function fmtSci(value) {
  if (value === null || value === undefined) return "—";
  if (value !== 0 && Math.abs(value) < 0.01) return value.toExponential(2).replace("-", "−");
  return fmt(value);
}

function provenanceCard(run, index) {
  const m = run?.manifest;
  if (!m) return card("Provenance", {}, emptyState("No manifest", "This run wrote no manifest.json."));
  const started = m.started_at ? new Date(m.started_at) : null;
  const finished = m.finished_at ? new Date(m.finished_at) : null;
  const seconds = started && finished ? (finished - started) / 1000 : null;
  const hw = m.hardware_info ?? {};
  const pk = m.packages ?? {};
  const rows = [
    ["Experiment", index.experiment_id],
    ["Config hash", m.config_hash],
    ["Seed", m.seed],
    ["Finished", finished ? `${finished.toLocaleString()}${seconds !== null ? ` · ${fmt(seconds, 0)} s` : ""}` : null],
    ["Git commit", m.git_commit],
    ["GPU", hw.gpu ?? (hw.cuda ? `CUDA ${hw.cuda}` : "none")],
    ["Platform", hw.platform],
    ["Python", pk.python],
    ["Key packages", ["numpy", "torch", "cv2", "pandas"].filter((k) => pk[k]).map((k) => `${k} ${pk[k]}`).join(" · ")],
  ].filter(([, v]) => v !== null && v !== undefined && v !== "");
  return card("Provenance", { sub: "Everything needed to rerun this exactly: sensorsim reproduce <config>" },
    h("dl.kv", null, rows.map(([k, v]) => [h("dt", null, k), h("dd", null, String(v))])));
}

function gapsCard(run) {
  const gaps = run?.gaps ?? [];
  if (!gaps.length) return null;
  return card("Gaps in the bank", {
    sub: "A pipeline that could not run is recorded, not omitted - an absent row would read as “not measured yet”; these read as “measured, and it could not run”.",
    cls: "gaps",
  }, h("ul.gap-list", null, gaps.map((gap) => h("li", null,
    pipelineName(gap.pipeline, { id: true }),
    h("span.gap-reason", null, gap.reason.replace(/^unavailable:\s*/, "")),
    h("span.muted", null, ` · ${gap.rows} rows`)))));
}
