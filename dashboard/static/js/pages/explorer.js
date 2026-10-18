/* Explorer: one scene, one hardware condition, any two outputs side by side.
 *
 * The severity slider, scene and metric scope everything below them. The
 * image comparison, the ranking at this condition, the sweep chart and the
 * full metric table all re-render against the same slice. */

import { api, previewURL } from "../api.js";
import { lineChart } from "../chart.js";
import { compareViewer } from "../viewer.js";
import {
  h, clear, card, table, select, field, fmt, fmtMetric, fmtSigned, metricLabel, metricUnit,
  directionNote, pipelineInfo, pipelineName, swatch, isBetter, GROUPS, FAMILY_LABELS,
  metricOptions, defaultMetric, sortedConditions, describeCondition, emptyState, loading, icon,
} from "../ui.js";
import { armPresets } from "./overview.js";

export const title = "Explorer";

const BASELINE = "classical_isp";
const REFERENCE = "__reference__";
const DEGRADED = "__degraded__";

export async function render(main, ctx, stale) {
  const index = ctx.index;
  const exp = ctx.exp;
  const conditions = sortedConditions(index);
  const directions = index.directions ?? {};
  const families = index.families ?? {};
  const available = index.pipelines.filter((p) => p.available).map((p) => p.name);
  const presets = armPresets(available);
  // Flask sorts JSON keys, so put arms back in bank order before listing them.
  const inBankOrder = (names) => [...names].sort((a, b) => available.indexOf(a) - available.indexOf(b));
  const p = ctx.params;

  const defaultActive = new Set(presets ? presets.main : available);
  const state = {
    image: index.images.some((i) => i.image_id === p.image) ? p.image : index.images[0]?.image_id,
    sev: Math.min(conditions.length - 1, Math.max(0, Number.isFinite(Number(p.sev)) ? Number(p.sev) : 0)),
    metric: defaultMetric(index, p.metric),
    active: p.pipes ? new Set(p.pipes.split(",").filter((n) => available.includes(n))) : new Set(defaultActive),
    left: p.left ?? DEGRADED,
    right: p.right && available.includes(p.right) ? p.right : null,
    series: null,
    values: null,
  };
  if (state.right && !state.active.has(state.right)) state.active.add(state.right);
  if (!state.right) state.right = [...state.active].find((n) => pipelineInfo(n).proposed) ?? [...state.active].pop() ?? REFERENCE;
  const validSide = (v) => v === REFERENCE || v === DEGRADED || available.includes(v);
  if (!validSide(state.left)) state.left = DEGRADED;

  ctx.header("Explorer", "One scene under one hardware condition — compare any two outputs pixel for pixel");

  if (!state.image) {
    clear(main, emptyState("This run has no scenes", "dashboard.json lists no images."));
    return;
  }

  /* ---------------------------------------------------------- filters --- */

  const scenePicker = h("div.scenes", { role: "group", "aria-label": "Scene" });
  for (const image of index.images) {
    const button = h("button.scene", { type: "button", "aria-pressed": String(image.image_id === state.image), title: image.image_id },
      h("img", { src: previewURL(exp, image.image_id, REFERENCE), alt: "", loading: "lazy" }),
      h("span", null, image.image_id));
    button.addEventListener("click", () => {
      state.image = image.image_id;
      for (const b of scenePicker.children) b.setAttribute("aria-pressed", String(b === button));
      sync();
      loadSeries();
      loadValues();
    });
    scenePicker.append(button);
  }

  const slider = h("input", {
    type: "range", min: 0, max: conditions.length - 1, step: 1, value: state.sev,
    "aria-label": "Degradation severity", "aria-valuetext": "",
  });
  const sevValue = h("b.sev-value");
  const sevDesc = h("span.hint.sev-desc");
  const ticks = h("div.ticks", { "aria-hidden": "true" }, conditions.map((c, i) => {
    const tick = h("button.tick-btn", { type: "button", tabindex: -1 }, fmt(c.severity, 2));
    tick.addEventListener("click", () => setSev(i));
    return tick;
  }));
  slider.addEventListener("input", () => setSev(Number(slider.value)));

  const metricSelect = select(metricOptions(index), state.metric, (value) => {
    state.metric = value;
    sync();
    loadSeries();
  }, { label: "Metric" });

  const filters = h("div.filters.sticky", null,
    field("Scene", scenePicker),
    h("div.field.grow", null,
      h("span.field-label", null, "Hardware severity ", sevValue),
      slider, ticks, sevDesc),
    field("Metric", metricSelect));

  /* -------------------------------------------------------- pipelines --- */

  const chips = new Map();
  const picker = h("div.picker");
  for (const group of GROUPS) {
    const members = index.pipelines.filter((pl) => pipelineInfo(pl.name).group === group.id);
    if (!members.length) continue;
    const names = members.filter((m) => m.available).map((m) => m.name);
    const groupToggle = h("button.link-btn", { type: "button" });
    const updateGroupToggle = () => {
      const all = names.length && names.every((n) => state.active.has(n));
      groupToggle.textContent = all ? "none" : "all";
      groupToggle.hidden = !names.length;
    };
    groupToggle.addEventListener("click", () => {
      const all = names.every((n) => state.active.has(n));
      for (const n of names) {
        if (all) state.active.delete(n); else state.active.add(n);
      }
      afterActiveChange();
    });
    const row = h("div.picker-row", null,
      h("span.picker-group", null, group.label, groupToggle),
      h("div.chips", null, members.map((member) => {
        const info = pipelineInfo(member.name);
        const chip = h("button.chip", {
          type: "button",
          "aria-pressed": String(state.active.has(member.name)),
          disabled: !member.available,
          title: member.available ? member.name : `${member.name}: ${member.reason}`,
        }, swatch(member.name, { off: !member.available }), h("span", null, info.label),
        member.available ? null : h("span.chip-note", null, "did not run"));
        chip.addEventListener("click", () => {
          if (state.active.has(member.name)) state.active.delete(member.name);
          else state.active.add(member.name);
          afterActiveChange();
        });
        chips.set(member.name, chip);
        return chip;
      })));
    row.updateToggle = updateGroupToggle;
    picker.append(row);
  }
  const presetBar = presets ? h("div.preset-bar", null,
    h("span.hint", null, "Quick sets:"),
    [["main", "Main arms"], ["ablations", "Ablations"], ["all", "Everything"]].map(([key, label]) => {
      const b = h("button.link-btn", { type: "button" }, label);
      b.addEventListener("click", () => { state.active = new Set(presets[key]); afterActiveChange(); });
      return b;
    })) : null;

  function syncChips() {
    for (const [name, chip] of chips) chip.setAttribute("aria-pressed", String(state.active.has(name)));
    for (const row of picker.children) row.updateToggle?.();
  }

  function afterActiveChange() {
    syncChips();
    if (!validSide(state.right)) state.right = REFERENCE;
    sync();
    loadSeries();
    loadValues();
  }

  /* ------------------------------------------------------------ viewer --- */

  const viewer = compareViewer({ aspect: "1 / 1" });
  const sideOptions = () => [
    { group: "Reference", items: [[REFERENCE, "Ground truth (clean sensor)"], [DEGRADED, "Degraded input, no restoration"]] },
    ...GROUPS.map((g) => ({
      group: g.label,
      items: available.filter((n) => pipelineInfo(n).group === g.id).map((n) => [n, pipelineInfo(n).label]),
    })).filter((g) => g.items.length),
  ];
  const leftSelect = select(sideOptions(), state.left, (v) => { state.left = v; sync(); }, { label: "Left image" });
  const rightSelect = select(sideOptions(), state.right, (v) => { state.right = v; sync(); renderCondition(); }, { label: "Right image" });
  const swap = h("button.icon-btn", { type: "button", title: "Swap sides", "aria-label": "Swap sides" }, "⇄");
  swap.addEventListener("click", () => {
    [state.left, state.right] = [state.right, state.left];
    sync();
    renderCondition();
  });

  const viewerCard = card("Side-by-side", {
    sub: "Drag the divider to wipe · drag to pan · scroll or double-click to zoom",
    tools: h("div.side-selects", null, leftSelect, swap, rightSelect),
    cls: "viewer-card",
  }, viewer.el);

  /* ------------------------------------------------------------ panels --- */

  const conditionSlot = h("div");
  const chartSlot = h("div");
  const tableSlot = h("div");

  clear(main,
    filters,
    card("Pipelines", { sub: "Pick the arms to compare. Colour and line style follow the arm everywhere on the page.", tools: presetBar, cls: "picker-card" }, picker),
    h("div.explore-grid", null, viewerCard, conditionSlot),
    chartSlot,
    tableSlot);
  syncChips();

  function label(which) {
    if (which === REFERENCE) return "Ground truth";
    if (which === DEGRADED) return "Degraded input";
    return pipelineInfo(which).label;
  }

  function setSev(i) {
    state.sev = i;
    sync();
    loadValues();
    renderCondition();
    chart?.setMarker(conditions[i].severity);
  }

  /* Everything that follows from state without a fetch. */
  function sync() {
    const condition = conditions[state.sev];
    slider.value = String(state.sev);
    slider.setAttribute("aria-valuetext", `severity ${fmt(condition.severity, 2)}: ${describeCondition(condition)}`);
    sevValue.textContent = fmt(condition.severity, 2);
    sevDesc.textContent = state.sev === 0 ? `${describeCondition(condition)} — the clean sensor` : describeCondition(condition);
    for (const [i, tick] of [...ticks.children].entries()) tick.classList.toggle("on", i === state.sev);
    leftSelect.value = state.left;
    rightSelect.value = state.right;
    viewer.set({
      left: previewURL(exp, state.image, state.left, condition.hash),
      right: previewURL(exp, state.image, state.right, condition.hash),
      leftLabel: label(state.left),
      rightLabel: label(state.right),
    });
    ctx.setParams({
      image: state.image === index.images[0].image_id ? null : state.image,
      sev: state.sev || null,
      metric: state.metric === defaultMetric(index) ? null : state.metric,
      left: state.left === DEGRADED ? null : state.left,
      right: state.right,
      pipes: sameSet(state.active, defaultActive) ? null : [...state.active].join(","),
    });
  }

  /* ------------------------------------------------------------- data --- */

  let chart = null;
  let seriesTicket = 0;
  let valuesTicket = 0;

  async function loadSeries() {
    const ticket = ++seriesTicket;
    const names = [...state.active];
    if (!names.length) {
      state.series = null;
      renderChart();
      renderCondition();
      return;
    }
    chartSlot.classList.add("refreshing");
    try {
      const data = await api.series(exp, state.image, state.metric, names);
      if (ticket !== seriesTicket || stale()) return;
      state.series = data;
    } catch (err) {
      if (ticket !== seriesTicket || stale()) return;
      state.series = { error: String(err.message || err) };
    }
    chartSlot.classList.remove("refreshing");
    renderChart();
    renderCondition();
  }

  async function loadValues() {
    const ticket = ++valuesTicket;
    const names = [...state.active];
    if (!names.length) { state.values = null; renderTable(); return; }
    tableSlot.classList.add("refreshing");
    try {
      const data = await api.values(exp, state.image, conditions[state.sev].hash, names);
      if (ticket !== valuesTicket || stale()) return;
      state.values = data.values;
    } catch (err) {
      if (ticket !== valuesTicket || stale()) return;
      state.values = { error: String(err.message || err) };
    }
    tableSlot.classList.remove("refreshing");
    renderTable();
  }

  const pointAt = (name, i) => state.series?.series?.[name]?.[i]?.value ?? null;

  function renderCondition() {
    const condition = conditions[state.sev];
    const direction = directions[state.metric];
    const unit = metricUnit(state.metric);
    if (!state.series || state.series.error) {
      clear(conditionSlot, card(`At severity ${fmt(condition.severity, 2)}`, {},
        state.series?.error ? emptyState("Could not load values", state.series.error)
          : emptyState("No pipeline selected", "Switch on at least one arm above.")));
      return;
    }
    const names = inBankOrder(Object.keys(state.series.series));
    const focus = names.includes(state.right) ? state.right : null;
    const rows = names.map((name) => {
      const value = pointAt(name, state.sev);
      const clean = pointAt(name, 0);
      return { name, value, delta: value !== null && clean !== null ? value - clean : null };
    }).sort((a, b) => {
      const better = isBetter(direction, a.value, b.value);
      if (better === null) return a.value === null ? 1 : -1;
      return a.value === b.value ? 0 : better ? -1 : 1;
    });

    let hero = h("div.focus-hero.muted-hero", null,
      h("p.hint", null, "Pick an arm for the right side of the comparison to see its numbers here."));
    if (focus) {
      const value = pointAt(focus, state.sev);
      const clean = pointAt(focus, 0);
      const base = names.includes(BASELINE) && focus !== BASELINE ? pointAt(BASELINE, state.sev) : null;
      const rank = rows.findIndex((r) => r.name === focus) + 1;
      hero = h("div.focus-hero", null,
        h("span.stat-label", null, `${metricLabel(state.metric)} · `, pipelineName(focus)),
        h("span.hero-value", null, fmtMetric(state.metric, value), unit ? h("span.stat-unit", null, unit) : null),
        h("div.hero-notes", null,
          deltaNote(value, clean, direction, state.sev === 0 ? null : "vs this arm on the clean sensor"),
          base !== null ? deltaNote(value, base, direction, "vs the baseline here") : null,
          h("span.hint", null, `ranked ${rank} of ${rows.length} · ${directionNote(direction)}`)));
    }

    clear(conditionSlot, card(`At severity ${fmt(condition.severity, 2)}`, {
      sub: `${state.image} · ${describeCondition(condition)}`,
      cls: "condition-card",
    }, hero, table([
      { key: "rank", label: "#", render: (r) => String(rows.indexOf(r) + 1) },
      { key: "name", label: "Pipeline", align: "text", render: (r) => pipelineName(r.name) },
      { key: "value", label: metricLabel(state.metric), best: direction, render: (r) => fmtMetric(state.metric, r.value) },
      { key: "delta", label: "vs clean", title: "change from the same arm on the clean sensor", render: (r) => state.sev === 0 ? "—" : signed(r.delta, direction) },
    ], rows, {
      onRowClick: (row) => { state.right = row.name; sync(); renderCondition(); },
      activeRow: (row) => row.name === state.right,
    })));
  }

  function renderChart() {
    if (!state.series || state.series.error) {
      clear(chartSlot);
      chart = null;
      return;
    }
    const direction = directions[state.metric];
    const unit = metricUnit(state.metric);
    const names = inBankOrder(Object.keys(state.series.series));
    chart = lineChart({
      series: names.map((name) => {
        const info = pipelineInfo(name);
        return {
          name, label: info.label, color: info.color, dash: info.dash,
          points: state.series.series[name].map((pt) => ({ x: pt.severity, y: pt.value })),
        };
      }),
      height: 320,
      marker: conditions[state.sev].severity,
      xTitle: "Degradation severity",
      yTitle: `${metricLabel(state.metric)}${unit ? ` (${unit})` : ""} · ${state.image}`,
      xFormat: (x) => fmt(x, 2),
      yFormat: (y) => fmtMetric(state.metric, y),
      ariaLabel: `${metricLabel(state.metric)} across the sweep for ${state.image}`,
      onPick: (severity) => setSev(conditions.findIndex((c) => c.severity === severity)),
      tooltipNote: () => "Click to jump the comparison here",
    });
    clear(chartSlot, card(`${metricLabel(state.metric)} across the sweep`, {
      sub: `${state.image} · ${directionNote(direction)} · the shaded column is the condition shown above`,
      tools: chart.controls,
      cls: "chart-card",
    }, chart.el));
  }

  function renderTable() {
    if (!state.values || state.values.error) {
      clear(tableSlot, state.values?.error ? card("Every metric", {}, emptyState("Could not load values", state.values.error)) : null);
      return;
    }
    const names = inBankOrder(Object.keys(state.values));
    const metrics = [...new Set(names.flatMap((n) => Object.keys(state.values[n])))];
    const order = ["fidelity", "perceptual", "machine_task"];
    metrics.sort((a, b) => {
      const fa = order.indexOf(families[a] ?? "other");
      const fb = order.indexOf(families[b] ?? "other");
      return fa === fb ? 0 : (fa + 1 || 99) - (fb + 1 || 99);
    });
    const condition = conditions[state.sev];
    const rows = names.map((name) => ({ name, ...state.values[name] }));
    clear(tableSlot, card("Every metric at this condition", {
      sub: `${state.image} · severity ${fmt(condition.severity, 2)} · the best value in each column is marked · click a row to put it on the right`,
      cls: "metrics-card",
    }, table([
      { key: "name", label: "Pipeline", align: "text", render: (r) => pipelineName(r.name) },
      ...metrics.map((metric) => ({
        key: metric,
        label: h("span.th-metric", null, metricLabel(metric), h("span.th-family", null, FAMILY_LABELS[families[metric]] ?? "")),
        title: `${metric} · ${directionNote(directions[metric])}`,
        best: directions[metric] === undefined ? undefined : directions[metric],
        render: (r) => fmtMetric(metric, r[metric]),
      })),
    ], rows, {
      onRowClick: (row) => { state.right = row.name; sync(); renderCondition(); },
      activeRow: (row) => row.name === state.right,
    })));
  }

  sync();
  clear(conditionSlot, card(null, {}, loading()));
  await Promise.all([loadSeries(), loadValues()]);

  const onKey = (e) => {
    if (e.target.closest("input, select, textarea, [role=slider], .hit")) return;
    if (e.key === "[" || e.key === "]") {
      e.preventDefault();
      setSev(Math.min(conditions.length - 1, Math.max(0, state.sev + (e.key === "]" ? 1 : -1))));
    }
  };
  document.addEventListener("keydown", onKey);
  return () => document.removeEventListener("keydown", onKey);
}

function sameSet(a, b) {
  return a.size === b.size && [...a].every((x) => b.has(x));
}

function signed(delta, direction) {
  if (delta === null || delta === undefined) return "—";
  const state = delta === 0 || direction === null ? "" : (direction ? delta > 0 : delta < 0) ? ".good" : ".bad";
  return h(`span.delta${state}`, null, fmtSigned(delta));
}

function deltaNote(value, reference, direction, text) {
  if (text === null || value === null || reference === null) return null;
  const delta = value - reference;
  return h("span.hero-note", null, signed(delta, direction), ` ${text}`);
}
