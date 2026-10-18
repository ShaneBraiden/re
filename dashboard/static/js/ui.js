/* Shared building blocks: a tiny element builder, number formatting, and the
 * pipeline/metric vocabulary every page draws from.
 *
 * Untrusted text (file names, reasons, anything from a run) only ever reaches
 * the page as text nodes - h() never sets innerHTML.
 */

export function h(tag, props, ...children) {
  const [name, ...classes] = tag.split(".");
  const el = document.createElement(name || "div");
  if (classes.length) el.className = classes.join(" ");
  applyProps(el, props);
  append(el, children);
  return el;
}

const SVG_NS = "http://www.w3.org/2000/svg";

export function s(tag, props, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  applyProps(el, props, true);
  append(el, children);
  return el;
}

function applyProps(el, props, isSvg = false) {
  if (!props) return;
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.setAttribute("class", `${el.getAttribute("class") || ""} ${value}`.trim());
    else if (key === "style" && typeof value === "object") {
      for (const [prop, v] of Object.entries(value)) {
        if (v !== null && v !== undefined) el.style.setProperty(prop.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`), v);
      }
    } else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (!isSvg && (key === "hidden" || key === "disabled" || key === "checked" || key === "selected")) el[key] = Boolean(value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
}

export function clear(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

/* ------------------------------------------------------------ numbers --- */

export function fmt(value, digits = null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (!Number.isFinite(value)) return value > 0 ? "∞" : "−∞";
  const abs = Math.abs(value);
  let text;
  if (digits !== null) text = value.toFixed(digits);
  else if (abs >= 1000) text = value.toFixed(0);
  else if (abs >= 100) text = value.toFixed(1);
  else if (abs >= 1) text = value.toFixed(2);
  else if (abs === 0) text = "0";
  else text = value.toFixed(3);
  return text.replace(/^-/, "−");
}

export function fmtP(p) {
  if (p === null || p === undefined) return "—";
  if (p < 0.001) return "< 0.001";
  return p.toFixed(3);
}

export function fmtSigned(value, digits = null) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : "−"}${fmt(Math.abs(value), digits)}`;
}

/* ------------------------------------------------------------ metrics --- */

const METRICS = {
  psnr: { label: "PSNR", unit: "dB", digits: 2 },
  ssim: { label: "SSIM", digits: 3 },
  delta_e: { label: "ΔE", digits: 2 },
  sharpness: { label: "Sharpness", digits: 1 },
  noise_sigma: { label: "Noise σ", digits: 3 },
  colorfulness: { label: "Colorfulness", digits: 1 },
  nr_quality: { label: "NR quality", digits: 2 },
  lpips: { label: "LPIPS", digits: 3 },
  niqe: { label: "NIQE", digits: 2 },
  clip_iqa: { label: "CLIP-IQA", digits: 3 },
  sift_keypoints: { label: "SIFT keypoints", digits: 0 },
  sift_matches: { label: "SIFT matches", digits: 0 },
  sift_inliers: { label: "SIFT inliers", digits: 0 },
  sift_inlier_ratio: { label: "SIFT inlier ratio", digits: 3 },
  sift_reprojection_error: { label: "SIFT reproj. error", unit: "px", digits: 3 },
  mtf50_edge: { label: "MTF50", unit: "cy/px", digits: 3 },
  edge_acutance: { label: "Edge acutance", digits: 3 },
  ocr_accuracy: { label: "OCR accuracy", digits: 3 },
  ocr_cer: { label: "OCR CER", digits: 3 },
  detection_map: { label: "Detection mAP", digits: 3 },
  consistency_psnr: { label: "Consistency PSNR", unit: "dB", digits: 2 },
  consistency_ssim: { label: "Consistency SSIM", digits: 4 },
};

export const FAMILY_LABELS = { fidelity: "Fidelity", perceptual: "Perceptual", machine_task: "Machine vision", other: "Other" };

export function metricLabel(metric) {
  return METRICS[metric]?.label ?? metric;
}

export function metricUnit(metric) {
  return METRICS[metric]?.unit ?? "";
}

export function fmtMetric(metric, value) {
  const digits = METRICS[metric]?.digits ?? null;
  if (digits === 0 && value !== null && value !== undefined && !Number.isInteger(value)) return fmt(value, 1);
  return fmt(value, digits);
}

/* Direction semantics from dashboard.json: true = higher is better, false =
 * lower is better, null = closest to the target (1.0) is better. */
export function directionNote(direction) {
  if (direction === null) return "closest to 1.0 is best";
  return direction ? "higher is better" : "lower is better";
}

export function score(direction, value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  if (direction === null) return -Math.abs(value - 1);
  return direction === false ? -value : value;
}

/* The index of the best entry, or -1. Ties keep the first. */
export function bestIndex(values, direction) {
  let best = -1;
  let bestScore = -Infinity;
  values.forEach((value, i) => {
    const sc = score(direction, value);
    if (sc !== null && sc > bestScore) { bestScore = sc; best = i; }
  });
  return best;
}

export function isBetter(direction, value, reference) {
  const a = score(direction, value);
  const b = score(direction, reference);
  if (a === null || b === null) return null;
  return a >= b;
}

/* ---------------------------------------------------------- pipelines --- */

/* Colour follows the pipeline, fixed here by name - never by its position in
 * whatever subset is switched on, so filtering never repaints the survivors.
 *
 * The bank has more arms than the palette has validated hues, so identity is
 * composite: an ablation wears its parent's hue with a dash pattern, and the
 * baseline is a neutral reference line rather than a competing series. The
 * pairs the study actually compares (NAG vs each component and each
 * classical arm, NC-DCE vs Zero-DCE, G-ESRGAN vs Real-ESRGAN) sit on hues that
 * clear the CVD check in both themes. Legends, endpoint labels, tooltips and
 * the table view carry identity too, so it is never colour alone. */
const PIPELINES = {
  classical_isp: { label: "Classical ISP", group: "classical", color: "--baseline", baseline: true },
  nlm_denoise: { label: "NLM denoise", group: "classical", color: "--series-2" },
  curve_tone: { label: "Curve tone", group: "classical", color: "--series-3" },
  rl_deconv: { label: "RL deconvolution", group: "classical", color: "--series-7" },
  zero_dce: { label: "Zero-DCE", group: "learned", color: "--series-4" },
  real_esrgan: { label: "Real-ESRGAN", group: "learned", color: "--series-6" },
  nafnet: { label: "NAFNet", group: "learned", color: "--series-6", dash: "7 4" },
  restormer: { label: "Restormer", group: "learned", color: "--series-2", dash: "7 4" },
  learned_isp: { label: "Learned ISP", group: "learned", color: "--series-3", dash: "7 4" },
  zero_dce_ft: { label: "Zero-DCE fine-tuned", group: "learned", color: "--series-4", dash: "7 4" },
  nc_dce: { label: "NC-DCE", group: "nag", color: "--series-8" },
  g_esrgan: { label: "G-ESRGAN", group: "nag", color: "--series-5" },
  nag: { label: "NAG", group: "nag", color: "--series-1", proposed: true },
  nag_nodenoise: { label: "NAG − VP-denoise", group: "ablation", color: "--series-1", dash: "7 4" },
  nag_nogate: { label: "NAG − gate", group: "ablation", color: "--series-1", dash: "2 3" },
  nag_nobp: { label: "NAG − back-projection", group: "ablation", color: "--series-1", dash: "9 3 2 3" },
  nc_dce_nosigma: { label: "NC-DCE − σ input", group: "ablation", color: "--series-8", dash: "7 4" },
  nc_dce_nonoise: { label: "NC-DCE − noise loss", group: "ablation", color: "--series-8", dash: "2 3" },
};

export const GROUPS = [
  { id: "classical", label: "Classical" },
  { id: "learned", label: "Learned" },
  { id: "nag", label: "NAG · proposed" },
  { id: "ablation", label: "Ablations" },
  { id: "other", label: "Other" },
];

/* The arms the paper's main table compares; the default selection when a run
 * has too many to read at once. */
export const MAIN_ARMS = ["classical_isp", "nlm_denoise", "curve_tone", "rl_deconv",
  "zero_dce", "real_esrgan", "nc_dce", "g_esrgan", "nag"];

const unknown = [];

export function pipelineInfo(name) {
  if (PIPELINES[name]) return { name, ...PIPELINES[name] };
  if (!unknown.includes(name)) unknown.push(name);
  const i = unknown.indexOf(name);
  return { name, label: name, group: "other", color: `--series-${(i % 8) + 1}`, dash: "1 3" };
}

export const pipelineLabel = (name) => pipelineInfo(name).label;

export function lineStyle(name) {
  const info = pipelineInfo(name);
  return { stroke: `var(${info.color})`, strokeDasharray: info.dash ?? null };
}

/* A short line sample: hue + dash, so the legend shows exactly what the chart draws. */
export function swatch(name, { off = false } = {}) {
  const info = pipelineInfo(name);
  return s("svg", { class: `swatch${off ? " off" : ""}`, width: 20, height: 10, viewBox: "0 0 20 10", "aria-hidden": "true" },
    s("line", {
      x1: 2, y1: 5, x2: 18, y2: 5, "stroke-width": 2.5, "stroke-linecap": "round",
      style: { stroke: off ? "var(--axis)" : `var(${info.color})`, strokeDasharray: info.dash ?? null },
    }));
}

export function pipelineName(name, { id = false } = {}) {
  const info = pipelineInfo(name);
  return h("span.pname", { title: name },
    swatch(name),
    h("span", null, info.label),
    info.proposed ? h("span.tag.accent", null, "ours") : null,
    info.baseline ? h("span.tag", null, "baseline") : null,
    id && info.label !== name ? h("code.pid", null, name) : null);
}

/* ------------------------------------------------------------ widgets --- */

const ICONS = {
  overview: "M3 13h8V3H3zm0 8h8v-6H3zm10 0h8V11h-8zm0-18v6h8V3z",
  explore: "M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14",
  analysis: "M5 9.2h3V19H5zM10.6 5h2.8v14h-2.8zm5.6 8H19v6h-2.8z",
  enhance: "M19 9l1.25-2.75L23 5l-2.75-1.25L19 1l-1.25 2.75L15 5l2.75 1.25zm-7.5.5L9 4 6.5 9.5 1 12l5.5 2.5L9 20l2.5-5.5L17 12zM19 15l-1.25 2.75L15 19l2.75 1.25L19 23l1.25-2.75L23 19l-2.75-1.25z",
  check: "M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z",
  cross: "M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z",
  warn: "M1 21h22L12 2zm12-3h-2v-2h2zm0-4h-2v-4h2z",
  info: "M11 7h2v2h-2zm0 4h2v6h-2zm1-9C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2m0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8",
  sun: "M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10M2 13h2a1 1 0 0 0 0-2H2a1 1 0 0 0 0 2m18 0h2a1 1 0 0 0 0-2h-2a1 1 0 0 0 0 2M11 2v2a1 1 0 0 0 2 0V2a1 1 0 0 0-2 0m0 18v2a1 1 0 0 0 2 0v-2a1 1 0 0 0-2 0M5.99 4.58a1 1 0 0 0-1.41 1.41l1.06 1.06a1 1 0 0 0 1.41-1.41zm12.37 12.37a1 1 0 0 0-1.41 1.41l1.06 1.06a1 1 0 0 0 1.41-1.41zm1.06-10.96a1 1 0 0 0-1.41-1.41l-1.06 1.06a1 1 0 0 0 1.41 1.41zM7.05 18.36a1 1 0 0 0-1.41-1.41l-1.06 1.06a1 1 0 0 0 1.41 1.41z",
  moon: "M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.39 5.39 0 0 1-4.4 2.26 5.4 5.4 0 0 1-3.14-9.8c-.44-.06-.9-.1-1.36-.1",
  auto: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20m0 18V4a8 8 0 0 1 0 16",
  upload: "M5 20h14v-2H5zm0-10h4v6h6v-6h4l-7-7z",
  expand: "M7 14H5v5h5v-2H7zm-2-4h2V7h3V5H5zm12 7h-3v2h5v-5h-2zM14 5v2h3v3h2V5z",
  reset: "M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8",
  image: "M21 19V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2M8.5 13.5l2.5 3.01L14.5 12l4.5 6H5z",
  file: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zm4 18H6V4h7v5h5z",
  close: "M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z",
  back: "M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20z",
};

export function icon(name, size = 18) {
  return s("svg", { class: "icon", width: size, height: size, viewBox: "0 0 24 24", "aria-hidden": "true" },
    s("path", { d: ICONS[name] ?? ICONS.info, fill: "currentColor" }));
}

/* Status always ships with an icon and a word, never colour alone. */
export function badge(state, text) {
  const iconName = { good: "check", bad: "cross", warn: "warn", neutral: "info" }[state] ?? "info";
  return h(`span.badge.${state}`, null, icon(iconName, 14), h("span", null, text));
}

export function card(title, { sub = null, tools = null, cls = "" } = {}, ...body) {
  return h(`section.card${cls ? "." + cls.split(" ").join(".") : ""}`, null,
    title || tools ? h("header.card-head", null,
      h("div.card-titles", null,
        title ? h("h2", null, title) : null,
        sub ? h("p.card-sub", null, sub) : null),
      tools ? h("div.card-tools", null, tools) : null) : null,
    body);
}

export function emptyState(title, detail = null, command = null) {
  return h("div.empty", null,
    icon("info", 22),
    h("p.empty-title", null, title),
    detail ? h("p.empty-detail", null, detail) : null,
    command ? h("code.cmd", null, command) : null);
}

export function stat(label, value, { note = null, noteState = null, unit = null } = {}) {
  return h("div.stat", null,
    h("span.stat-label", null, label),
    h("span.stat-value", null, value, unit ? h("span.stat-unit", null, unit) : null),
    note ? h(`span.stat-note${noteState ? "." + noteState : ""}`, null, note) : null);
}

export function segmented(options, value, onChange, { label = null } = {}) {
  const group = h("div.segmented", { role: "group", "aria-label": label });
  const buttons = options.map(([key, text]) => {
    const button = h("button", { type: "button", "aria-pressed": String(key === value) }, text);
    button.addEventListener("click", () => {
      for (const b of buttons) b.setAttribute("aria-pressed", String(b === button));
      onChange(key);
    });
    return button;
  });
  group.append(...buttons);
  return group;
}

export function select(options, value, onChange, { label = null, id = null } = {}) {
  const el = h("select", { id, "aria-label": label });
  for (const option of options) {
    if (option.group) {
      const group = h("optgroup", { label: option.group });
      for (const [key, text] of option.items) group.append(h("option", { value: key }, text));
      el.append(group);
    } else {
      const [key, text] = option;
      el.append(h("option", { value: key }, text));
    }
  }
  el.value = value;
  el.addEventListener("change", () => onChange(el.value));
  return el;
}

/* A div, not a <label>: a label wrapping a group of buttons would press the
 * first one whenever its caption is clicked. Controls carry aria-label. */
export function field(label, control, { grow = false, hint = null } = {}) {
  return h(`div.field${grow ? ".grow" : ""}`, null,
    h("span.field-label", null, label), control, hint ? h("span.hint", null, hint) : null);
}

/* A plain data table. columns: [{key, label, render?, align?, best?: direction}] */
export function table(columns, rows, { onRowClick = null, activeRow = null, caption = null } = {}) {
  const bests = new Map();
  for (const column of columns) {
    if (column.best === undefined) continue;
    const values = rows.map((row) => (column.value ? column.value(row) : row[column.key]));
    bests.set(column.key, bestIndex(values, column.best));
  }
  return h("div.table-wrap", null,
    h("table", null,
      caption ? h("caption", null, caption) : null,
      h("thead", null, h("tr", null, columns.map((c) => h(`th.${c.align ?? "num"}`, { scope: "col", title: c.title ?? null }, c.label)))),
      h("tbody", null, rows.map((row, i) => {
        const tr = h("tr", {
          class: [onRowClick ? "clickable" : "", activeRow !== null && activeRow(row) ? "active" : ""].join(" ").trim() || null,
          tabindex: onRowClick ? 0 : null,
        }, columns.map((c) => {
          const isBest = bests.get(c.key) === i;
          const content = c.render ? c.render(row) : fmt(row[c.key]);
          return h(`td.${c.align ?? "num"}${isBest ? ".best" : ""}`, { title: isBest ? "best in column" : null }, content);
        }));
        if (onRowClick) {
          tr.addEventListener("click", () => onRowClick(row));
          tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onRowClick(row); } });
        }
        return tr;
      }))));
}

export function errorBox(message) {
  return h("div.banner.error", { role: "alert" }, icon("warn", 16), h("span", null, message));
}

export function loading(text = "Loading…") {
  return h("div.loading", { role: "status" }, h("span.spinner", { "aria-hidden": "true" }), h("span", null, text));
}

/* Lightbox for figures and full-size images. Uses <dialog>, which brings its
 * own focus trap and Escape handling. */
export function openLightbox(src, caption) {
  const dialog = h("dialog.lightbox", { "aria-label": caption },
    h("form.lightbox-bar", { method: "dialog" },
      h("span.lightbox-caption", null, caption),
      h("a.btn.ghost", { href: src, target: "_blank", rel: "noopener" }, icon("expand", 16), "Open"),
      h("button.btn.ghost", { value: "close", "aria-label": "Close" }, icon("close", 16))),
    h("img", { src, alt: caption }));
  dialog.addEventListener("close", () => dialog.remove());
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  document.body.append(dialog);
  dialog.showModal();
}

/* Metric picker options, grouped by family in the study's order. */
export function metricOptions(index) {
  const order = ["fidelity", "perceptual", "machine_task"];
  const groups = new Map();
  for (const metric of index.metrics) {
    const family = index.families?.[metric] ?? "other";
    if (!groups.has(family)) groups.set(family, []);
    groups.get(family).push([metric, metricLabel(metric)]);
  }
  return [...groups]
    .sort(([a], [b]) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99))
    .map(([family, items]) => ({ group: FAMILY_LABELS[family] ?? family, items }));
}

export function defaultMetric(index, wanted) {
  if (wanted && index.metrics.includes(wanted)) return wanted;
  return index.metrics.includes("psnr") ? "psnr" : index.metrics[0];
}

export function sortedConditions(index) {
  return [...index.conditions].sort((a, b) => a.severity - b.severity);
}

/* What one degradation condition means in hardware terms. */
export function describeCondition(condition) {
  const p = condition.params ?? {};
  const parts = [];
  if (p.alpha !== undefined) parts.push(`${fmt(p.alpha * 100, p.alpha < 0.1 ? 1 : 0)}% light`);
  if (p.sigma_r_dn !== undefined) parts.push(`read noise ${fmt(p.sigma_r_dn, 1)} DN`);
  if (p.r_psf !== undefined) parts.push(`blur ${fmt(p.r_psf, 2)} px`);
  if (p.binning !== undefined && p.binning > 1) parts.push(`${p.binning}×${p.binning} binning`);
  if (p.bit_depth !== undefined) parts.push(`${p.bit_depth}-bit`);
  return parts.length ? parts.join(" · ") : condition.label;
}
