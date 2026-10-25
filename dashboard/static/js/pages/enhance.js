/* Enhance: run NAG on an uploaded image (when the server was started with
 * --upload), and browse every enhancement run already on disk. The history is
 * read-only and shows without the flag. */

import { api } from "../api.js";
import { lineChart } from "../chart.js";
import { compareViewer } from "../viewer.js";
import {
  h, clear, card, table, select, segmented, fmt, badge, icon, emptyState, errorBox, openLightbox,
} from "../ui.js";

export const title = "Enhance";
export const needsIndex = false;

const OUTPUTS = {
  original: "Original upload",
  input_bicubic: "Bicubic upscale",
  classical: "Classical pipeline",
  nag: "NAG (ours)",
  gate: "Wiener gate map",
  comparison: "Comparison figure",
};

const METRIC_COLUMNS = [
  ["sharpness", "Sharpness", true, 2],
  ["noise_sigma", "Noise σ̂", false, 3],
  ["nr_quality", "NR quality", true, 2],
  ["colorfulness", "Colorfulness", undefined, 1],
  ["consistency_psnr", "Consistency PSNR", true, 2],
  ["consistency_ssim", "Consistency SSIM", true, 4],
];
const METRIC_ROWS = [["input_bicubic", "Bicubic upscale"], ["classical", "Classical pipeline"], ["nag", "NAG (ours)"]];

export async function render(main, ctx, stale) {
  if (ctx.params.id) return renderResult(main, ctx, stale, ctx.params.id);

  ctx.header("Enhance", "Run NAG on your own image — noise-aware curves, VP-denoise and guarded Real-ESRGAN, on this machine");
  const historySlot = h("div");
  clear(main,
    h("div.grid-2.enhance-top", null, ctx.upload ? uploadCard(ctx, () => loadHistory()) : disabledCard(ctx), methodCard()),
    historySlot);

  async function loadHistory() {
    let listing;
    try {
      listing = await api.enhancements();
    } catch (err) {
      clear(historySlot, card("Past runs", {}, errorBox(String(err.message || err))));
      return;
    }
    if (stale()) return;
    clear(historySlot, historyCard(listing.runs, ctx));
  }
  await loadHistory();
}

/* ---------------------------------------------------------------- upload --- */

function uploadCard(ctx, onDone) {
  const input = h("input", { type: "file", accept: ".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff", class: "sr" });
  const preview = h("img.dz-preview", { alt: "", hidden: true });
  const titleEl = h("span.dz-title", null, "Drop an image here, or click to choose");
  const meta = h("span.hint", null, "PNG, JPEG, WebP, BMP or TIFF · up to 20 MB");
  const zone = h("label.dropzone", null, input, preview, icon("upload", 26), titleEl, meta);
  let kind = "auto";
  const submit = h("button.btn.primary", { type: "submit", disabled: true }, icon("enhance", 16), "Enhance image");
  const status = h("div.status", { role: "status", "aria-live": "polite" });
  let file = null;
  let objectUrl = null;

  const pick = (chosen) => {
    if (!chosen) return;
    file = chosen;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(chosen);
    preview.src = objectUrl;
    preview.hidden = false;
    zone.classList.add("has-file");
    titleEl.textContent = chosen.name;
    meta.textContent = `${(chosen.size / 1024 / 1024).toFixed(2)} MB`;
    preview.onload = () => { meta.textContent = `${preview.naturalWidth}×${preview.naturalHeight} px · ${(chosen.size / 1024 / 1024).toFixed(2)} MB`; };
    submit.disabled = chosen.size > 20 * 1024 * 1024;
    clear(status, chosen.size > 20 * 1024 * 1024 ? errorBox("That file is over the 20 MB limit.") : null);
  };

  input.addEventListener("change", () => pick(input.files[0]));
  ["dragenter", "dragover"].forEach((type) => zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((type) => zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));

  const form = h("form.upload-form", null,
    zone,
    h("div.option", null, h("span.field-label", null, "Image type"),
      segmented([["auto", "Detect"], ["photo", "Photo"], ["diagram", "Diagram / screenshot"]], kind, (v) => { kind = v; }, { label: "Image type" })),
    h("div.form-actions", null, submit),
    status);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!file) return;
    const body = new FormData();
    body.append("file", file);
    body.append("kind", kind);

    submit.disabled = true;
    form.classList.add("busy");
    const started = performance.now();
    const elapsed = h("b", null, "0 s");
    clear(status, h("div.progress", null,
      h("span.spinner", { "aria-hidden": "true" }),
      h("span", null, "Enhancing… ", elapsed),
      h("span.hint.block", null, "A large photo takes a minute or two on a laptop GPU. Keep this tab open.")));
    const tick = setInterval(() => { elapsed.textContent = `${Math.round((performance.now() - started) / 1000)} s`; }, 500);
    try {
      const response = await fetch("/api/enhance", { method: "POST", body });
      const payload = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      if (!response.ok) {
        throw new Error(response.status === 413 ? "That file is over the 20 MB limit." : payload.error || `HTTP ${response.status}`);
      }
      clear(status, h("div.progress.done", null, icon("check", 16), `Done in ${fmt(payload.seconds, 1)} s.`));
      await onDone();
      ctx.go("enhance", { id: payload.id });
    } catch (err) {
      clear(status, errorBox(String(err.message || err)));
    } finally {
      clearInterval(tick);
      submit.disabled = false;
      form.classList.remove("busy");
    }
  });

  return card("Upload an image", { sub: "Enhanced on this machine, one image at a time. The result is saved under experiments/enhance/ and nothing else is written." }, form);
}

function disabledCard(ctx) {
  return card("Upload is off", {
    sub: "The dashboard is read-only unless you ask for the one write route.",
  },
  h("p", null, "Restart the dashboard with ", h("code", null, "--upload"), " to enhance images from the browser:"),
  h("code.cmd", null, `python -m sensorsim.cli dashboard experiments/${ctx.defaultExp ?? "pilot"} --upload`),
  h("p", null, "Or run one image from the command line:"),
  h("code.cmd", null, "python -m sensorsim.cli enhance image.png"),
  h("p.hint", null, "Past runs below are on disk already and open either way."));
}

function methodCard() {
  const steps = [
    ["NC-DCE", "Zero-DCE's curve network with a per-pixel noise map as a fourth input, trained so its curves do not amplify noise in flat regions."],
    ["VP-denoise", "A guided filter whose strength at each pixel is the noise the curves left there."],
    ["G-ESRGAN", "Real-ESRGAN's added detail scaled by a Wiener gate, kept only where the measurement holds more than noise, then back-projected so the output stays consistent with the input."],
  ];
  return card("What NAG does", { sub: "Noise-aware guarded enhancement. On uploads the noise map is estimated blindly." },
    h("ol.method", null, steps.map(([name, text], i) => h("li", null,
      h("span.method-num", null, String(i + 1)),
      h("div", null, h("b", null, name), h("p", null, text))))));
}

function historyCard(runs, ctx) {
  if (!runs.length) {
    return card("Past runs", {}, emptyState("No enhancement runs yet", "Upload an image above, or run one from the command line.", "python -m sensorsim.cli enhance image.png"));
  }
  return card(`Past runs (${runs.length})`, { sub: "Every enhancement on disk, newest first. Open one to compare it with the baselines." },
    h("div.runs", null, runs.map((run) => {
      const [ih, iw] = run.input_shape ?? [];
      const [oh, ow] = run.output_shape ?? [];
      const thumb = run.urls.nag ?? run.urls.original;
      const link = h("a.run", { href: ctx.link("enhance", { id: run.id }) },
        h("div.run-thumb", null, thumb ? h("img", { src: thumb, alt: "", loading: "lazy" }) : icon("image", 28)),
        h("div.run-body", null,
          h("span.run-name", { title: run.name }, run.name ?? run.id),
          h("span.run-meta", null, badge("neutral", run.kind ?? "image"),
            iw ? h("span", null, `${iw}×${ih} → ${ow}×${oh}`) : null),
          h("span.hint", null, `${new Date(run.modified * 1000).toLocaleString()} · ${fmt(run.seconds, 1)} s`)));
      return link;
    })));
}

/* ---------------------------------------------------------------- result --- */

async function renderResult(main, ctx, stale, id) {
  const back = h("a.btn.ghost", { href: ctx.link("enhance") }, icon("back", 16), "All runs");
  ctx.header("Enhancement", "", back);
  const listing = await api.enhancements();
  if (stale()) return;
  const run = listing.runs.find((r) => r.id === id);
  if (!run) {
    clear(main, emptyState("No enhancement run with that id", `Nothing in experiments/enhance is called “${id}”.`));
    return;
  }
  const [ih, iw] = run.input_shape ?? [];
  const [oh, ow] = run.output_shape ?? [];
  ctx.header(run.name ?? run.id,
    `${run.kind ?? "image"} · ${iw}×${ih} → ${ow}×${oh} px (×${fmt(run.scale, 1)}) · ${fmt(run.seconds, 1)} s`, back);

  const viewer = compareViewer({ aspect: ow && oh ? `${ow} / ${oh}` : "4 / 3", pixelated: "zoom", maxZoom: 24 });
  const leftChoices = ["original", "input_bicubic", "classical", "gate"].filter((k) => run.urls[k]);
  const rightChoices = ["nag", "classical", "input_bicubic"].filter((k) => run.urls[k]);
  let left = run.urls.input_bicubic ? "input_bicubic" : leftChoices[0];
  let right = rightChoices[0];
  const update = () => viewer.set({ left: run.urls[left], right: run.urls[right], leftLabel: OUTPUTS[left], rightLabel: OUTPUTS[right] });
  const tools = h("div.side-selects", null,
    select(leftChoices.map((k) => [k, OUTPUTS[k]]), left, (v) => { left = v; update(); }, { label: "Left image" }),
    h("span.muted", null, "vs"),
    select(rightChoices.map((k) => [k, OUTPUTS[k]]), right, (v) => { right = v; update(); }, { label: "Right image" }));
  update();

  const metricsRows = METRIC_ROWS.filter(([key]) => run.metrics?.[key]).map(([key, label]) => ({ key, label, ...run.metrics[key] }));
  const metricsCard = card("Metrics", {
    sub: "No ground truth exists for an upload, so these are no-reference proxies plus a consistency check.",
  }, table([
    { key: "label", label: "Output", align: "text", render: (r) => h("span", null, r.label, r.key === "nag" ? h("span.tag.accent", null, "ours") : null) },
    ...METRIC_COLUMNS.filter(([key]) => metricsRows.some((r) => key in r)).map(([key, label, direction, digits]) => ({
      key, label, best: direction, render: (r) => fmt(r[key], digits),
      title: direction === undefined ? "no better direction" : direction ? "higher is better" : "lower is better",
    })),
  ], metricsRows),
  h("p.card-foot", null, "Consistency: the output is downscaled to the input size, tone-matched per channel and compared with the upload, so invented detail lowers it. Bicubic scores highest by construction."));

  const features = run.features ?? {};
  const classifier = card("Classifier", { sub: `Treated as a ${run.kind}` },
    h("dl.kv", null,
      "palette_top8" in features ? [h("dt", null, "Top-8 colour share"), h("dd", null, fmt(features.palette_top8, 2))] : null,
      "flat_share" in features ? [h("dt", null, "Flat area share"), h("dd", null, fmt(features.flat_share, 2))] : null,
      "edge_density" in features ? [h("dt", null, "Edge density"), h("dd", null, fmt(features.edge_density, 3))] : null,
      Object.entries(run.checkpoints ?? {}).map(([name, sha]) => [h("dt", null, name), h("dd", null, h("code", { title: sha }, `${sha.slice(0, 12)}…`))])));

  const steps = card("What was done", {}, h("ol.timeline", null, (run.steps ?? []).map((step) =>
    h("li", null, h("b", null, step.step), h("span", null, step.detail)))));

  const nlf = run.noise_level_function;
  let noiseCard = null;
  if (nlf?.centres?.length && nlf.sigma?.length) {
    const channels = nlf.sigma.length === 3 ? [["R", "--series-8"], ["G", "--series-6"], ["B", "--series-1"]] : nlf.sigma.map((_, i) => [`ch ${i}`, `--series-${i + 1}`]);
    const chart = lineChart({
      series: channels.map(([label, color], i) => ({
        name: label, label, color,
        points: nlf.centres.map((x, j) => ({ x, y: nlf.sigma[i][j] === null ? null : nlf.sigma[i][j] * 255 })),
      })),
      height: 240,
      xTitle: "Intensity",
      yTitle: "Noise σ (8-bit levels)",
      xFormat: (x) => fmt(x, 2),
      yFormat: (y) => fmt(y, 2),
      labels: false,
      ariaLabel: "Estimated noise level per intensity",
    });
    noiseCard = card("Noise map", {
      sub: `${nlf.estimated ? "Estimated blindly from the upload" : "From the capture record"} — this is the per-pixel σ every NAG stage is steered by`,
      tools: chart.controls,
      cls: "chart-card",
    }, chart.el);
  }

  const files = card("Files", { sub: `experiments/enhance/${run.id}/` },
    h("ul.files", null, Object.entries(run.urls).map(([key, url]) => {
      const name = url.split("/").pop();
      const open = h("button.link-btn", { type: "button" }, OUTPUTS[key] ?? key);
      open.addEventListener("click", () => openLightbox(url, `${OUTPUTS[key] ?? key} — ${name}`));
      return h("li", null, icon("image", 16), open, h("a.hint", { href: url, target: "_blank", rel: "noopener" }, name));
    })));

  clear(main,
    h("div.result-grid", null,
      card("Before and after", {
        sub: "Drag the divider to wipe · scroll or double-click to zoom into detail",
        tools,
        cls: "viewer-card",
      }, viewer.el),
      h("div.stack", null, steps, classifier)),
    metricsCard,
    noiseCard ? h("div.grid-2", null, noiseCard, files) : files);
}
