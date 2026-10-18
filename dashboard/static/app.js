/* SensorSim dashboard - the shell: navigation, run switcher, theme, router.
 *
 * No build step and no CDN: the demo has to run offline from cached results,
 * and a script tag pointing at a CDN is a network dependency wearing a
 * convenience costume. Plain ES modules, served as they are.
 *
 * The frontend renders; it never computes a result. Every number it shows was
 * produced by an earlier run and read from the files that run wrote.
 *
 * Routes live in the hash so every view is a link you can share:
 *   #/overview  #/explore?image=nat_00&sev=4&metric=psnr  #/analysis  #/enhance?id=...
 * and ?exp=<run> on any of them picks a run other than the default.
 */

import { api, query } from "./js/api.js";
import { h, clear, icon, segmented, emptyState, errorBox, loading } from "./js/ui.js";
import * as overview from "./js/pages/overview.js";
import * as explorer from "./js/pages/explorer.js";
import * as analysis from "./js/pages/analysis.js";
import * as enhance from "./js/pages/enhance.js";

const PAGES = { overview, explore: explorer, analysis, enhance };
const $ = (id) => document.getElementById(id);

const ctx = {
  health: null,
  experiments: [],
  defaultExp: null,
  exp: null,
  index: null,
  indexError: null,
  page: null,
  params: {},
  upload: false,

  header(title, sub = "", tools = null) {
    $("page-title").textContent = title;
    $("page-sub").textContent = sub;
    clear($("page-tools"), tools);
    document.title = title === "SensorSim" ? title : `${title} · SensorSim`;
  },

  /* Record view state in the URL without re-rendering the page. */
  setParams(patch) {
    for (const [key, value] of Object.entries(patch)) {
      if (value === null || value === undefined || value === "") delete ctx.params[key];
      else ctx.params[key] = String(value);
    }
    history.replaceState(null, "", hashFor(ctx.page, ctx.params));
  },

  go(page, params = {}) {
    const next = { ...params };
    if (ctx.exp && ctx.exp !== ctx.defaultExp && !("exp" in next)) next.exp = ctx.exp;
    location.hash = hashFor(page, next);
  },

  link(page, params = {}) {
    const next = { ...params };
    if (ctx.exp && ctx.exp !== ctx.defaultExp && !("exp" in next)) next.exp = ctx.exp;
    return hashFor(page, next);
  },
};

function hashFor(page, params) {
  const q = query(params);
  return `#/${page}${q ? `?${q}` : ""}`;
}

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, search = ""] = raw.split("?");
  let page = path || (location.pathname.startsWith("/upload") ? "enhance" : "overview");
  if (!PAGES[page]) page = "overview";
  return { page, params: Object.fromEntries(new URLSearchParams(search)) };
}

/* ------------------------------------------------------------- shell --- */

function buildNav() {
  for (const slot of document.querySelectorAll(".nav-icon")) slot.replaceWith(icon(slot.dataset.icon, 18));
}

function syncNav() {
  for (const link of document.querySelectorAll("#nav a")) {
    const page = link.dataset.page;
    link.href = ctx.link(page);
    if (page === ctx.page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  document.querySelector(".brand").href = ctx.link("overview");
}

function buildRunSelect() {
  const select = $("run-select");
  clear(select);
  if (!ctx.experiments.length) {
    select.append(h("option", { value: "" }, "no runs found"));
    select.disabled = true;
    return;
  }
  for (const run of ctx.experiments) {
    const note = run.error ? "no index" : `${run.available} arms · ${run.images} scenes · ${run.conditions} conditions`;
    select.append(h("option", { value: run.id, title: note }, run.error ? `${run.id} (no index)` : `${run.id} · ${run.available} arms`));
  }
  select.value = ctx.exp ?? ctx.defaultExp;
  select.onchange = () => {
    const exp = select.value;
    location.hash = hashFor(ctx.page, exp === ctx.defaultExp ? {} : { exp });
  };
}

function buildTheme() {
  const root = document.documentElement;
  let current = "auto";
  try { current = localStorage.getItem("sensorsim-theme") || "auto"; } catch (_) { /* blocked */ }
  const options = [
    ["auto", h("span.seg-icon", { title: "Follow system" }, icon("auto", 15), h("span.sr", null, "Auto"))],
    ["light", h("span.seg-icon", { title: "Light" }, icon("sun", 15), h("span.sr", null, "Light"))],
    ["dark", h("span.seg-icon", { title: "Dark" }, icon("moon", 15), h("span.sr", null, "Dark"))],
  ];
  $("theme-switch").replaceWith(segmented(options, current, (theme) => {
    if (theme === "auto") delete root.dataset.theme;
    else root.dataset.theme = theme;
    try { localStorage.setItem("sensorsim-theme", theme); } catch (_) { /* blocked */ }
  }, { label: "Theme" }));
}

/* ------------------------------------------------------------ router --- */

let cleanup = null;
let generation = 0;

async function loadIndex(exp) {
  ctx.exp = exp;
  ctx.index = null;
  ctx.indexError = null;
  if (!exp) {
    ctx.indexError = ctx.health?.note || "no experiment run found";
    return;
  }
  try {
    ctx.index = await api.index(exp);
  } catch (err) {
    ctx.indexError = String(err.message || err);
  }
  const provenance = ctx.index ? `${ctx.index.experiment_id} · from ${ctx.index.generated_from}` : "";
  $("footer-provenance").textContent = provenance;
}

async function route() {
  const mine = ++generation;
  const { page, params } = parseHash();
  ctx.page = page;
  ctx.params = params;
  const exp = params.exp || ctx.defaultExp;
  if (exp !== ctx.exp || (!ctx.index && !ctx.indexError)) await loadIndex(exp);
  if (mine !== generation) return;

  syncNav();
  $("run-select").value = ctx.exp ?? "";

  if (typeof cleanup === "function") cleanup();
  cleanup = null;
  const main = $("page");
  clear(main, loading());

  const module = PAGES[page];
  if (module.needsIndex !== false && !ctx.index) {
    ctx.header(module.title ?? "SensorSim", "");
    clear(main, emptyState(
      "No finished run to show yet",
      ctx.indexError,
      "python -m sensorsim.cli run configs/experiments/pilot.yaml"));
    return;
  }

  try {
    cleanup = await module.render(main, ctx, () => mine !== generation);
  } catch (err) {
    if (mine !== generation) return;
    console.error(err);
    clear(main, errorBox(String(err.message || err)));
  }
}

async function boot() {
  buildNav();
  buildTheme();
  const [health, listing] = await Promise.all([
    api.health(),
    api.experiments().catch(() => ({ experiments: [], default: null })),
  ]);
  ctx.health = health;
  ctx.upload = Boolean(health.upload);
  ctx.experiments = listing.experiments.filter((run) => !run.error || run.default);
  ctx.defaultExp = listing.default ?? health.default ?? null;
  if (ctx.defaultExp && !ctx.experiments.some((run) => run.id === ctx.defaultExp)) ctx.defaultExp = ctx.experiments[0]?.id ?? null;
  $("upload-badge").hidden = !ctx.upload;
  buildRunSelect();

  window.addEventListener("hashchange", route);
  await route();
}

boot();
