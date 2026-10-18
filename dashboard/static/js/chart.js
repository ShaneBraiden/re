/* Line chart: one axis, hairline grid, 2px lines with ringed markers, a
 * crosshair tooltip that answers "what are all the values at this x", a legend
 * that doubles as an emphasis control, selective endpoint labels, and a table
 * twin so no value is reachable only by hovering.
 *
 * series: [{ name, label, color, dash?, points: [{ x, y, lo?, hi?, n? }] }]
 * color is a CSS custom property name ("--series-1"); every colour is applied
 * through style so a theme switch repaints without a redraw.
 */

import { h, s, clear, fmt, segmented } from "./ui.js";

function niceStep(span, count) {
  const raw = span / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  return [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((step) => step >= raw) ?? magnitude * 10;
}

export function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const step = niceStep(hi - lo, count);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

const finiteNum = (v) => typeof v === "number" && Number.isFinite(v);

export function lineChart(options) {
  const {
    series,
    height = 300,
    xFormat = (x) => fmt(x),
    yFormat = (y) => fmt(y),
    xTitle = "",
    yTitle = "",
    refLines = [],
    legend = true,
    labels = "auto",
    onPick = null,
    ariaLabel = yTitle,
    emptyText = "No finite values to plot.",
    tooltipNote = null,
  } = options;
  let marker = options.marker ?? null;

  const root = h("div.chart");
  const legendEl = legend && series.length >= 2 ? h("div.legend") : null;
  const plot = h("div.chart-plot");
  const svg = s("svg", { class: "chart-svg", role: "img", "aria-label": ariaLabel });
  const tip = h("div.tooltip", { hidden: true });
  plot.append(svg, tip);
  const tableWrap = h("div.chart-table", { hidden: true });
  if (legendEl) root.append(legendEl);
  root.append(plot, tableWrap);

  let emphasis = null;
  let cursor = null;
  const xs = [...new Set(series.flatMap((ser) => ser.points.map((p) => p.x)))].sort((a, b) => a - b);

  if (legendEl) {
    for (const ser of series) {
      const item = h("span.legend-item", { tabindex: 0, title: ser.name },
        legendSwatch(ser), h("span", null, ser.label));
      const on = () => { emphasis = ser.name; draw(); };
      const off = () => { emphasis = null; draw(); };
      item.addEventListener("pointerenter", on);
      item.addEventListener("pointerleave", off);
      item.addEventListener("focus", on);
      item.addEventListener("blur", off);
      legendEl.append(item);
    }
  }

  let geometry = null;

  function draw() {
    const width = plot.clientWidth;
    if (!width) return;
    clear(svg);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", String(height));

    // The axis fits the lines, not their intervals: a wide interval on one arm
    // would otherwise flatten every line, and the axis would jump on hover.
    // A lone series has its band always on, so there the band counts too.
    const ys = [];
    for (const ser of series) {
      for (const p of ser.points) {
        for (const v of series.length === 1 ? [p.y, p.lo, p.hi] : [p.y]) if (finiteNum(v)) ys.push(v);
      }
    }
    if (!ys.length || !xs.length) {
      svg.append(s("text", { x: width / 2, y: height / 2, "text-anchor": "middle", class: "chart-empty" }, emptyText));
      geometry = null;
      return;
    }
    for (const ref of refLines) ys.push(ref.y);

    const labelOn = labels === true || (labels === "auto" && series.length <= 6 && width > 560);
    const longest = Math.max(...series.map((ser) => ser.label.length), 0);
    const M = {
      top: yTitle ? 26 : 12,
      right: labelOn ? Math.min(170, longest * 6.6 + 18) : 20,
      bottom: xTitle ? 46 : 30,
      left: 58,
    };
    const plotW = Math.max(40, width - M.left - M.right);
    const plotH = height - M.top - M.bottom;

    let lo = Math.min(...ys);
    let hi = Math.max(...ys);
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
    lo -= pad; hi += pad;
    const xMin = xs[0];
    const xMax = xs[xs.length - 1];
    const sx = (v) => M.left + (xMax === xMin ? plotW / 2 : ((v - xMin) / (xMax - xMin)) * plotW);
    const sy = (v) => M.top + plotH - ((v - lo) / (hi - lo)) * plotH;
    geometry = { sx, sy, M, plotW, plotH, width };

    if (yTitle) svg.append(s("text", { x: M.left - 50, y: 12, class: "axis-title" }, yTitle));
    if (xTitle) svg.append(s("text", { x: M.left + plotW / 2, y: height - 6, "text-anchor": "middle", class: "axis-title" }, xTitle));

    for (const tick of niceTicks(lo, hi, Math.max(3, Math.round(plotH / 55)))) {
      svg.append(s("line", { x1: M.left, x2: M.left + plotW, y1: sy(tick), y2: sy(tick), class: "grid" }));
      svg.append(s("text", { x: M.left - 8, y: sy(tick) + 3.5, "text-anchor": "end", class: "tick" }, yFormat(tick)));
    }
    svg.append(s("line", { x1: M.left, x2: M.left + plotW, y1: M.top + plotH, y2: M.top + plotH, class: "axis" }));

    const xTicks = xs.length <= 14 ? xs : niceTicks(xMin, xMax, Math.max(3, Math.round(plotW / 90)));
    for (const x of xTicks) {
      svg.append(s("line", { x1: sx(x), x2: sx(x), y1: M.top + plotH, y2: M.top + plotH + 4, class: "axis" }));
      svg.append(s("text", { x: sx(x), y: M.top + plotH + 17, "text-anchor": "middle", class: "tick" }, xFormat(x)));
    }

    for (const ref of refLines) {
      svg.append(s("line", { x1: M.left, x2: M.left + plotW, y1: sy(ref.y), y2: sy(ref.y), class: "ref" }));
      svg.append(s("text", { x: M.left + 6, y: sy(ref.y) - 5, class: "ref-label" }, ref.label));
    }

    if (marker !== null && finiteNum(marker)) {
      svg.append(s("rect", { x: sx(marker) - 10, y: M.top, width: 20, height: plotH, class: "marker-band" }));
      svg.append(s("line", { x1: sx(marker), x2: sx(marker), y1: M.top, y2: M.top + plotH, class: "marker" }));
    }

    // Bands only for the emphasised series (or a lone one): nine overlapping
    // ranges are a fog, one is a reading. Clipped to the plot, which is sized
    // to the lines.
    const clipId = `clip-${Math.random().toString(36).slice(2, 9)}`;
    svg.append(s("clipPath", { id: clipId }, s("rect", { x: M.left, y: M.top, width: plotW, height: plotH })));
    for (const ser of series) {
      const showBand = emphasis === ser.name || series.length === 1;
      if (!showBand) continue;
      const pts = ser.points.filter((p) => finiteNum(p.lo) && finiteNum(p.hi));
      if (pts.length < 2) continue;
      const top = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.hi)}`).join(" ");
      const bottom = [...pts].reverse().map((p) => `L${sx(p.x)},${sy(p.lo)}`).join(" ");
      svg.append(s("path", { d: `${top} ${bottom} Z`, class: "band", "clip-path": `url(#${clipId})`, style: { fill: `var(${ser.color})` } }));
    }

    const ends = [];
    const ordered = emphasis ? [...series.filter((x) => x.name !== emphasis), ...series.filter((x) => x.name === emphasis)] : series;
    for (const ser of ordered) {
      const pts = ser.points.filter((p) => finiteNum(p.y));
      if (!pts.length) continue;
      const dim = emphasis && emphasis !== ser.name;
      const g = s("g", { class: `series${dim ? " dim" : ""}${emphasis === ser.name ? " focus" : ""}` });
      g.append(s("path", {
        d: pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" "),
        class: "line",
        style: { stroke: `var(${ser.color})`, strokeDasharray: ser.dash ?? null },
      }));
      if (xs.length <= 30) {
        for (const p of pts) {
          g.append(s("circle", { cx: sx(p.x), cy: sy(p.y), r: 4, class: "dot", style: { fill: `var(${ser.color})` } }));
        }
      }
      svg.append(g);
      const last = pts[pts.length - 1];
      ends.push({ ser, x: sx(last.x), y: sy(last.y) });
    }

    // Endpoint labels, nudged apart so none overlaps; in ink, not series colour.
    if (labelOn) {
      ends.sort((a, b) => a.y - b.y);
      let prev = -Infinity;
      for (const end of ends) {
        end.ly = Math.max(end.y, prev + 13);
        prev = end.ly;
      }
      const overflow = prev - (M.top + plotH);
      if (overflow > 0) for (const end of ends) end.ly -= overflow;
      for (const end of ends) {
        const dim = emphasis && emphasis !== end.ser.name;
        svg.append(s("text", {
          x: end.x + 9, y: end.ly + 3.5, class: `end-label${dim ? " dim" : ""}`,
        }, end.ser.label));
      }
    }

    const crosshair = s("line", { y1: M.top, y2: M.top + plotH, class: "crosshair", visibility: "hidden" });
    svg.append(crosshair);
    const hit = s("rect", {
      x: M.left - 12, y: M.top, width: plotW + 24, height: plotH,
      class: `hit${onPick ? " pickable" : ""}`, tabindex: 0,
      "aria-label": `${ariaLabel}. Use arrow keys to read values${onPick ? ", Enter to select" : ""}.`,
    });
    svg.append(hit);

    const nearest = (clientX) => {
      const box = svg.getBoundingClientRect();
      const px = ((clientX - box.left) / box.width) * width;
      let best = 0;
      xs.forEach((x, i) => { if (Math.abs(sx(x) - px) < Math.abs(sx(xs[best]) - px)) best = i; });
      return best;
    };

    const show = (i) => {
      cursor = i;
      const x = xs[i];
      crosshair.setAttribute("x1", sx(x));
      crosshair.setAttribute("x2", sx(x));
      crosshair.setAttribute("visibility", "visible");
      renderTip(x);
      const plotBox = plot.getBoundingClientRect();
      const svgBox = svg.getBoundingClientRect();
      const left = (sx(x) / width) * svgBox.width + (svgBox.left - plotBox.left);
      tip.hidden = false;
      const tw = tip.offsetWidth;
      tip.style.left = `${left + 14 + tw > plotBox.width ? Math.max(0, left - 14 - tw) : left + 14}px`;
      tip.style.top = `${M.top}px`;
    };
    const hide = () => {
      crosshair.setAttribute("visibility", "hidden");
      tip.hidden = true;
    };

    hit.addEventListener("pointermove", (e) => show(nearest(e.clientX)));
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("click", (e) => { if (onPick) onPick(xs[nearest(e.clientX)]); });
    hit.addEventListener("focus", () => {
      const start = marker !== null ? xs.indexOf(marker) : -1;
      show(start >= 0 ? start : 0);
    });
    hit.addEventListener("blur", hide);
    hit.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const next = Math.min(xs.length - 1, Math.max(0, (cursor ?? 0) + (e.key === "ArrowRight" ? 1 : -1)));
        show(next);
      } else if ((e.key === "Enter" || e.key === " ") && onPick && cursor !== null) {
        e.preventDefault();
        onPick(xs[cursor]);
      }
    });
  }

  function renderTip(x) {
    const rows = series
      .map((ser) => ({ ser, p: ser.points.find((p) => p.x === x) }))
      .sort((a, b) => (b.p?.y ?? -Infinity) - (a.p?.y ?? -Infinity));
    clear(tip,
      h("div.tt-head", null, xTitle ? `${xTitle} ${xFormat(x)}` : xFormat(x)),
      rows.map(({ ser, p }) => h(`div.tt-row${emphasis === ser.name ? ".focus" : ""}`, null,
        legendSwatch(ser),
        h("span.tt-name", null, ser.label),
        h("span.tt-value", null, p && finiteNum(p.y) ? yFormat(p.y) : "—"))),
      tooltipNote ? h("div.tt-note", null, tooltipNote(x)) : null);
  }

  function renderTable() {
    clear(tableWrap, h("div.table-wrap", null, h("table", null,
      h("thead", null, h("tr", null,
        h("th.text", { scope: "col" }, ""),
        xs.map((x) => h("th.num", { scope: "col" }, xFormat(x))))),
      h("tbody", null, series.map((ser) => h("tr", null,
        h("th.text", { scope: "row" }, h("span.pname", null, legendSwatch(ser), h("span", null, ser.label))),
        xs.map((x) => {
          const p = ser.points.find((q) => q.x === x);
          return h("td.num", null, p && finiteNum(p.y) ? yFormat(p.y) : "—");
        })))))));
  }

  const resize = new ResizeObserver(() => {
    if (!root.isConnected && geometry) { resize.disconnect(); return; }
    draw();
  });
  resize.observe(plot);

  const controls = segmented([["chart", "Chart"], ["table", "Table"]], "chart", (view) => {
    plot.hidden = view !== "chart";
    if (legendEl) legendEl.hidden = view !== "chart";
    tableWrap.hidden = view !== "table";
    if (view === "table") renderTable();
    else draw();
  }, { label: "View as" });

  return {
    el: root,
    controls,
    setMarker(x) { marker = x; draw(); },
  };
}

function legendSwatch(ser) {
  return s("svg", { class: "swatch", width: 20, height: 10, viewBox: "0 0 20 10", "aria-hidden": "true" },
    s("line", {
      x1: 2, y1: 5, x2: 18, y2: 5, "stroke-width": 2.5, "stroke-linecap": "round",
      style: { stroke: `var(${ser.color})`, strokeDasharray: ser.dash ?? null },
    }));
}
