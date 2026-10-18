/* Synchronised wipe comparison: two images under one transform, so the panes
 * can never drift apart - that is the whole point of the comparison.
 *
 * Drag the divider (or focus it and use the arrow keys) to wipe, drag the
 * image to pan, scroll or use the buttons to zoom, double-click to jump
 * between fit and 4x at the pointer. */

import { h, icon } from "./ui.js";

export function compareViewer({ aspect = "1 / 1", maxZoom = 16, pixelated = true } = {}) {
  // pixelated: true always, "zoom" only when magnified, false never.
  const state = { scale: 1, x: 0, y: 0, wipe: 0.5 };

  const imgLeft = h("img", { alt: "", draggable: "false" });
  const imgRight = h("img", { alt: "", draggable: "false" });
  const layerLeft = h("div.layer", null, imgLeft);
  const layerRight = h("div.layer", null, imgRight);
  const tagLeft = h("div.corner-tag.left");
  const tagRight = h("div.corner-tag.right");
  const missingLeft = h("div.missing.left", { hidden: true }, "no preview");
  const missingRight = h("div.missing.right", { hidden: true }, "no preview");
  const spinner = h("div.viewer-loading", { hidden: true }, h("span.spinner", { "aria-hidden": "true" }));
  const divider = h("div.divider", {
    role: "slider", tabindex: 0, "aria-label": "Comparison divider",
    "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": 50,
  }, h("span.grip", { "aria-hidden": "true" }, "‹ ›"));
  const zoomText = h("span.zoom-level", null, "1×");

  const zoomBy = (factor, cx = null, cy = null) => {
    const box = stage.getBoundingClientRect();
    const px = cx ?? box.width / 2;
    const py = cy ?? box.height / 2;
    const next = Math.min(maxZoom, Math.max(1, state.scale * factor));
    const ratio = next / state.scale;
    // Zoom about the pointer: the pixel under the cursor stays under it.
    state.x = px - (px - state.x) * ratio;
    state.y = py - (py - state.y) * ratio;
    state.scale = next;
    apply();
  };

  const tools = h("div.viewer-tools", null,
    h("button.icon-btn", { type: "button", title: "Zoom out", "aria-label": "Zoom out", onclick: () => zoomBy(1 / 1.5) }, "−"),
    zoomText,
    h("button.icon-btn", { type: "button", title: "Zoom in", "aria-label": "Zoom in", onclick: () => zoomBy(1.5) }, "+"),
    h("button.icon-btn", { type: "button", title: "Reset view", "aria-label": "Reset view", onclick: () => reset() }, icon("reset", 15)));

  const stage = h("div.viewer", {
    tabindex: -1, style: { aspectRatio: aspect },
  }, layerLeft, layerRight, missingLeft, missingRight, divider, tagLeft, tagRight, spinner, tools);

  function clampPan() {
    const box = stage.getBoundingClientRect();
    if (!box.width) return;
    state.x = Math.min(0, Math.max(box.width * (1 - state.scale), state.x));
    state.y = Math.min(0, Math.max(box.height * (1 - state.scale), state.y));
  }

  function apply() {
    if (state.scale <= 1.0001) { state.scale = 1; state.x = 0; state.y = 0; }
    clampPan();
    const transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
    imgLeft.style.transform = transform;
    imgRight.style.transform = transform;
    layerRight.style.clipPath = `inset(0 0 0 ${state.wipe * 100}%)`;
    divider.style.left = `${state.wipe * 100}%`;
    divider.setAttribute("aria-valuenow", String(Math.round(state.wipe * 100)));
    // Nearest-neighbour so a 1:1 pixel comparison is honest; for photos larger
    // than the box ("zoom") only once magnified, so the fitted view is not aliased.
    stage.classList.toggle("pixelated", pixelated === true || (pixelated === "zoom" && state.scale >= 3));
    zoomText.textContent = `${state.scale < 10 ? state.scale.toFixed(1).replace(/\.0$/, "") : Math.round(state.scale)}×`;
  }

  function reset() {
    state.scale = 1; state.x = 0; state.y = 0; state.wipe = 0.5;
    apply();
  }

  let drag = null;
  divider.addEventListener("pointerdown", (e) => {
    drag = { mode: "wipe" };
    divider.setPointerCapture(e.pointerId);
    e.stopPropagation();
  });
  stage.addEventListener("pointerdown", (e) => {
    if (drag || e.button !== 0 || e.target.closest(".viewer-tools")) return;
    drag = { mode: "pan", x: e.clientX, y: e.clientY };
    stage.classList.add("panning");
    stage.setPointerCapture(e.pointerId);
  });
  const move = (e) => {
    if (!drag) return;
    const box = stage.getBoundingClientRect();
    if (drag.mode === "wipe") {
      state.wipe = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
    } else {
      state.x += e.clientX - drag.x;
      state.y += e.clientY - drag.y;
      drag.x = e.clientX;
      drag.y = e.clientY;
    }
    apply();
  };
  const stop = () => { drag = null; stage.classList.remove("panning"); };
  stage.addEventListener("pointermove", move);
  divider.addEventListener("pointermove", move);
  for (const target of [stage, divider]) {
    target.addEventListener("pointerup", stop);
    target.addEventListener("pointercancel", stop);
  }

  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const box = stage.getBoundingClientRect();
    zoomBy(Math.exp(-e.deltaY * 0.0015), e.clientX - box.left, e.clientY - box.top);
  }, { passive: false });

  stage.addEventListener("dblclick", (e) => {
    if (e.target.closest(".viewer-tools") || e.target === divider) return;
    const box = stage.getBoundingClientRect();
    if (state.scale > 1) reset();
    else zoomBy(4, e.clientX - box.left, e.clientY - box.top);
  });

  divider.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 0.1 : 0.02;
    if (e.key === "ArrowLeft") state.wipe = Math.max(0, state.wipe - step);
    else if (e.key === "ArrowRight") state.wipe = Math.min(1, state.wipe + step);
    else if (e.key === "Home") state.wipe = 0;
    else if (e.key === "End") state.wipe = 1;
    else return;
    e.preventDefault();
    apply();
  });

  const watch = (img, missing) => {
    img.addEventListener("load", () => { missing.hidden = true; img.hidden = false; settle(); });
    img.addEventListener("error", () => { missing.hidden = false; img.hidden = true; settle(); });
  };
  const settle = () => {
    if (imgLeft.complete && imgRight.complete) spinner.hidden = true;
  };
  watch(imgLeft, missingLeft);
  watch(imgRight, missingRight);

  apply();

  return {
    el: stage,
    set({ left, right, leftLabel, rightLabel }) {
      let pending = false;
      if (left !== undefined && imgLeft.getAttribute("src") !== left) { imgLeft.src = left; pending = true; }
      if (right !== undefined && imgRight.getAttribute("src") !== right) { imgRight.src = right; pending = true; }
      if (leftLabel !== undefined) { tagLeft.textContent = leftLabel; imgLeft.alt = leftLabel; }
      if (rightLabel !== undefined) { tagRight.textContent = rightLabel; imgRight.alt = rightLabel; }
      // Only show the spinner if loading is actually slow; cached previews swap instantly.
      if (pending) setTimeout(() => { if (!(imgLeft.complete && imgRight.complete)) spinner.hidden = false; }, 150);
    },
    setAspect(value) { stage.style.aspectRatio = value; reset(); },
    reset,
  };
}
