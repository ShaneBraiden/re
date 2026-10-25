/* Synchronised wipe comparison: two images under one transform, so the panes
 * can never drift apart - that is the whole point of the comparison.
 *
 * Drag the divider (or focus it and use the arrow keys) to wipe, drag the image
 * to pan, scroll or use the buttons to zoom, double-click to jump between fit
 * and 4x at the pointer. Transforms are written straight to the DOM so a drag
 * never waits on a React render. */

import clsx from "clsx";
import { ImageOff, Maximize2, Minus, Plus, RotateCcw } from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Spinner } from "./ui";

interface Side {
  src: string | undefined;
  label: string;
}

export function CompareViewer({
  left,
  right,
  aspect = "1 / 1",
  maxZoom = 16,
  pixelated = true,
  onFullscreen,
  className,
}: {
  left: Side;
  right: Side;
  aspect?: string;
  maxZoom?: number;
  /** true always, "zoom" only once magnified (for photos larger than the box), false never. */
  pixelated?: boolean | "zoom";
  onFullscreen?: () => void;
  className?: string;
}) {
  const stage = useRef<HTMLDivElement>(null);
  const imgLeft = useRef<HTMLImageElement>(null);
  const imgRight = useRef<HTMLImageElement>(null);
  const layerRight = useRef<HTMLDivElement>(null);
  const divider = useRef<HTMLDivElement>(null);
  const zoomText = useRef<HTMLSpanElement>(null);
  const view = useRef({ scale: 1, x: 0, y: 0, wipe: 0.5 });
  const drag = useRef<null | { mode: "wipe" } | { mode: "pan"; x: number; y: number }>(null);

  const [status, setStatus] = useState<{ left: "loading" | "ok" | "missing"; right: "loading" | "ok" | "missing" }>({
    left: "loading",
    right: "loading",
  });
  const [slow, setSlow] = useState(false);
  const [magnified, setMagnified] = useState(false);

  const apply = useCallback(() => {
    const v = view.current;
    const box = stage.current?.getBoundingClientRect();
    if (v.scale <= 1.0001) Object.assign(v, { scale: 1, x: 0, y: 0 });
    if (box?.width) {
      v.x = Math.min(0, Math.max(box.width * (1 - v.scale), v.x));
      v.y = Math.min(0, Math.max(box.height * (1 - v.scale), v.y));
    }
    const transform = `translate(${v.x}px, ${v.y}px) scale(${v.scale})`;
    if (imgLeft.current) imgLeft.current.style.transform = transform;
    if (imgRight.current) imgRight.current.style.transform = transform;
    if (layerRight.current) layerRight.current.style.clipPath = `inset(0 0 0 ${v.wipe * 100}%)`;
    if (divider.current) {
      divider.current.style.left = `${v.wipe * 100}%`;
      divider.current.setAttribute("aria-valuenow", String(Math.round(v.wipe * 100)));
    }
    if (zoomText.current) zoomText.current.textContent = `${v.scale < 10 ? v.scale.toFixed(1).replace(/\.0$/, "") : Math.round(v.scale)}×`;
    setMagnified(v.scale >= 3);
  }, []);

  const zoomBy = useCallback(
    (factor: number, cx?: number, cy?: number) => {
      const box = stage.current?.getBoundingClientRect();
      if (!box) return;
      const v = view.current;
      const px = cx ?? box.width / 2;
      const py = cy ?? box.height / 2;
      const next = Math.min(maxZoom, Math.max(1, v.scale * factor));
      const ratio = next / v.scale;
      // Zoom about the pointer: the pixel under the cursor stays under it.
      v.x = px - (px - v.x) * ratio;
      v.y = py - (py - v.y) * ratio;
      v.scale = next;
      apply();
    },
    [apply, maxZoom],
  );

  const reset = useCallback(() => {
    Object.assign(view.current, { scale: 1, x: 0, y: 0, wipe: 0.5 });
    apply();
  }, [apply]);

  useEffect(() => {
    apply();
    const el = stage.current;
    if (!el) return;
    // Non-passive, so the page does not scroll while zooming.
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const box = el.getBoundingClientRect();
      zoomBy(Math.exp(-event.deltaY * 0.0015), event.clientX - box.left, event.clientY - box.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    const observer = new ResizeObserver(() => apply());
    observer.observe(el);
    return () => {
      el.removeEventListener("wheel", onWheel);
      observer.disconnect();
    };
  }, [apply, zoomBy]);

  useEffect(() => reset(), [aspect, reset]);

  // A new pair of images: show a spinner only if loading is actually slow.
  useEffect(() => {
    const done = (img: HTMLImageElement | null) => Boolean(img?.complete && img.naturalWidth);
    setStatus({ left: done(imgLeft.current) ? "ok" : "loading", right: done(imgRight.current) ? "ok" : "loading" });
    setSlow(false);
    const timer = setTimeout(() => setSlow(true), 180);
    return () => clearTimeout(timer);
  }, [left.src, right.src]);

  const onPointerDown = (event: React.PointerEvent) => {
    if (event.button !== 0 || (event.target as HTMLElement).closest("[data-tools]")) return;
    if ((event.target as HTMLElement).closest("[data-divider]")) {
      drag.current = { mode: "wipe" };
    } else {
      drag.current = { mode: "pan", x: event.clientX, y: event.clientY };
      stage.current?.classList.add("cursor-grabbing");
    }
    stage.current?.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: React.PointerEvent) => {
    const d = drag.current;
    const box = stage.current?.getBoundingClientRect();
    if (!d || !box) return;
    if (d.mode === "wipe") {
      view.current.wipe = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    } else {
      view.current.x += event.clientX - d.x;
      view.current.y += event.clientY - d.y;
      d.x = event.clientX;
      d.y = event.clientY;
    }
    apply();
  };

  const stop = () => {
    drag.current = null;
    stage.current?.classList.remove("cursor-grabbing");
  };

  const onDoubleClick = (event: React.MouseEvent) => {
    if ((event.target as HTMLElement).closest("[data-tools], [data-divider]")) return;
    const box = stage.current!.getBoundingClientRect();
    if (view.current.scale > 1) reset();
    else zoomBy(4, event.clientX - box.left, event.clientY - box.top);
  };

  const onDividerKey = (event: React.KeyboardEvent) => {
    const v = view.current;
    const step = event.shiftKey ? 0.1 : 0.02;
    if (event.key === "ArrowLeft") v.wipe = Math.max(0, v.wipe - step);
    else if (event.key === "ArrowRight") v.wipe = Math.min(1, v.wipe + step);
    else if (event.key === "Home") v.wipe = 0;
    else if (event.key === "End") v.wipe = 1;
    else return;
    event.preventDefault();
    apply();
  };

  const pixel = pixelated === true || (pixelated === "zoom" && magnified);
  const loading = slow && (status.left === "loading" || status.right === "loading");

  return (
    <div
      ref={stage}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={stop}
      onPointerCancel={stop}
      onDoubleClick={onDoubleClick}
      style={{ aspectRatio: aspect }}
      className={clsx(
        "relative w-full touch-none select-none overflow-hidden rounded-lg bg-[repeating-conic-gradient(var(--surface-2)_0%_25%,var(--surface-3)_0%_50%)] bg-[length:16px_16px]",
        "cursor-grab",
        pixel && "pixelated",
        className,
      )}
    >
      <Layer imgRef={imgLeft} side={left} onState={(s) => setStatus((p) => ({ ...p, left: s }))} />
      <Layer
        layerRef={layerRight}
        imgRef={imgRight}
        side={right}
        onState={(s) => setStatus((p) => ({ ...p, right: s }))}
        clip
      />
      {status.left === "missing" && <Missing className="left-0" />}
      {status.right === "missing" && <Missing className="right-0" />}

      <div
        ref={divider}
        data-divider
        role="slider"
        tabIndex={0}
        aria-label="Comparison divider"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={50}
        onKeyDown={onDividerKey}
        className="group absolute inset-y-0 z-10 -ml-4 flex w-8 cursor-ew-resize justify-center focus-visible:outline-none"
        style={{ left: "50%" }}
      >
        <span className="h-full w-0.5 bg-white/90 shadow-[0_0_0_1px_rgb(0_0_0/0.25)]" />
        <span className="absolute top-1/2 flex size-8 -translate-y-1/2 items-center justify-center rounded-full border border-black/10 bg-white text-[13px] font-semibold text-neutral-700 shadow-lg ring-accent group-focus-visible:ring-2">
          ‹›
        </span>
      </div>

      <CornerTag className="left-2.5">{left.label}</CornerTag>
      <CornerTag className="right-2.5">{right.label}</CornerTag>

      {loading && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/10">
          <Spinner size={22} />
        </div>
      )}

      <div
        data-tools
        className="absolute bottom-2.5 right-2.5 z-20 flex items-center gap-0.5 rounded-lg border border-white/10 bg-black/60 p-0.5 text-white backdrop-blur"
      >
        <ToolButton label="Zoom out" onClick={() => zoomBy(1 / 1.5)}>
          <Minus size={14} />
        </ToolButton>
        <span ref={zoomText} className="num w-9 text-center text-xs font-medium">
          1×
        </span>
        <ToolButton label="Zoom in" onClick={() => zoomBy(1.5)}>
          <Plus size={14} />
        </ToolButton>
        <ToolButton label="Reset view" onClick={reset}>
          <RotateCcw size={13} />
        </ToolButton>
        {onFullscreen && (
          <ToolButton label="Full screen" onClick={onFullscreen}>
            <Maximize2 size={13} />
          </ToolButton>
        )}
      </div>
    </div>
  );
}

function Layer({
  side,
  imgRef,
  layerRef,
  onState,
  clip = false,
}: {
  side: Side;
  imgRef: React.RefObject<HTMLImageElement | null>;
  layerRef?: React.RefObject<HTMLDivElement | null>;
  onState: (state: "ok" | "missing") => void;
  clip?: boolean;
}) {
  return (
    <div ref={layerRef} className="absolute inset-0" style={clip ? { clipPath: "inset(0 0 0 50%)" } : undefined}>
      {side.src && (
        <img
          ref={imgRef}
          src={side.src}
          alt={side.label}
          draggable={false}
          onLoad={() => onState("ok")}
          onError={() => onState("missing")}
          className="absolute inset-0 size-full origin-top-left object-contain"
        />
      )}
    </div>
  );
}

function Missing({ className }: { className: string }) {
  return (
    <div className={clsx("absolute inset-y-0 flex w-1/2 flex-col items-center justify-center gap-1.5 text-xs text-fg-3", className)}>
      <ImageOff size={20} />
      no preview
    </div>
  );
}

function CornerTag({ children, className }: { children: React.ReactNode; className: string }) {
  return (
    <div
      className={clsx(
        "pointer-events-none absolute top-2.5 z-10 max-w-[45%] truncate rounded-md bg-black/60 px-2 py-1 text-[11.5px] font-medium text-white backdrop-blur",
        className,
      )}
    >
      {children}
    </div>
  );
}

function ToolButton({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="inline-flex size-7 items-center justify-center rounded-md transition-colors hover:bg-white/15"
    >
      {children}
    </button>
  );
}
