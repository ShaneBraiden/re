import { pipelineInfo } from "../lib/pipelines";
import { Tag } from "./ui";

/** A short line sample: hue and dash, so the legend shows exactly what the chart draws. */
export function Swatch({ name, color, dash, off = false }: { name?: string; color?: string; dash?: string; off?: boolean }) {
  const info = name ? pipelineInfo(name) : null;
  const stroke = off ? "var(--axis)" : `var(${color ?? info?.color ?? "--series-1"})`;
  return (
    <svg width="20" height="10" viewBox="0 0 20 10" aria-hidden className="shrink-0">
      <line x1="2" y1="5" x2="18" y2="5" strokeWidth="2.5" strokeLinecap="round" style={{ stroke }} strokeDasharray={dash ?? info?.dash} />
    </svg>
  );
}

export function PipelineName({ name, showId = false, tags = true }: { name: string; showId?: boolean; tags?: boolean }) {
  const info = pipelineInfo(name);
  return (
    <span className="inline-flex min-w-0 items-center gap-2 whitespace-nowrap" title={info.blurb ?? name}>
      <Swatch name={name} />
      <span className="truncate">{info.label}</span>
      {tags && info.proposed && <Tag tone="accent">ours</Tag>}
      {tags && info.baseline && <Tag>baseline</Tag>}
      {showId && info.label !== name && <code className="text-[11px] text-fg-3">{name}</code>}
    </span>
  );
}
