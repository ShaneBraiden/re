import { fmt, isNum, type Num } from "./format";
import type { Direction, RunIndex } from "./types";

interface MetricMeta {
  label: string;
  unit?: string;
  digits?: number;
}

const METRICS: Record<string, MetricMeta> = {
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

export const FAMILY_ORDER = ["fidelity", "perceptual", "machine_task"];

export const FAMILY_LABELS: Record<string, string> = {
  fidelity: "Fidelity",
  perceptual: "Perceptual",
  machine_task: "Machine vision",
  other: "Other",
};

export const metricLabel = (metric: string) => METRICS[metric]?.label ?? metric;
export const metricUnit = (metric: string) => METRICS[metric]?.unit ?? "";

export function metricTitle(metric: string): string {
  const unit = metricUnit(metric);
  return unit ? `${metricLabel(metric)} (${unit})` : metricLabel(metric);
}

export function fmtMetric(metric: string, value: Num): string {
  const digits = METRICS[metric]?.digits ?? null;
  if (digits === 0 && isNum(value) && !Number.isInteger(value)) return fmt(value, 1);
  return fmt(value, digits);
}

export function directionNote(direction: Direction | undefined): string {
  if (direction === null) return "closest to 1.0 is best";
  return direction === false ? "lower is better" : "higher is better";
}

/** A comparable score where bigger is always better, or null for a missing value. */
export function score(direction: Direction | undefined, value: Num): number | null {
  if (!isNum(value)) return null;
  if (direction === null) return -Math.abs(value - 1);
  return direction === false ? -value : value;
}

/** Index of the best entry, or -1. Ties keep the first. */
export function bestIndex(values: Num[], direction: Direction | undefined): number {
  let best = -1;
  let bestScore = -Infinity;
  values.forEach((value, i) => {
    const sc = score(direction, value);
    if (sc !== null && sc > bestScore) {
      bestScore = sc;
      best = i;
    }
  });
  return best;
}

/** Sort comparator: best first, missing values last. */
export function byDirection(direction: Direction | undefined) {
  return (a: Num, b: Num) => {
    const sa = score(direction, a);
    const sb = score(direction, b);
    if (sa === null && sb === null) return 0;
    if (sa === null) return 1;
    if (sb === null) return -1;
    return sb - sa;
  };
}

/** "good" | "bad" | null for a difference, given which way is better. */
export function deltaTone(delta: Num, direction: Direction | undefined): "good" | "bad" | null {
  // A target-valued metric has no "better" sign for a difference.
  if (!isNum(delta) || delta === 0 || direction === null) return null;
  return (direction === false ? delta < 0 : delta > 0) ? "good" : "bad";
}

export interface OptionGroup {
  group: string;
  items: [string, string][];
}

/** Metric picker options, grouped by family in the study's order. */
export function metricGroups(index: RunIndex): OptionGroup[] {
  const groups = new Map<string, [string, string][]>();
  for (const metric of index.metrics) {
    const family = index.families?.[metric] ?? "other";
    if (!groups.has(family)) groups.set(family, []);
    groups.get(family)!.push([metric, metricLabel(metric)]);
  }
  const rank = (family: string) => (FAMILY_ORDER.indexOf(family) + 1 || 99);
  return [...groups]
    .sort(([a], [b]) => rank(a) - rank(b))
    .map(([family, items]) => ({ group: FAMILY_LABELS[family] ?? family, items }));
}

export function defaultMetric(index: RunIndex, wanted?: string | null): string {
  if (wanted && index.metrics.includes(wanted)) return wanted;
  return index.metrics.includes("psnr") ? "psnr" : index.metrics[0];
}

export function sortByFamily(metrics: string[], families: Record<string, string>): string[] {
  const rank = (metric: string) => FAMILY_ORDER.indexOf(families[metric] ?? "other") + 1 || 99;
  return [...metrics].sort((a, b) => rank(a) - rank(b));
}
