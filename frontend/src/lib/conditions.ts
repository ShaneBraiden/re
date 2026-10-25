import { fmt, fmtPercent } from "./format";
import type { Condition, RunIndex } from "./types";

export const sortedConditions = (index: RunIndex): Condition[] => [...index.conditions].sort((a, b) => a.severity - b.severity);

/** What one degradation condition means in hardware terms. */
export function describeCondition(condition: Condition): string {
  const p = condition.params ?? {};
  const parts: string[] = [];
  if (p.alpha !== undefined) parts.push(`${fmtPercent(p.alpha)} light`);
  if (p.sigma_r_dn !== undefined) parts.push(`read noise ${fmt(p.sigma_r_dn, 1)} DN`);
  if (p.r_psf !== undefined) parts.push(`blur ${fmt(p.r_psf, 2)} px`);
  if (p.binning !== undefined && p.binning > 1) parts.push(`${p.binning}×${p.binning} binning`);
  if (p.bit_depth !== undefined) parts.push(`${p.bit_depth}-bit ADC`);
  return parts.length ? parts.join(" · ") : condition.label;
}
