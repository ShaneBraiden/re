/* Number formatting. A typographic minus, an em dash for a missing value, and
 * precision that follows magnitude unless a metric pins it. */

export type Num = number | null | undefined;

export const isNum = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

export function fmt(value: Num, digits: number | null = null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (!Number.isFinite(value)) return value > 0 ? "∞" : "−∞";
  const abs = Math.abs(value);
  let text: string;
  if (digits !== null) text = value.toFixed(digits);
  else if (abs >= 1000) text = value.toFixed(0);
  else if (abs >= 100) text = value.toFixed(1);
  else if (abs >= 1) text = value.toFixed(2);
  else if (abs === 0) text = "0";
  else text = value.toFixed(3);
  return text.replace(/^-/, "−");
}

export function fmtSigned(value: Num, digits: number | null = null): string {
  if (!isNum(value)) return "—";
  return `${value >= 0 ? "+" : "−"}${fmt(Math.abs(value), digits)}`;
}

export function fmtP(p: Num): string {
  if (!isNum(p)) return "—";
  if (p < 0.001) return "< 0.001";
  return p.toFixed(3);
}

/** Small magnitudes in scientific notation, the rest as usual. */
export function fmtSci(value: Num): string {
  if (!isNum(value)) return "—";
  if (value !== 0 && Math.abs(value) < 0.01) return value.toExponential(2).replace("-", "−");
  return fmt(value);
}

export function fmtPercent(fraction: Num, digits?: number): string {
  if (!isNum(fraction)) return "—";
  return `${fmt(fraction * 100, digits ?? (fraction < 0.1 ? 1 : 0))}%`;
}

export function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

export function fmtDate(value: string | number | Date | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const date = value instanceof Date ? value : new Date(typeof value === "number" ? value * 1000 : value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function fmtDuration(seconds: Num): string {
  if (!isNum(seconds)) return "—";
  if (seconds < 60) return `${fmt(seconds, seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${Math.round(seconds - minutes * 60)} s`;
}
