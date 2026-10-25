/* The only module that talks to the server. Caching and request state live in
 * TanStack Query (src/lib/queries.ts); this file is the URL vocabulary. */

import type {
  AggregateResponse,
  AnalysisResponse,
  EnhanceResult,
  EnhancementList,
  ExperimentList,
  Health,
  RunIndex,
  RunInfo,
  SeriesResponse,
  ValuesResponse,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

type Params = Record<string, string | number | string[] | null | undefined>;

export function query(params: Params): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    search.set(key, Array.isArray(value) ? value.join(",") : String(value));
  }
  return search.toString();
}

async function getJSON<T>(url: string): Promise<T> {
  const response = await fetch(url);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(body.error || body.note || `${response.status} ${response.statusText}`, response.status);
  }
  return body as T;
}

export const api = {
  async health(): Promise<Health> {
    // 503 is an answer here ("no run yet"), not a failure.
    const response = await fetch("/api/health");
    return response.json().catch(() => ({ ready: false, upload: false }));
  },
  experiments: () => getJSON<ExperimentList>("/api/experiments"),
  index: (exp: string) => getJSON<RunIndex>(`/api/index?${query({ exp })}`),
  run: (exp: string) => getJSON<RunInfo>(`/api/run?${query({ exp })}`),
  values: (exp: string, image: string, condition: string, pipelines: string[]) =>
    getJSON<ValuesResponse>(`/api/values?${query({ exp, image, condition, pipelines })}`),
  series: (exp: string, image: string, metric: string, pipelines: string[]) =>
    getJSON<SeriesResponse>(`/api/series?${query({ exp, image, metric, pipelines })}`),
  aggregate: (exp: string, metric: string, pipelines: string[]) =>
    getJSON<AggregateResponse>(`/api/aggregate?${query({ exp, metric, pipelines })}`),
  analysis: (exp: string) => getJSON<AnalysisResponse>(`/api/analysis?${query({ exp })}`),
  enhancements: () => getJSON<EnhancementList>("/api/enhancements"),

  async enhance(form: FormData): Promise<EnhanceResult> {
    const response = await fetch("/api/enhance", { method: "POST", body: form });
    const body = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
    if (!response.ok) {
      const message = response.status === 413 ? "That file is over the 20 MB limit." : body.error || `HTTP ${response.status}`;
      throw new ApiError(message, response.status);
    }
    return body as EnhanceResult;
  },
};

export const REFERENCE = "__reference__";
export const DEGRADED = "__degraded__";

/** The preview JPEG an earlier run wrote for one scene, condition and output. */
export function previewURL(exp: string, image: string, which: string, conditionHash?: string): string {
  const name =
    which === REFERENCE ? "reference.jpg" : which === DEGRADED ? `${conditionHash}__degraded.jpg` : `${conditionHash}__${which}.jpg`;
  return `/previews/${encodeURIComponent(image)}/${name}?${query({ exp })}`;
}
