import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "./api";

/* A finished run does not change under the page, so its slices are cached for
 * the session. The server reloads dashboard.json when a rerun rewrites it, and
 * the run list and the enhancement history - which do grow - are always fresh. */

const FOREVER = { staleTime: Infinity, gcTime: 30 * 60_000 } as const;
const sorted = (names: string[]) => [...names].sort();

export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: api.health, ...FOREVER });

export const useExperiments = () => useQuery({ queryKey: ["experiments"], queryFn: api.experiments, staleTime: 0 });

export const useIndex = (exp: string | null) =>
  useQuery({ queryKey: ["index", exp], queryFn: () => api.index(exp!), enabled: Boolean(exp), ...FOREVER });

export const useRunInfo = (exp: string | null) =>
  useQuery({ queryKey: ["run", exp], queryFn: () => api.run(exp!), enabled: Boolean(exp), ...FOREVER });

export const useAnalysis = (exp: string | null) =>
  useQuery({ queryKey: ["analysis", exp], queryFn: () => api.analysis(exp!), enabled: Boolean(exp), ...FOREVER });

export const useAggregate = (exp: string | null, metric: string, pipelines: string[]) =>
  useQuery({
    queryKey: ["aggregate", exp, metric, sorted(pipelines)],
    queryFn: () => api.aggregate(exp!, metric, pipelines),
    enabled: Boolean(exp) && pipelines.length > 0,
    placeholderData: keepPreviousData,
    ...FOREVER,
  });

export const useSeries = (exp: string | null, image: string | undefined, metric: string, pipelines: string[]) =>
  useQuery({
    queryKey: ["series", exp, image, metric, sorted(pipelines)],
    queryFn: () => api.series(exp!, image!, metric, pipelines),
    enabled: Boolean(exp && image) && pipelines.length > 0,
    placeholderData: keepPreviousData,
    ...FOREVER,
  });

export const useValues = (exp: string | null, image: string | undefined, condition: string | undefined, pipelines: string[]) =>
  useQuery({
    queryKey: ["values", exp, image, condition, sorted(pipelines)],
    queryFn: () => api.values(exp!, image!, condition!, pipelines),
    enabled: Boolean(exp && image && condition) && pipelines.length > 0,
    placeholderData: keepPreviousData,
    ...FOREVER,
  });

export const useEnhancements = () => useQuery({ queryKey: ["enhancements"], queryFn: api.enhancements, staleTime: 0 });
