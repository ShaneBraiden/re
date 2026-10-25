/* Which run the page is looking at, and links that keep looking at it.
 *
 * Routes live in the hash so every view is a link you can share - the same
 * shape the classic dashboard used, so old links still open:
 *   #/explore?exp=enhance_study&image=nat_00&sev=4
 * ?exp=<run> on any route picks a run other than the server's default. */

import { createContext, useCallback, useContext, useMemo, type ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { query } from "./api";
import { useExperiments, useHealth, useIndex } from "./queries";
import type { ExperimentSummary, Health, RunIndex } from "./types";

interface RunContextValue {
  booting: boolean;
  health: Health | undefined;
  experiments: ExperimentSummary[];
  defaultExp: string | null;
  exp: string | null;
  index: RunIndex | undefined;
  indexLoading: boolean;
  indexError: string | null;
  upload: boolean;
  /** A route that keeps the current run selected. */
  link: (path: string, params?: Record<string, string | number | null | undefined>) => string;
  setExp: (exp: string) => void;
}

const RunContext = createContext<RunContextValue | null>(null);

export function RunProvider({ children }: { children: ReactNode }) {
  const health = useHealth();
  const listing = useExperiments();
  const [params] = useSearchParams();
  const navigate = useNavigate();

  const experiments = useMemo(
    () => (listing.data?.experiments ?? []).filter((run) => !run.error || run.default),
    [listing.data],
  );
  let defaultExp = listing.data?.default ?? health.data?.default ?? null;
  if (defaultExp && experiments.length && !experiments.some((run) => run.id === defaultExp)) defaultExp = experiments[0].id;

  const exp = params.get("exp") || defaultExp;
  const index = useIndex(exp);

  const link = useCallback(
    (path: string, extra: Record<string, string | number | null | undefined> = {}) => {
      const next = { ...extra };
      if (exp && exp !== defaultExp && !("exp" in next)) next.exp = exp;
      const q = query(next);
      return `${path}${q ? `?${q}` : ""}`;
    },
    [exp, defaultExp],
  );

  const setExp = useCallback(
    (value: string) => {
      const path = window.location.hash.replace(/^#/, "").split("?")[0] || "/";
      // Run-specific view state (scene, severity...) does not carry across runs.
      navigate(value === defaultExp ? path : `${path}?${query({ exp: value })}`);
    },
    [navigate, defaultExp],
  );

  let indexError: string | null = null;
  if (!exp && !health.isLoading && !listing.isLoading) indexError = health.data?.note || "No experiment run found.";
  else if (index.error) indexError = String((index.error as Error).message || index.error);

  const value: RunContextValue = {
    booting: health.isLoading || listing.isLoading,
    health: health.data,
    experiments,
    defaultExp,
    exp,
    index: index.data,
    indexLoading: index.isLoading,
    indexError,
    upload: Boolean(health.data?.upload),
    link,
    setExp,
  };
  return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

export function useRun(): RunContextValue {
  const value = useContext(RunContext);
  if (!value) throw new Error("useRun outside RunProvider");
  return value;
}

/** A page whose content needs the run's index. Throws only on misuse: pages
 * that call it are rendered behind <RequireIndex>. */
export function useIndexed(): RunContextValue & { index: RunIndex; exp: string } {
  const value = useRun();
  if (!value.index || !value.exp) throw new Error("useIndexed before the index loaded");
  return value as RunContextValue & { index: RunIndex; exp: string };
}

/** View state kept in the URL, written with replace so it does not flood history. */
export function useUrlState() {
  const [params, setParams] = useSearchParams();
  const patch = useCallback(
    (changes: Record<string, string | number | null | undefined>) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          for (const [key, value] of Object.entries(changes)) {
            if (value === null || value === undefined || value === "") next.delete(key);
            else next.set(key, String(value));
          }
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );
  return [params, patch] as const;
}
