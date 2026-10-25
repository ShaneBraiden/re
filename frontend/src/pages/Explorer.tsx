/* Explorer: one scene, one hardware condition, any two outputs side by side.
 *
 * The URL is the state: scene, severity, metric, the arms switched on and the
 * two sides of the comparison all live in the query string, so every view is a
 * link. The image comparison, the ranking at this condition, the sweep chart
 * and the full metric table all render against the same slice. */

import clsx from "clsx";
import { ArrowLeftRight } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, type ReactNode } from "react";
import { CompareViewer } from "../components/CompareViewer";
import { DataTable, ExportButton, type Column } from "../components/DataTable";
import { PageHeader } from "../components/Layout";
import { ChartCard } from "../components/LineChart";
import { PipelineName, Swatch } from "../components/PipelineName";
import { Card, Delta, EmptyState, ErrorBanner, Field, IconButton, Kbd, Select, Skeleton } from "../components/ui";
import { DEGRADED, previewURL, REFERENCE } from "../lib/api";
import { describeCondition, sortedConditions } from "../lib/conditions";
import { fmt, fmtSigned } from "../lib/format";
import {
  byDirection,
  defaultMetric,
  deltaTone,
  directionNote,
  FAMILY_LABELS,
  fmtMetric,
  metricGroups,
  metricLabel,
  metricUnit,
  sortByFamily,
} from "../lib/metrics";
import { armPresets, BASELINE, GROUPS, inBankOrder, pipelineInfo, type Preset } from "../lib/pipelines";
import { useSeries, useValues } from "../lib/queries";
import { useIndexed, useUrlState } from "../lib/run";
import type { MetricValues } from "../lib/types";

const sameSet = (a: Set<string>, b: Set<string>) => a.size === b.size && [...a].every((x) => b.has(x));

export default function Explorer() {
  const { index, exp } = useIndexed();
  const [params, patch] = useUrlState();

  const conditions = useMemo(() => sortedConditions(index), [index]);
  const directions = index.directions ?? {};
  const families = index.families ?? {};
  const available = useMemo(() => index.pipelines.filter((p) => p.available).map((p) => p.name), [index]);
  const presets = useMemo(() => armPresets(available), [available]);
  const defaultActive = useMemo(() => new Set(presets ? presets.main : available), [presets, available]);

  /* ---------------------------------------------------- state from URL --- */

  const image = index.images.some((i) => i.image_id === params.get("image")) ? params.get("image")! : index.images[0]?.image_id;
  const sevParam = Number(params.get("sev"));
  const sev = Math.min(conditions.length - 1, Math.max(0, Number.isFinite(sevParam) ? Math.round(sevParam) : 0));
  const metric = defaultMetric(index, params.get("metric"));
  const active = useMemo(() => {
    const pipes = params.get("pipes");
    const set = pipes !== null ? new Set(pipes.split(",").filter((n) => available.includes(n))) : new Set(defaultActive);
    const right = params.get("right");
    if (right && available.includes(right)) set.add(right);
    return set;
  }, [params, available, defaultActive]);
  const validSide = (v: string | null): v is string => v === REFERENCE || v === DEGRADED || (v !== null && available.includes(v));
  const left = validSide(params.get("left")) ? params.get("left")! : DEGRADED;
  const rightParam = params.get("right");
  const right = validSide(rightParam)
    ? rightParam
    : ([...active].find((n) => pipelineInfo(n).proposed) ?? inBankOrder([...active], available).pop() ?? REFERENCE);

  const condition = conditions[sev];
  const names = useMemo(() => inBankOrder([...active], available), [active, available]);

  const setActive = (next: Set<string>) => {
    const pipes = sameSet(next, defaultActive) ? null : next.size ? [...next].join(",") : "none";
    // An arm switched off leaves the comparison too; the reference sides stay.
    const keepRight = rightParam === REFERENCE || rightParam === DEGRADED || (rightParam !== null && next.has(rightParam));
    patch({ pipes, right: keepRight ? rightParam : null });
  };

  const setSev = useCallback(
    (i: number) => patch({ sev: Math.min(conditions.length - 1, Math.max(0, i)) || null }),
    [patch, conditions.length],
  );

  // [ and ] step through the sweep from anywhere on the page.
  const sevRef = useRef(sev);
  sevRef.current = sev;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement).closest("input, select, textarea, [role=slider]")) return;
      if (event.key === "[" || event.key === "]") {
        event.preventDefault();
        setSev(sevRef.current + (event.key === "]" ? 1 : -1));
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [setSev]);

  /* ------------------------------------------------------------- data --- */

  const series = useSeries(exp, image, metric, names);
  const values = useValues(exp, image, condition?.hash, names);

  if (!image) {
    return (
      <>
        <PageHeader title="Explorer" />
        <Card>
          <EmptyState title="This run has no scenes" detail="dashboard.json lists no images." />
        </Card>
      </>
    );
  }

  const label = (which: string) => (which === REFERENCE ? "Ground truth" : which === DEGRADED ? "Degraded input" : pipelineInfo(which).label);
  const sideOptions = [
    {
      group: "Reference",
      items: [
        [REFERENCE, "Ground truth"],
        [DEGRADED, "Degraded input"],
      ] as [string, string][],
    },
    ...GROUPS.map((g) => ({
      group: g.label,
      items: available.filter((n) => pipelineInfo(n).group === g.id).map((n) => [n, pipelineInfo(n).label] as [string, string]),
    })).filter((g) => g.items.length),
  ];

  return (
    <>
      <PageHeader
        title="Explorer"
        sub="One scene under one hardware condition. Compare any two outputs pixel for pixel, then see where they rank."
        actions={
          <span className="hidden items-center gap-1.5 text-xs text-fg-3 md:inline-flex">
            <Kbd>[</Kbd>
            <Kbd>]</Kbd> step severity
          </span>
        }
      />

      <div className="flex flex-col gap-5">
        {/* Controls that scope everything below. */}
        <div className="sticky top-14 z-20 -mx-4 border-y border-line bg-bg/90 px-4 py-3 backdrop-blur sm:-mx-6 sm:px-6 lg:top-0 lg:-mx-8 lg:px-8">
          <div className="grid gap-4 md:grid-cols-[auto_minmax(0,1fr)_auto] md:items-end">
            <Field label="Scene">
              <div className="flex gap-1.5 overflow-x-auto scroll-thin" role="group" aria-label="Scene">
                {index.images.map((img) => (
                  <button
                    key={img.image_id}
                    type="button"
                    aria-pressed={img.image_id === image}
                    title={img.image_id}
                    onClick={() => patch({ image: img.image_id === index.images[0].image_id ? null : img.image_id })}
                    className={clsx(
                      "flex shrink-0 items-center gap-2 rounded-lg border p-1 pr-2.5 text-xs font-medium transition-colors",
                      img.image_id === image ? "border-accent bg-accent-soft text-accent" : "border-line bg-surface text-fg-2 hover:border-line-strong",
                    )}
                  >
                    <img src={previewURL(exp, img.image_id, REFERENCE)} alt="" loading="lazy" className="size-8 rounded-md object-cover" />
                    {img.image_id}
                  </button>
                ))}
              </div>
            </Field>

            <div className="flex min-w-0 flex-col gap-1.5">
              <div className="flex items-baseline justify-between gap-3">
                <span className="shrink-0 whitespace-nowrap text-[11px] font-semibold uppercase tracking-[0.06em] text-fg-3">
                  Hardware severity <b className="num ml-1 text-[13px] normal-case tracking-normal text-fg">{fmt(condition.severity, 2)}</b>
                </span>
                <span className="truncate text-xs text-fg-3">{sev === 0 ? `${describeCondition(condition)} · the clean sensor` : describeCondition(condition)}</span>
              </div>
              <input
                type="range"
                className="range"
                min={0}
                max={conditions.length - 1}
                step={1}
                value={sev}
                onChange={(event) => setSev(Number(event.target.value))}
                aria-label="Degradation severity"
                aria-valuetext={`severity ${fmt(condition.severity, 2)}: ${describeCondition(condition)}`}
                style={{ ["--fill" as string]: `${(sev / Math.max(1, conditions.length - 1)) * 100}%` }}
              />
              <div className="flex justify-between" aria-hidden>
                {conditions.map((c, i) => (
                  <button
                    key={c.hash}
                    type="button"
                    tabIndex={-1}
                    onClick={() => setSev(i)}
                    className={clsx("num rounded px-1 text-[11px] transition-colors", i === sev ? "font-semibold text-accent" : "text-fg-3 hover:text-fg")}
                  >
                    {fmt(c.severity, 2)}
                  </button>
                ))}
              </div>
            </div>

            <Field label="Metric">
              <Select label="Metric" value={metric} onChange={(v) => patch({ metric: v === defaultMetric(index) ? null : v })} options={metricGroups(index)} className="w-full md:w-52" />
            </Field>
          </div>
        </div>

        <PipelinePicker index={index} active={active} setActive={setActive} presets={presets} />

        <div className="grid gap-5 lg:grid-cols-12">
          <ViewerCard
            className="lg:col-span-7"
            left={{ src: previewURL(exp, image, left, condition.hash), label: label(left) }}
            right={{ src: previewURL(exp, image, right, condition.hash), label: label(right) }}
            leftSelect={<Select label="Left image" value={left} onChange={(v) => patch({ left: v === DEGRADED ? null : v })} options={sideOptions} className="min-w-0 flex-1 sm:w-48 sm:flex-none" />}
            rightSelect={<Select label="Right image" value={right} onChange={(v) => patch({ right: v })} options={sideOptions} className="min-w-0 flex-1 sm:w-48 sm:flex-none" />}
            onSwap={() => patch({ left: right === DEGRADED ? null : right, right: left })}
          />
          <div className="lg:col-span-5">
            <ConditionCard
              names={names}
              series={series.data?.series}
              loading={series.isLoading}
              error={series.error ? String((series.error as Error).message) : null}
              sev={sev}
              severity={condition.severity}
              description={describeCondition(condition)}
              image={image}
              metric={metric}
              direction={directions[metric]}
              focus={right}
              onFocus={(name) => patch({ right: name })}
            />
          </div>
        </div>

        {names.length > 0 && (
          <ChartCard
            title={`${metricLabel(metric)} across the sweep`}
            sub={`${image} · ${directionNote(directions[metric])} · the shaded column is the condition shown above · click a point to jump there`}
            filename={`${index.experiment_id}_${image}_${metric}.csv`}
            loading={series.isFetching}
            chart={{
              series: series.data
                ? inBankOrder(Object.keys(series.data.series), available).map((name) => {
                    const info = pipelineInfo(name);
                    return {
                      name,
                      label: info.label,
                      color: info.color,
                      dash: info.dash,
                      points: series.data!.series[name].map((pt) => ({ x: pt.severity, y: pt.value })),
                    };
                  })
                : [],
              height: 320,
              marker: condition.severity,
              xTitle: "Degradation severity",
              yTitle: `${metricLabel(metric)}${metricUnit(metric) ? ` (${metricUnit(metric)})` : ""} · ${image}`,
              xFormat: (x) => fmt(x, 2),
              yFormat: (y) => fmtMetric(metric, y),
              direction: directions[metric],
              onPick: (severity) => setSev(conditions.findIndex((c) => c.severity === severity)),
              pickHint: "Click to jump the comparison here",
            }}
          />
        )}

        <MetricsTable
          names={names}
          values={values.data?.values}
          loading={values.isLoading}
          fetching={values.isFetching}
          error={values.error ? String((values.error as Error).message) : null}
          families={families}
          directions={directions}
          image={image}
          severity={condition.severity}
          focus={right}
          onFocus={(name) => patch({ right: name })}
          filename={`${index.experiment_id}_${image}_s${fmt(condition.severity, 2)}_metrics.csv`}
        />
      </div>
    </>
  );
}

/* ------------------------------------------------------------ picker --- */

function PipelinePicker({
  index,
  active,
  setActive,
  presets,
}: {
  index: ReturnType<typeof useIndexed>["index"];
  active: Set<string>;
  setActive: (next: Set<string>) => void;
  presets: Record<Preset, string[]> | null;
}) {
  const toggle = (name: string) => {
    const next = new Set(active);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setActive(next);
  };
  return (
    <Card
      title="Pipelines"
      sub="Pick the arms to compare. Colour and line style follow each arm everywhere on the page."
      tools={
        presets && (
          <div className="flex items-center gap-1 text-xs">
            <span className="mr-1 text-fg-3">Quick sets:</span>
            {(
              [
                ["main", "Main arms"],
                ["ablations", "Ablations"],
                ["all", "Everything"],
              ] as [Preset, string][]
            ).map(([key, text]) => (
              <button
                key={key}
                type="button"
                onClick={() => setActive(new Set(presets[key]))}
                className={clsx(
                  "rounded-md px-2 py-1 font-medium transition-colors",
                  sameSet(active, new Set(presets[key])) ? "bg-accent-soft text-accent" : "text-fg-2 hover:bg-surface-2 hover:text-fg",
                )}
              >
                {text}
              </button>
            ))}
          </div>
        )
      }
    >
      <div className="flex flex-col gap-3">
        {GROUPS.map((group) => {
          const members = index.pipelines.filter((p) => pipelineInfo(p.name).group === group.id);
          if (!members.length) return null;
          const runnable = members.filter((m) => m.available).map((m) => m.name);
          const all = runnable.length > 0 && runnable.every((n) => active.has(n));
          return (
            <div key={group.id} className="flex flex-col gap-1.5 sm:flex-row sm:items-start sm:gap-4">
              <div className="flex w-36 shrink-0 items-center gap-2 pt-1.5">
                <span className="text-xs font-semibold text-fg-2">{group.label}</span>
                {runnable.length > 0 && (
                  <button
                    type="button"
                    className="text-[11px] text-accent hover:underline"
                    onClick={() => {
                      const next = new Set(active);
                      for (const n of runnable) {
                        if (all) next.delete(n);
                        else next.add(n);
                      }
                      setActive(next);
                    }}
                  >
                    {all ? "none" : "all"}
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {members.map((member) => {
                  const on = active.has(member.name);
                  const info = pipelineInfo(member.name);
                  return (
                    <button
                      key={member.name}
                      type="button"
                      aria-pressed={on}
                      disabled={!member.available}
                      title={member.available ? (info.blurb ?? member.name) : `${member.name}: ${member.reason}`}
                      onClick={() => toggle(member.name)}
                      className={clsx(
                        "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[12.5px] font-medium transition-colors",
                        !member.available
                          ? "cursor-not-allowed border-dashed border-line text-fg-3 opacity-70"
                          : on
                            ? "border-line-strong bg-surface text-fg shadow-card"
                            : "border-transparent bg-surface-2 text-fg-3 hover:text-fg-2",
                      )}
                    >
                      <Swatch name={member.name} off={!member.available || !on} />
                      {info.label}
                      {!member.available && <span className="text-[10.5px] font-normal">did not run</span>}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------ viewer --- */

function ViewerCard({
  left,
  right,
  leftSelect,
  rightSelect,
  onSwap,
  className,
}: {
  left: { src: string; label: string };
  right: { src: string; label: string };
  leftSelect: ReactNode;
  rightSelect: ReactNode;
  onSwap: () => void;
  className?: string;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  return (
    <Card
      className={className}
      title="Side by side"
      sub="Drag the divider to wipe · drag to pan · scroll or double-click to zoom"
      tools={
        <div className="flex w-full items-center gap-1.5 sm:w-auto">
          {leftSelect}
          <IconButton label="Swap sides" onClick={onSwap} className="shrink-0">
            <ArrowLeftRight size={15} />
          </IconButton>
          {rightSelect}
        </div>
      }
    >
      <div ref={wrap} className="flex items-center justify-center [&:fullscreen]:bg-black">
        <CompareViewer
          left={left}
          right={right}
          aspect="1 / 1"
          className="[:fullscreen_&]:h-full [:fullscreen_&]:w-auto [:fullscreen_&]:max-w-full [:fullscreen_&]:rounded-none"
          onFullscreen={() => (document.fullscreenElement ? document.exitFullscreen() : wrap.current?.requestFullscreen())}
        />
      </div>
    </Card>
  );
}

/* --------------------------------------------------------- condition --- */

function ConditionCard({
  names,
  series,
  loading,
  error,
  sev,
  severity,
  description,
  image,
  metric,
  direction,
  focus,
  onFocus,
}: {
  names: string[];
  series: Record<string, { severity: number; value: number | null }[]> | undefined;
  loading: boolean;
  error: string | null;
  sev: number;
  severity: number;
  description: string;
  image: string;
  metric: string;
  direction: boolean | null | undefined;
  focus: string;
  onFocus: (name: string) => void;
}) {
  const unit = metricUnit(metric);
  const title = `At severity ${fmt(severity, 2)}`;
  if (!names.length) {
    return (
      <Card title={title}>
        <EmptyState title="No pipeline selected" detail="Switch on at least one arm above." />
      </Card>
    );
  }
  if (error) {
    return (
      <Card title={title}>
        <ErrorBanner>{error}</ErrorBanner>
      </Card>
    );
  }
  if (loading || !series) return <Skeleton className="h-[420px]" />;

  const pointAt = (name: string, i: number) => series[name]?.[i]?.value ?? null;
  const shown = names.filter((n) => series[n]);
  const rows = shown
    .map((name) => {
      const value = pointAt(name, sev);
      const clean = pointAt(name, 0);
      return { name, value, delta: value !== null && clean !== null ? value - clean : null };
    })
    .sort((a, b) => byDirection(direction)(a.value, b.value));
  const focused = shown.includes(focus) ? focus : null;

  let hero = <p className="text-[13px] text-fg-3">Pick an arm for the right side of the comparison to see its numbers here.</p>;
  if (focused) {
    const value = pointAt(focused, sev);
    const clean = pointAt(focused, 0);
    const base = shown.includes(BASELINE) && focused !== BASELINE ? pointAt(BASELINE, sev) : null;
    const rank = rows.findIndex((r) => r.name === focused) + 1;
    const note = (delta: number | null, text: string) =>
      delta === null ? null : (
        <span className="text-xs text-fg-3">
          <Delta value={delta} tone={deltaTone(delta, direction)}>
            {fmtSigned(delta)}
          </Delta>{" "}
          {text}
        </span>
      );
    hero = (
      <div className="flex flex-col gap-1">
        <span className="flex items-center gap-1.5 text-xs font-medium text-fg-2">
          {metricLabel(metric)} · <PipelineName name={focused} />
        </span>
        <span className="num flex items-baseline gap-1 text-[34px] font-semibold leading-10 tracking-[-0.02em] text-fg">
          {fmtMetric(metric, value)}
          {unit && <span className="text-sm font-medium tracking-normal text-fg-3">{unit}</span>}
        </span>
        <div className="flex flex-col gap-0.5">
          {sev !== 0 && note(value !== null && clean !== null ? value - clean : null, "vs this arm on the clean sensor")}
          {base !== null && note(value !== null ? value - base : null, "vs the baseline here")}
          <span className="text-xs text-fg-3">
            ranked {rank} of {rows.length} · {directionNote(direction)}
          </span>
        </div>
      </div>
    );
  }

  type R = (typeof rows)[number];
  const columns: Column<R>[] = [
    { key: "rank", header: "#", sortable: false, render: (r) => <span className="text-fg-3">{rows.indexOf(r) + 1}</span> },
    { key: "name", header: "Pipeline", align: "left", value: (r) => r.name, render: (r) => <PipelineName name={r.name} /> },
    { key: "value", header: metricLabel(metric), value: (r) => r.value, best: direction, render: (r) => fmtMetric(metric, r.value) },
    {
      key: "delta",
      header: "vs clean",
      title: "change from the same arm on the clean sensor",
      value: (r) => r.delta,
      render: (r) =>
        sev === 0 ? (
          "—"
        ) : (
          <Delta value={r.delta} tone={deltaTone(r.delta, direction)}>
            {fmtSigned(r.delta)}
          </Delta>
        ),
    },
  ];

  return (
    <Card title={title} sub={`${image} · ${description}`} flush className="h-full">
      <div className="border-b border-line px-4 pb-4 sm:px-5">{hero}</div>
      <DataTable columns={columns} rows={rows} rowKey={(r) => r.name} onRowClick={(row) => onFocus(row.name)} isActive={(row) => row.name === focus} dense />
    </Card>
  );
}

/* ----------------------------------------------------------- metrics --- */

function MetricsTable({
  names,
  values,
  loading,
  fetching,
  error,
  families,
  directions,
  image,
  severity,
  focus,
  onFocus,
  filename,
}: {
  names: string[];
  values: Record<string, MetricValues> | undefined;
  loading: boolean;
  fetching: boolean;
  error: string | null;
  families: Record<string, string>;
  directions: Record<string, boolean | null>;
  image: string;
  severity: number;
  focus: string;
  onFocus: (name: string) => void;
  filename: string;
}) {
  if (!names.length) return null;
  if (error) return <ErrorBanner>{error}</ErrorBanner>;
  if (loading || !values) return <Skeleton className="h-64" />;

  const shown = names.filter((n) => values[n]);
  const metrics = sortByFamily([...new Set(shown.flatMap((n) => Object.keys(values[n])))], families);
  const rows = shown.map((name) => ({ name, ...values[name] }));
  type R = (typeof rows)[number];
  const columns: Column<R>[] = [
    { key: "name", header: "Pipeline", align: "left", value: (r) => r.name, render: (r) => <PipelineName name={r.name} tags={false} /> },
    ...metrics.map((m) => ({
      key: m,
      label: m,
      header: (
        <span className="flex flex-col items-end gap-0.5 leading-tight">
          <span>{metricLabel(m)}</span>
          <span className="text-[9.5px] font-medium tracking-normal text-fg-3/80 normal-case">{FAMILY_LABELS[families[m]] ?? ""}</span>
        </span>
      ),
      title: `${m} · ${directionNote(directions[m])}`,
      best: directions[m] === undefined ? undefined : directions[m],
      value: (r: R) => (r as unknown as Record<string, number | null>)[m],
      render: (r: R) => fmtMetric(m, (r as unknown as Record<string, number | null>)[m]),
    })),
  ];

  return (
    <Card
      title="Every metric at this condition"
      sub={`${image} · severity ${fmt(severity, 2)} · the best value in each column is marked · click a row to put it on the right`}
      tools={<ExportButton columns={columns} rows={rows} filename={filename} />}
      flush
      className={clsx("transition-opacity", fetching && "opacity-70")}
    >
      <DataTable columns={columns} rows={rows} rowKey={(r) => r.name} onRowClick={(row) => onFocus(row.name)} isActive={(row) => row.name === focus} stickyFirst dense />
    </Card>
  );
}
