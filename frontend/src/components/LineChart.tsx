/* Line chart over Recharts: hairline grid, 2px lines, a crosshair tooltip that
 * answers "what are all the values at this x" (best first), a legend that
 * doubles as an emphasis control and reveals the series' range band, and a
 * table twin so no value is reachable only by hovering.
 *
 * Colours are CSS custom properties, so a theme switch repaints without a redraw. */

import clsx from "clsx";
import { Fragment, useMemo, useState, type ReactNode } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import { fmt, isNum } from "../lib/format";
import { byDirection } from "../lib/metrics";
import type { Direction } from "../lib/types";
import { DataTable, ExportButton, type Column } from "./DataTable";
import { Swatch } from "./PipelineName";
import { Card, CardFoot, EmptyState, Segmented } from "./ui";

export interface ChartPoint {
  x: number;
  y: number | null;
  lo?: number | null;
  hi?: number | null;
  n?: number;
}

export interface ChartSeries {
  name: string;
  label: string;
  /** A CSS custom property name such as "--series-1". */
  color: string;
  dash?: string;
  points: ChartPoint[];
}

export interface LineChartProps {
  series: ChartSeries[];
  height?: number;
  xTitle?: string;
  yTitle?: string;
  xFormat?: (x: number) => string;
  yFormat?: (y: number) => string;
  refLines?: { y: number; label: string }[];
  /** An x to shade, such as the condition shown elsewhere on the page. */
  marker?: number | null;
  onPick?: (x: number) => void;
  pickHint?: string;
  /** Orders the tooltip best first. */
  direction?: Direction;
  rangeLabel?: string;
  legend?: boolean;
  ariaLabel?: string;
}

type Datum = Record<string, number | [number, number] | null>;

const rangeKey = (name: string) => `${name}::range`;

export function LineChart({
  series,
  height = 320,
  xTitle,
  yTitle,
  xFormat = (x) => fmt(x),
  yFormat = (y) => fmt(y),
  refLines = [],
  marker = null,
  onPick,
  pickHint,
  direction,
  rangeLabel = "range",
  legend = true,
  ariaLabel,
}: LineChartProps) {
  const [emphasis, setEmphasis] = useState<string | null>(null);

  const { data, xs, hasRange, finite } = useMemo(() => {
    const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b);
    let hasRange = false;
    let finite = false;
    const data: Datum[] = xs.map((x) => {
      const row: Datum = { x };
      for (const s of series) {
        const point = s.points.find((p) => p.x === x);
        row[s.name] = isNum(point?.y) ? point!.y : null;
        if (isNum(point?.y)) finite = true;
        if (isNum(point?.lo) && isNum(point?.hi)) {
          row[rangeKey(s.name)] = [point!.lo!, point!.hi!];
          hasRange = true;
        }
      }
      return row;
    });
    return { data, xs, hasRange, finite };
  }, [series]);

  if (!finite) return <EmptyState title="Nothing to plot" detail="No finite values for this selection." />;

  const span = xs.length > 1 ? xs[xs.length - 1] - xs[0] : 1;
  const pick = onPick
    ? (state: { activeLabel?: string | number; activeIndex?: unknown }) => {
        const i = typeof state?.activeIndex === "number" ? state.activeIndex : Number(state?.activeIndex);
        const x = Number.isInteger(i) && xs[i] !== undefined ? xs[i] : Number(state?.activeLabel);
        if (Number.isFinite(x)) onPick(x);
      }
    : undefined;

  return (
    <div className="flex flex-col gap-3" aria-label={ariaLabel}>
      {legend && series.length >= 2 && (
        <div className="flex flex-wrap gap-x-1 gap-y-0.5" role="list" aria-label="Legend">
          {series.map((s) => (
            <span
              key={s.name}
              role="listitem"
              tabIndex={0}
              title={hasRange ? `Hover to highlight ${s.label} and show its ${rangeLabel}` : `Hover to highlight ${s.label}`}
              onPointerEnter={() => setEmphasis(s.name)}
              onPointerLeave={() => setEmphasis(null)}
              onFocus={() => setEmphasis(s.name)}
              onBlur={() => setEmphasis(null)}
              className={clsx(
                "inline-flex cursor-default items-center gap-1.5 rounded-md px-1.5 py-1 text-xs transition-opacity",
                emphasis && emphasis !== s.name ? "opacity-40" : "text-fg-2",
                emphasis === s.name && "bg-surface-2 text-fg",
              )}
            >
              <Swatch color={s.color} dash={s.dash} />
              {s.label}
            </span>
          ))}
        </div>
      )}
      {yTitle && <div className="-mb-1 text-[11px] font-medium text-fg-3">{yTitle}</div>}
      <div style={{ height }} className={clsx("min-w-0", onPick && "[&_.recharts-surface]:cursor-pointer")} role="img" aria-label={ariaLabel ?? yTitle}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: refLines.length ? 104 : 16, bottom: xTitle ? 22 : 4, left: 0 }} onClick={pick}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis
              dataKey="x"
              type="number"
              domain={[xs[0], xs[xs.length - 1]]}
              ticks={xs.length <= 12 ? xs : undefined}
              tickFormatter={(x: number) => xFormat(x)}
              stroke="var(--axis)"
              tickLine={false}
              axisLine={{ stroke: "var(--border-strong)" }}
              tickMargin={8}
              label={
                xTitle
                  ? { value: xTitle, position: "insideBottom", offset: -16, style: { fill: "var(--text-3)", fontSize: 11 } }
                  : undefined
              }
            />
            <YAxis
              type="number"
              domain={["auto", "auto"]}
              tickFormatter={(y: number) => yFormat(y)}
              stroke="var(--axis)"
              tickLine={false}
              axisLine={false}
              width={56}
              tickMargin={6}
            />
            {marker !== null && marker !== undefined && (
              <ReferenceLine x={marker} stroke="var(--accent)" strokeOpacity={0.12} strokeWidth={14} ifOverflow="hidden" />
            )}
            {refLines.map((ref) => (
              <ReferenceLine
                key={ref.label}
                y={ref.y}
                stroke="var(--border-strong)"
                strokeDasharray="4 4"
                ifOverflow="extendDomain"
                label={{ value: ref.label, position: "right", fill: "var(--text-3)", fontSize: 10.5 }}
              />
            ))}
            {series.map((s) =>
              hasRange && emphasis === s.name ? (
                <Area
                  key={`${s.name}-range`}
                  dataKey={rangeKey(s.name)}
                  stroke="none"
                  fill={`var(${s.color})`}
                  fillOpacity={0.14}
                  isAnimationActive={false}
                  connectNulls={false}
                  activeDot={false}
                  legendType="none"
                />
              ) : null,
            )}
            {series.map((s) => {
              const dim = emphasis !== null && emphasis !== s.name;
              const stroke = `var(${s.color})`;
              // Markers help on a short sweep; on a long one (training epochs) they are noise.
              const dots = xs.length <= 24;
              return (
                <Line
                  key={s.name}
                  dataKey={s.name}
                  name={s.label}
                  type="linear"
                  stroke={stroke}
                  strokeWidth={emphasis === s.name ? 2.75 : 2}
                  strokeDasharray={s.dash}
                  strokeOpacity={dim ? 0.15 : 1}
                  dot={dots ? { r: 2.75, strokeWidth: 0, fill: stroke, fillOpacity: dim ? 0.15 : 1 } : false}
                  activeDot={{ r: 4.5, strokeWidth: 2, stroke: "var(--surface)", fill: stroke }}
                  isAnimationActive={false}
                  connectNulls={false}
                />
              );
            })}
            <Tooltip
              isAnimationActive={false}
              cursor={{ stroke: "var(--border-strong)", strokeWidth: 1 }}
              content={(props: TooltipContentProps) => (
                <ChartTooltip {...props} series={series} xFormat={xFormat} yFormat={yFormat} direction={direction} span={span} pickHint={pickHint} />
              )}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ChartTooltip({
  active,
  label,
  series,
  xFormat,
  yFormat,
  direction,
  pickHint,
}: TooltipContentProps & {
  series: ChartSeries[];
  xFormat: (x: number) => string;
  yFormat: (y: number) => string;
  direction?: Direction;
  span: number;
  pickHint?: string;
}) {
  if (!active || label === undefined) return null;
  const x = Number(label);
  const entries = series
    .map((s) => ({ s, point: s.points.find((p) => p.x === x) }))
    .filter((e) => e.point)
    .sort((a, b) => byDirection(direction ?? true)(a.point!.y, b.point!.y));
  return (
    <div className="min-w-[200px] max-w-[300px] rounded-lg border border-line bg-surface/95 p-2.5 text-xs shadow-pop backdrop-blur">
      <div className="mb-1.5 font-semibold text-fg">{xFormat(x)}</div>
      <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-2 gap-y-1">
        {entries.map(({ s, point }) => (
          <Fragment key={s.name}>
            <Swatch color={s.color} dash={s.dash} />
            <span className="truncate text-fg-2">{s.label}</span>
            <span className="num text-right font-medium text-fg">
              {isNum(point!.y) ? yFormat(point!.y) : "—"}
              {isNum(point!.lo) && isNum(point!.hi) && (
                <span className="ml-1 font-normal text-fg-3">
                  [{yFormat(point!.lo)}–{yFormat(point!.hi)}]
                </span>
              )}
            </span>
          </Fragment>
        ))}
      </div>
      {pickHint && <div className="mt-2 border-t border-line pt-1.5 text-[11px] text-fg-3">{pickHint}</div>}
    </div>
  );
}

/* --------------------------------------------------------------- card --- */

/** A card holding a chart with a chart/table switch and a CSV export. */
export function ChartCard({
  title,
  sub,
  controls,
  foot,
  chart,
  filename,
  loading = false,
  className,
  id,
}: {
  title: ReactNode;
  sub?: ReactNode;
  controls?: ReactNode;
  foot?: ReactNode;
  chart: LineChartProps;
  filename: string;
  loading?: boolean;
  className?: string;
  id?: string;
}) {
  const [view, setView] = useState<"chart" | "table">("chart");
  const xFormat = chart.xFormat ?? ((x: number) => fmt(x));
  const yFormat = chart.yFormat ?? ((y: number) => fmt(y));

  const { rows, columns } = useMemo(() => {
    const xs = [...new Set(chart.series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b);
    const rows = xs.map((x) => {
      const row: Record<string, number | null> = { x };
      for (const s of chart.series) row[s.name] = s.points.find((p) => p.x === x)?.y ?? null;
      return row;
    });
    const columns: Column<Record<string, number | null>>[] = [
      { key: "x", header: chart.xTitle ?? "x", label: chart.xTitle ?? "x", align: "left", value: (r) => r.x, render: (r) => xFormat(r.x!) },
      ...chart.series.map((s) => ({
        key: s.name,
        label: s.label,
        header: (
          <span className="inline-flex items-center gap-1.5 normal-case">
            <Swatch color={s.color} dash={s.dash} />
            {s.label}
          </span>
        ),
        value: (r: Record<string, number | null>) => r[s.name],
        render: (r: Record<string, number | null>) => (isNum(r[s.name]) ? yFormat(r[s.name]!) : "—"),
      })),
    ];
    return { rows, columns };
  }, [chart.series, chart.xTitle]);

  return (
    <Card
      id={id}
      title={title}
      sub={sub}
      className={className}
      tools={
        <>
          <Segmented
            size="sm"
            label="View"
            value={view}
            onChange={setView}
            options={[
              ["chart", "Chart"],
              ["table", "Table"],
            ]}
          />
          <ExportButton columns={columns} rows={rows} filename={filename} />
        </>
      }
    >
      {controls && <div className="mb-4 flex flex-wrap items-end gap-4">{controls}</div>}
      <div className={clsx("transition-opacity", loading && "opacity-50")}>
        {view === "chart" ? (
          <LineChart {...chart} />
        ) : (
          <div className="-mx-4 sm:-mx-5">
            <DataTable columns={columns} rows={rows} rowKey={(r) => String(r.x)} dense />
          </div>
        )}
      </div>
      {foot && <CardFoot>{foot}</CardFoot>}
    </Card>
  );
}
