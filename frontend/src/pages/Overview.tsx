/* Overview: the run at a glance - what hardware was emulated, what each arm
 * recovers on average, which arm holds up best on the worst hardware, and
 * whether the run can be trusted (calibration gate, gaps, provenance). */

import { ArrowRight } from "lucide-react";
import { useMemo, type ReactNode } from "react";
import { Link, useNavigate } from "react-router";
import { DataTable, ExportButton, type Column } from "../components/DataTable";
import { PageHeader } from "../components/Layout";
import { ChartCard } from "../components/LineChart";
import { PipelineName } from "../components/PipelineName";
import { Badge, Callout, Card, CardFoot, Delta, EmptyState, ErrorBanner, Field, KeyValues, Segmented, Select, Skeleton, Stat } from "../components/ui";
import { describeCondition, sortedConditions } from "../lib/conditions";
import { fmt, fmtDate, fmtDuration, fmtPercent, fmtSci, fmtSigned } from "../lib/format";
import { byDirection, defaultMetric, deltaTone, directionNote, fmtMetric, metricGroups, metricLabel, metricUnit, bestIndex } from "../lib/metrics";
import { armPresets, BASELINE, pipelineInfo, PRESET_LABELS, type Preset } from "../lib/pipelines";
import { useAggregate, useRunInfo } from "../lib/queries";
import { useIndexed, useUrlState } from "../lib/run";
import type { AggregateResponse, Condition, RunIndex, RunInfo } from "../lib/types";

export default function Overview() {
  const { index, exp, link } = useIndexed();
  const [params, patch] = useUrlState();
  const navigate = useNavigate();

  const conditions = useMemo(() => sortedConditions(index), [index]);
  const clean = conditions[0];
  const worst = conditions[conditions.length - 1];
  const available = useMemo(() => index.pipelines.filter((p) => p.available).map((p) => p.name), [index]);
  const presets = useMemo(() => armPresets(available), [available]);
  const directions = index.directions ?? {};

  const metric = defaultMetric(index, params.get("metric"));
  const preset: Preset = presets && params.get("arms") && params.get("arms")! in presets ? (params.get("arms") as Preset) : "main";
  const arms = presets ? presets[preset] : available;
  const names = useMemo(() => [...new Set([...(available.includes(BASELINE) ? [BASELINE] : []), ...arms])], [available, arms]);

  const runInfo = useRunInfo(exp);
  const aggregate = useAggregate(exp, metric, names);
  const data = aggregate.data;
  const direction = directions[metric];
  const unit = metricUnit(metric);

  const valueAt = (name: string, condition: Condition) => data?.series[name]?.find((p) => p.hash === condition.hash)?.mean ?? null;

  return (
    <>
      <PageHeader
        title="Overview"
        sub={
          <>
            <b className="font-semibold text-fg">{index.experiment_id}</b> · {index.images.length} scenes × {conditions.length} hardware conditions ×{" "}
            {available.length} pipelines. How much of what weaker camera hardware loses can each restoration pipeline win back?
          </>
        }
        actions={
          <Link
            to={link("/explore", { sev: conditions.length - 1 })}
            className="inline-flex h-9 items-center gap-2 rounded-lg border border-line bg-surface px-3.5 text-[13px] font-medium text-fg shadow-card hover:border-line-strong"
          >
            Inspect the worst hardware <ArrowRight size={15} />
          </Link>
        }
      />

      <div className="flex flex-col gap-5">
        <Kpis
          index={index}
          data={data}
          names={names}
          metric={metric}
          clean={clean}
          worst={worst}
          available={available}
          run={runInfo.data}
          valueAt={valueAt}
        />

        <ChartCard
          title={`${metricLabel(metric)} across the sweep`}
          sub={`${directionNote(direction)} · severity 0 is the clean sensor, 1 the weakest hardware emulated · hover a legend entry to see the range across scenes, click a point to open it in the Explorer`}
          filename={`${index.experiment_id}_${metric}_sweep.csv`}
          loading={aggregate.isFetching}
          controls={
            <>
              <Field label="Metric">
                <Select label="Metric" value={metric} onChange={(value) => patch({ metric: value === defaultMetric(index) ? null : value })} options={metricGroups(index)} className="w-56" />
              </Field>
              {presets && (
                <Field label="Arms">
                  <Segmented label="Arms" value={preset} onChange={(value) => patch({ arms: value === "main" ? null : value })} options={PRESET_LABELS} />
                </Field>
              )}
            </>
          }
          chart={{
            series: data
              ? names.map((name) => {
                  const info = pipelineInfo(name);
                  return {
                    name,
                    label: info.label,
                    color: info.color,
                    dash: info.dash,
                    points: (data.series[name] ?? []).map((p) => ({ x: p.severity, y: p.mean, lo: p.min, hi: p.max, n: p.n })),
                  };
                })
              : [],
            height: 360,
            xTitle: "Degradation severity",
            yTitle: `${metricLabel(metric)}${unit ? ` (${unit})` : ""} · mean of ${data?.n_images ?? index.images.length} scenes`,
            xFormat: (x) => fmt(x, 2),
            yFormat: (y) => fmtMetric(metric, y),
            direction,
            rangeLabel: "min–max across scenes",
            onPick: (severity) => navigate(link("/explore", { sev: conditions.findIndex((c) => c.severity === severity), metric })),
            pickHint: "Click to open this condition in the Explorer",
            ariaLabel: `${metricLabel(metric)} across the degradation sweep`,
          }}
        />
        {aggregate.error && <ErrorBanner>{String((aggregate.error as Error).message)}</ErrorBanner>}

        <div className="grid gap-5 xl:grid-cols-2">
          <Leaderboard data={data} names={names} metric={metric} index={index} clean={clean} worst={worst} valueAt={valueAt} />
          <SweepTable conditions={conditions} />
        </div>

        <div className="grid gap-5 xl:grid-cols-2">
          <Calibration run={runInfo.data} loading={runInfo.isLoading} />
          <Provenance run={runInfo.data} index={index} loading={runInfo.isLoading} />
        </div>

        <Gaps run={runInfo.data} />
      </div>
    </>
  );
}

/* ------------------------------------------------------------- KPIs --- */

function Kpis({
  index,
  data,
  names,
  metric,
  clean,
  worst,
  available,
  run,
  valueAt,
}: {
  index: RunIndex;
  data: AggregateResponse | undefined;
  names: string[];
  metric: string;
  clean: Condition;
  worst: Condition;
  available: string[];
  run: RunInfo | undefined;
  valueAt: (name: string, condition: Condition) => number | null;
}) {
  if (!data) {
    return (
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
        {Array.from({ length: 5 }, (_, i) => (
          <Skeleton key={i} className="h-[118px]" />
        ))}
      </div>
    );
  }
  const direction = index.directions[metric];
  const unit = metricUnit(metric);
  const contenders = names.filter((name) => name !== BASELINE || names.length === 1);
  const worstValues = contenders.map((name) => valueAt(name, worst));
  const bestAt = bestIndex(worstValues, direction);
  const bestName = bestAt >= 0 ? contenders[bestAt] : null;
  const baseClean = valueAt(BASELINE, clean);
  const baseWorst = valueAt(BASELINE, worst);
  const lost = baseClean !== null && baseWorst !== null ? baseWorst - baseClean : null;
  const passed = run?.validation?.passed;
  const total = index.pipelines.length;

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
      <div className="sm:col-span-2 xl:col-span-1">
        <Stat
          hero
          label={`Best ${metricLabel(metric)} on the worst hardware`}
          value={bestName ? fmtMetric(metric, worstValues[bestAt]) : "—"}
          unit={unit}
          note={
            bestName ? (
              <span className="flex flex-col gap-1">
                <span className="text-fg">
                  <PipelineName name={bestName} />
                </span>
                <span>
                  severity {fmt(worst.severity, 2)} · {directionNote(direction)}
                </span>
              </span>
            ) : (
              "no finite values"
            )
          }
        />
      </div>
      <Stat
        label={`${metricLabel(metric)} lost to hardware`}
        value={lost === null ? "—" : fmtSigned(lost)}
        unit={unit}
        note={
          available.includes(BASELINE)
            ? `baseline, clean → worst (${fmtMetric(metric, baseClean)} → ${fmtMetric(metric, baseWorst)})`
            : "no baseline arm in this run"
        }
      />
      <Stat
        label="Hardware sweep"
        value={index.conditions.length}
        unit="conditions"
        note={`severity ${fmt(clean.severity, 2)} → ${fmt(worst.severity, 2)} · ${index.images.length} scenes`}
      />
      <Stat
        label="Pipelines that ran"
        value={available.length}
        unit={`of ${total}`}
        note={available.length === total ? "the whole bank ran" : `${total - available.length} recorded as gaps`}
        tone={available.length === total ? null : "warn"}
      />
      <Stat
        label="Calibration gate"
        value={
          <span className="text-base">
            {passed === undefined || passed === null ? (
              <Badge tone="neutral">not recorded</Badge>
            ) : passed ? (
              <Badge tone="good">Passed</Badge>
            ) : (
              <Badge tone="bad">Failed</Badge>
            )}
          </span>
        }
        note={run?.validation ? `${run.validation.checks?.length ?? 0} checks · noise model vs measurement` : "no validation.json in this run"}
      />
    </div>
  );
}

/* ------------------------------------------------------- leaderboard --- */

interface LeaderRow {
  name: string;
  clean: number | null;
  worst: number | null;
  vsBase: number | null;
}

function Leaderboard({
  data,
  names,
  metric,
  index,
  clean,
  worst,
  valueAt,
}: {
  data: AggregateResponse | undefined;
  names: string[];
  metric: string;
  index: RunIndex;
  clean: Condition;
  worst: Condition;
  valueAt: (name: string, condition: Condition) => number | null;
}) {
  const navigate = useNavigate();
  const { link } = useIndexed();
  const direction = index.directions[metric];
  const conditions = index.conditions.length;

  const rows: LeaderRow[] = useMemo(() => {
    if (!data) return [];
    const baseWorst = valueAt(BASELINE, worst);
    const rows = names.map((name) => {
      const w = valueAt(name, worst);
      return {
        name,
        clean: valueAt(name, clean),
        worst: w,
        vsBase: baseWorst !== null && w !== null && name !== BASELINE ? w - baseWorst : null,
      };
    });
    return rows.sort((a, b) => byDirection(direction)(a.worst, b.worst));
  }, [data, names, clean, worst, direction]);

  const columns: Column<LeaderRow>[] = [
    { key: "rank", header: "#", label: "rank", align: "right", sortable: false, value: (r) => rows.indexOf(r) + 1, render: (r) => <span className="text-fg-3">{rows.indexOf(r) + 1}</span> },
    { key: "name", header: "Pipeline", align: "left", value: (r) => r.name, render: (r) => <PipelineName name={r.name} /> },
    { key: "clean", header: "Clean", value: (r) => r.clean, render: (r) => fmtMetric(metric, r.clean), best: direction },
    { key: "worst", header: "Worst", value: (r) => r.worst, render: (r) => fmtMetric(metric, r.worst), best: direction },
    {
      key: "vsBase",
      header: "vs baseline",
      title: "difference from the baseline on the same hardware",
      value: (r) => r.vsBase,
      render: (r) => (
        <Delta value={r.vsBase} tone={deltaTone(r.vsBase, direction)}>
          {fmtSigned(r.vsBase)}
        </Delta>
      ),
    },
  ];

  return (
    <Card
      title="Worst-hardware leaderboard"
      sub={`${metricLabel(metric)} at severity ${fmt(worst.severity, 2)}, mean over scenes · click a row to inspect it`}
      tools={<ExportButton columns={columns} rows={rows} filename={`${index.experiment_id}_${metric}_leaderboard.csv`} />}
      flush
    >
      {data ? (
        <DataTable
          columns={columns}
          rows={rows}
          rowKey={(r) => r.name}
          onRowClick={(row) => navigate(link("/explore", { sev: conditions - 1, right: row.name, metric }))}
        />
      ) : (
        <div className="p-5">
          <Skeleton className="h-64" />
        </div>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------- sweep --- */

function SweepTable({ conditions }: { conditions: Condition[] }) {
  const navigate = useNavigate();
  const { link } = useIndexed();
  const rows = conditions.map((c, i) => ({ ...c.params, severity: c.severity, i, c }));
  type R = (typeof rows)[number];
  const last = conditions[conditions.length - 1];
  const columns: Column<R>[] = [
    { key: "severity", header: "Severity", align: "left", value: (r) => r.severity, render: (r) => <span className="num font-mono text-[12.5px] text-fg">{fmt(r.severity, 2)}</span> },
    { key: "alpha", header: "Light", title: "photon scale α: aperture, sensor area, exposure", value: (r) => r.alpha, render: (r) => (r.alpha === undefined ? "—" : fmtPercent(r.alpha)) },
    { key: "sigma_r_dn", header: "Read noise", title: "read noise σ_r in DN", value: (r) => r.sigma_r_dn, render: (r) => (r.sigma_r_dn === undefined ? "—" : `${fmt(r.sigma_r_dn, 1)} DN`) },
    { key: "r_psf", header: "Blur", title: "PSF radius in pixels", value: (r) => r.r_psf, render: (r) => (r.r_psf === undefined ? "—" : `${fmt(r.r_psf, 2)} px`) },
    { key: "binning", header: "Binning", value: (r) => r.binning, render: (r) => (r.binning === undefined ? "—" : `${r.binning}×`) },
    { key: "bit_depth", header: "ADC", value: (r) => r.bit_depth, render: (r) => (r.bit_depth === undefined ? "—" : `${r.bit_depth}-bit`) },
  ];
  return (
    <Card
      title="The hardware sweep"
      sub="What each severity emulates, applied to linear RAW before demosaicing · click a row to view it"
      flush
    >
      <DataTable columns={columns} rows={rows} rowKey={(r) => r.c.hash} onRowClick={(row) => navigate(link("/explore", { sev: row.i }))} />
      <div className="px-4 pb-4 sm:px-5">
        <CardFoot>
          Severity {fmt(last.severity, 2)}: {describeCondition(last)}.
        </CardFoot>
      </div>
    </Card>
  );
}

/* ------------------------------------------------------- calibration --- */

function Calibration({ run, loading }: { run: RunInfo | undefined; loading: boolean }) {
  if (loading) return <Skeleton className="h-64" />;
  const validation = run?.validation;
  if (!validation) {
    return (
      <Card title="Calibration gate">
        <EmptyState title="No calibration record" detail="This run wrote no validation.json." command="python -m sensorsim.cli validate configs/experiments/pilot.yaml" />
      </Card>
    );
  }
  type Check = NonNullable<typeof validation.checks>[number];
  const columns: Column<Check>[] = [
    { key: "name", header: "Check", align: "left", value: (c) => c.name, render: (c) => <span title={c.detail ?? ""}>{c.name.replaceAll("_", " ")}</span> },
    { key: "measured", header: "Measured", value: (c) => c.measured, render: (c) => fmtSci(c.measured) },
    { key: "synthetic", header: "Model", value: (c) => c.synthetic, render: (c) => fmtSci(c.synthetic) },
    { key: "relative_error", header: "Error", value: (c) => c.relative_error, render: (c) => (c.relative_error === undefined ? "—" : `${fmt(c.relative_error * 100, 2)}%`) },
    { key: "tolerance", header: "Tolerance", value: (c) => c.tolerance, render: (c) => (c.tolerance === undefined ? "—" : `${fmt(c.tolerance * 100, 0)}%`) },
    { key: "passed", header: "", align: "right", sortable: false, render: (c) => (c.passed ? <Badge tone="good">pass</Badge> : <Badge tone="bad">fail</Badge>) },
  ];
  const mode = run?.manifest?.stages?.validation?.mode;
  const stageNote = run?.manifest?.stages?.validation?.note;
  const note = typeof stageNote === "string" ? stageNote : null;
  return (
    <Card
      title="Calibration gate"
      sub="The degrader's noise model checked against measurement before any experiment ran"
      tools={validation.passed ? <Badge tone="good">Passed</Badge> : <Badge tone="bad">Failed</Badge>}
      flush
    >
      {validation.checks?.length ? <DataTable columns={columns} rows={validation.checks} rowKey={(c) => c.name} /> : null}
      {(validation.notes?.length || note) && (
        <div className="flex flex-col gap-2 px-4 pb-4 pt-3 sm:px-5">
          {note && (
            <Callout tone={mode === "self" ? "warn" : "neutral"}>
              {mode === "self" && <Badge tone="warn">self-consistency</Badge>}
              <span>{note}</span>
            </Callout>
          )}
          {validation.notes?.map((text) => (
            <p key={text} className="text-xs leading-relaxed text-fg-3">
              {text}
            </p>
          ))}
        </div>
      )}
    </Card>
  );
}

/* -------------------------------------------------------- provenance --- */

function Provenance({ run, index, loading }: { run: RunInfo | undefined; index: RunIndex; loading: boolean }) {
  if (loading) return <Skeleton className="h-64" />;
  const m = run?.manifest;
  if (!m) {
    return (
      <Card title="Provenance">
        <EmptyState title="No manifest" detail="This run wrote no manifest.json." />
      </Card>
    );
  }
  const started = m.started_at ? new Date(m.started_at) : null;
  const finished = m.finished_at ? new Date(m.finished_at) : null;
  const seconds = started && finished ? (finished.getTime() - started.getTime()) / 1000 : null;
  const hw = m.hardware_info ?? {};
  const pk = m.packages ?? {};
  const mono = (text: string | number | undefined) => <code className="text-[12px]">{text}</code>;
  const rows: [string, ReactNode][] = [
    ["Experiment", index.experiment_id],
    ["Config hash", m.config_hash ? mono(m.config_hash) : null],
    ["Seed", m.seed !== undefined ? mono(m.seed) : null],
    ["Finished", finished ? `${fmtDate(finished)}${seconds !== null ? ` · took ${fmtDuration(seconds)}` : ""}` : null],
    ["Git commit", m.git_commit ? mono(m.git_commit) : null],
    ["GPU", hw.gpu ?? (hw.cuda ? `CUDA ${hw.cuda}` : "none")],
    ["Platform", hw.platform],
    ["Python", pk.python],
    [
      "Key packages",
      ["numpy", "torch", "cv2", "pandas", "scipy"]
        .filter((k) => pk[k] && pk[k] !== "absent")
        .map((k) => `${k} ${pk[k]}`)
        .join(" · "),
    ],
  ].filter(([, v]) => v !== null && v !== undefined && v !== "") as [string, ReactNode][];
  return (
    <Card title="Provenance" sub={<>Everything needed to rerun this exactly: <code>sensorsim reproduce &lt;config&gt;</code></>}>
      <KeyValues rows={rows} />
    </Card>
  );
}

/* -------------------------------------------------------------- gaps --- */

function Gaps({ run }: { run: RunInfo | undefined }) {
  const gaps = run?.gaps ?? [];
  if (!gaps.length) return null;
  return (
    <Card
      title="Gaps in the bank"
      sub="A pipeline that could not run is recorded, not omitted: an absent row would read as “not measured yet”; these read as “measured, and it could not run”."
      className="border-warn/30"
    >
      <ul className="flex flex-col divide-y divide-line">
        {gaps.map((gap) => (
          <li key={gap.pipeline} className="flex flex-col gap-1 py-2.5 first:pt-0 last:pb-0 sm:flex-row sm:items-center sm:gap-4">
            <span className="shrink-0 text-[13px] text-fg">
              <PipelineName name={gap.pipeline} showId />
            </span>
            <span className="min-w-0 flex-1 text-[13px] text-fg-2">{gap.reason.replace(/^unavailable:\s*/, "")}</span>
            <span className="num shrink-0 text-xs text-fg-3">{gap.rows} rows</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}
