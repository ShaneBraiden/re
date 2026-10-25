/* Analysis: what `sensorsim analyze` and `sensorsim paper-nag` concluded from
 * this run - recovery curves, knees, significance, the NAG tables, training
 * curves, the noise-model check - plus every figure they rendered. Only the
 * sections whose tables exist are shown; nothing here is recomputed. */

import clsx from "clsx";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { DataTable, ExportButton, type Column } from "../components/DataTable";
import { PageHeader } from "../components/Layout";
import { Lightbox } from "../components/Lightbox";
import { ChartCard } from "../components/LineChart";
import { PipelineName } from "../components/PipelineName";
import { Badge, Callout, Card, EmptyState, ErrorBanner, Field, Segmented, Select, Skeleton } from "../components/ui";
import { fmt, fmtP, fmtPercent, fmtSigned, isNum } from "../lib/format";
import { directionNote, metricLabel, metricUnit } from "../lib/metrics";
import { armPresets, BASELINE, pipelineInfo, PRESET_LABELS, type Preset } from "../lib/pipelines";
import { useAnalysis } from "../lib/queries";
import { useIndexed } from "../lib/run";
import type { AnalysisResponse, Direction, Divergence, Row } from "../lib/types";

const MEAN_COLUMNS = ["psnr", "ssim", "delta_e", "nr_quality", "sharpness", "noise_sigma", "mtf50_edge", "sift_inlier_ratio"];

const num = (row: Row, key: string) => (typeof row[key] === "number" ? (row[key] as number) : null);
const str = (row: Row, key: string) => (row[key] === null || row[key] === undefined ? "" : String(row[key]));

export default function Analysis() {
  const { index, exp } = useIndexed();
  const analysis = useAnalysis(exp);

  const header = (
    <PageHeader
      title="Analysis"
      sub={
        <>
          <b className="font-semibold text-fg">{index.experiment_id}</b> · recovery, knees, significance and the figures the paper is built from.
        </>
      }
    />
  );

  if (analysis.isLoading) {
    return (
      <>
        {header}
        <div className="flex flex-col gap-5">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-[420px]" />
          <Skeleton className="h-64" />
        </div>
      </>
    );
  }
  if (analysis.error) {
    return (
      <>
        {header}
        <ErrorBanner>{String((analysis.error as Error).message)}</ErrorBanner>
      </>
    );
  }
  const data = analysis.data!;
  if (!data.figures_dir) {
    return (
      <>
        {header}
        <Card>
          <EmptyState
            title="This run has not been analysed yet"
            detail="The analysis stage writes its tables and figures next to the paper. Run it, then reload this page."
            command={`python -m sensorsim.cli analyze experiments/${exp}/metrics.csv --figures paper/figures/${exp}`}
          />
        </Card>
      </>
    );
  }
  return (
    <>
      {header}
      <Sections data={data} directions={index.directions ?? {}} />
    </>
  );
}

function Sections({ data, directions }: { data: AnalysisResponse; directions: Record<string, Direction> }) {
  const t = data.tables;
  const sections: [string, string, (() => ReactNode) | null][] = [
    ["recovery", "Recovery", t.recovery_aggregate?.length ? () => <RecoverySection rows={t.recovery_aggregate!} /> : null],
    ["knees", "Knees", t.knees?.length ? () => <KneesSection rows={t.knees!} /> : null],
    ["significance", "Significance", t.significance?.length ? () => <SignificanceSection rows={t.significance!} /> : null],
    ["planned", "Planned tests", t.planned_tests?.length ? () => <PlannedSection rows={t.planned_tests!} /> : null],
    [
      "arms",
      "Arm means",
      t.main_means?.length || t.ablation_means?.length ? () => <MeansSection main={t.main_means} ablation={t.ablation_means} directions={directions} /> : null,
    ],
    ["divergence", "Divergence", data.divergence ? () => <DivergenceSection divergence={data.divergence!} /> : null],
    ["training", "Training", t.nag_training?.length ? () => <TrainingSection rows={t.nag_training!} /> : null],
    [
      "noise",
      "Noise model",
      t.noise_validation?.length || t.photon_transfer?.length ? () => <NoiseSection validation={t.noise_validation} photon={t.photon_transfer} /> : null,
    ],
    ["figures", "Figures", data.figures.length ? () => <FiguresSection figures={data.figures} /> : null],
  ];
  const shown = sections.filter(([, , build]) => build) as [string, string, () => ReactNode][];
  const current = useScrollSpy(shown.map(([id]) => `sec-${id}`));

  return (
    <div className="flex flex-col gap-5">
      <p className="text-[13px] text-fg-3">
        Read from <code className="text-fg-2">{data.figures_dir}</code>; regenerate with <code className="text-fg-2">sensorsim analyze</code>.
      </p>
      <nav
        aria-label="Sections"
        className="sticky top-14 z-20 -mx-4 overflow-x-auto border-y border-line bg-bg/90 px-4 py-2 backdrop-blur scroll-thin sm:-mx-6 sm:px-6 lg:top-0 lg:-mx-8 lg:px-8"
      >
        <div className="flex gap-1">
          {shown.map(([id, label]) => (
            <a
              key={id}
              href={`#sec-${id}`}
              onClick={(event) => {
                event.preventDefault();
                document.getElementById(`sec-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
              }}
              className={clsx(
                "whitespace-nowrap rounded-md px-2.5 py-1.5 text-[13px] font-medium transition-colors",
                current === `sec-${id}` ? "bg-accent-soft text-accent" : "text-fg-2 hover:bg-surface-2 hover:text-fg",
              )}
            >
              {label}
            </a>
          ))}
        </div>
      </nav>
      {shown.map(([id, , build]) => (
        <div key={id} id={`sec-${id}`} className="scroll-mt-32 lg:scroll-mt-20">
          {build()}
        </div>
      ))}
    </div>
  );
}

/** The id of the section nearest the top of the viewport. */
function useScrollSpy(ids: string[]) {
  const [current, setCurrent] = useState<string | null>(ids[0] ?? null);
  const key = ids.join(",");
  useEffect(() => {
    const onScroll = () => {
      let best: string | null = ids[0] ?? null;
      for (const id of ids) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top < 160) best = id;
      }
      setCurrent(best);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [key]);
  return current;
}

/* --------------------------------------------------------------- recovery --- */

function RecoverySection({ rows }: { rows: Row[] }) {
  const metrics = useMemo(() => [...new Set(rows.map((r) => str(r, "metric_name")))], [rows]);
  const [metric, setMetric] = useState(metrics.includes("psnr") ? "psnr" : metrics[0]);
  const names = useMemo(() => [...new Set(rows.map((r) => str(r, "pipeline")))].filter((n) => n !== BASELINE), [rows]);
  const presets = armPresets(names);
  const [preset, setPreset] = useState<Preset>("main");
  const shown = presets ? presets[preset].filter((n) => n !== BASELINE) : names;

  const metricName = (m: string) => metricLabel(m.replace(/_error$/, "")) + (m.endsWith("_error") ? " (distance to target)" : "");

  return (
    <ChartCard
      title="Recovery ratio across the sweep"
      sub="R = (arm − baseline on degraded) / (baseline on clean − baseline on degraded): the share of the hardware loss an arm wins back. Hover a legend entry for its 95% interval."
      filename={`recovery_${metric}.csv`}
      controls={
        <>
          <Field label="Metric">
            <Select label="Recovery metric" value={metric} onChange={setMetric} options={metrics.map((m) => [m, metricName(m)])} className="w-60" />
          </Field>
          {presets && (
            <Field label="Arms">
              <Segmented label="Arms" value={preset} onChange={setPreset} options={PRESET_LABELS} />
            </Field>
          )}
        </>
      }
      foot="Low severities are dropped where the hardware loss is under 5% of the sweep's largest: there the ratio divides a small difference by a smaller one."
      chart={{
        series: shown
          .map((name) => {
            const info = pipelineInfo(name);
            return {
              name,
              label: info.label,
              color: info.color,
              dash: info.dash,
              points: rows
                .filter((r) => r.pipeline === name && r.metric_name === metric)
                .map((r) => ({ x: num(r, "severity")!, y: num(r, "median"), lo: num(r, "ci_low"), hi: num(r, "ci_high"), n: num(r, "n") ?? undefined })),
            };
          })
          .filter((s) => s.points.length),
        height: 380,
        xTitle: "Degradation severity",
        yTitle: `Recovery ratio R · ${metricLabel(metric)} · median over scenes`,
        xFormat: (x) => fmt(x, 2),
        yFormat: (y) => fmt(y, 2),
        refLines: [
          { y: 0, label: "R = 0: baseline" },
          { y: 1, label: "R = 1: clean" },
        ],
        direction: true,
        rangeLabel: "95% interval",
        ariaLabel: `Recovery ratio for ${metricLabel(metric)}`,
      }}
    />
  );
}

/* ------------------------------------------------------------------ knees --- */

function TableCard({
  title,
  sub,
  rows,
  columns,
  filename,
  tools,
  rowKey,
  maxHeight,
}: {
  title: string;
  sub: ReactNode;
  rows: Row[];
  columns: Column<Row>[];
  filename: string;
  tools?: ReactNode;
  rowKey?: (row: Row, i: number) => string;
  maxHeight?: number;
}) {
  return (
    <Card
      title={title}
      sub={sub}
      flush
      tools={
        <>
          {tools}
          <ExportButton columns={columns} rows={rows} filename={filename} />
        </>
      }
    >
      <DataTable columns={columns} rows={rows} rowKey={rowKey ?? ((_, i) => String(i))} maxHeight={maxHeight} />
    </Card>
  );
}

const pipelineCol = (key: string, header = "Pipeline"): Column<Row> => ({
  key,
  header,
  align: "left",
  value: (r) => str(r, key),
  render: (r) => <PipelineName name={str(r, key)} />,
});

const numCol = (key: string, header: string, digits: number | null = null, extra: Partial<Column<Row>> = {}): Column<Row> => ({
  key,
  header,
  value: (r) => num(r, key),
  render: (r) => fmt(num(r, key), digits),
  ...extra,
});

function KneesSection({ rows }: { rows: Row[] }) {
  return (
    <TableCard
      title="Where each arm stops keeping up"
      sub="The knee of each recovery curve, located three independent ways. When the methods disagree, the knee is reported as not identifiable rather than averaged into a number."
      rows={rows}
      filename="knees.csv"
      columns={[
        pipelineCol("pipeline"),
        { key: "metric_name", header: "Metric", align: "left", value: (r) => str(r, "metric_name"), render: (r) => metricLabel(str(r, "metric_name")) },
        numCol("knee", "Knee", 2),
        numCol("knee_kneedle", "Kneedle", 2),
        numCol("knee_curvature", "Curvature", 2),
        numCol("knee_piecewise", "Piecewise", 2),
        numCol("agreement", "Spread", 2, { title: "how far apart the three methods land" }),
        {
          key: "identifiable",
          header: "Verdict",
          align: "left",
          value: (r) => (r.identifiable ? 1 : 0),
          render: (r) => (r.identifiable ? <Badge tone="good">identifiable</Badge> : <Badge tone="warn">not identifiable</Badge>),
        },
        {
          key: "curve_model",
          header: "Fit",
          align: "left",
          value: (r) => str(r, "curve_model"),
          render: (r) => (
            <span title={str(r, "notes")}>
              {str(r, "curve_model") || "—"}
              {num(r, "curve_r2") !== null && <span className="text-fg-3"> R² {fmt(num(r, "curve_r2"), 3)}</span>}
            </span>
          ),
        },
      ]}
    />
  );
}

/* ----------------------------------------------------------- significance --- */

const sigBadge = (r: Row) => (r.significant ? <Badge tone="good">significant</Badge> : <Badge tone="neutral">n.s.</Badge>);

function SignificanceSection({ rows }: { rows: Row[] }) {
  return (
    <TableCard
      title="Pairwise significance"
      sub="Paired tests over every scene × condition, p adjusted for multiple comparisons."
      rows={rows}
      filename="significance.csv"
      columns={[
        { key: "metric_name", header: "Metric", align: "left", value: (r) => str(r, "metric_name"), render: (r) => metricLabel(str(r, "metric_name")) },
        {
          key: "pair",
          header: "Comparison",
          align: "left",
          value: (r) => `${str(r, "pipeline_a")} vs ${str(r, "pipeline_b")}`,
          render: (r) => (
            <span className="inline-flex items-center gap-2">
              <PipelineName name={str(r, "pipeline_a")} tags={false} />
              <span className="text-fg-3">vs</span>
              <PipelineName name={str(r, "pipeline_b")} tags={false} />
            </span>
          ),
        },
        numCol("n_pairs", "n", 0),
        { key: "mean_difference", header: "Mean diff.", title: "A − B", value: (r) => num(r, "mean_difference"), render: (r) => fmtSigned(num(r, "mean_difference")) },
        { key: "effect_size", header: "Effect", title: "rank-biserial effect size", value: (r) => num(r, "effect_size"), render: (r) => fmtSigned(num(r, "effect_size"), 2) },
        { key: "p_adjusted", header: "p (adj.)", value: (r) => num(r, "p_adjusted"), best: false, render: (r) => fmtP(num(r, "p_adjusted")) },
        { key: "significant", header: "", align: "left", sortable: false, render: sigBadge },
      ]}
    />
  );
}

function PlannedSection({ rows }: { rows: Row[] }) {
  const metrics = useMemo(() => [...new Set(rows.map((r) => str(r, "metric")))], [rows]);
  const [metric, setMetric] = useState(metrics.includes("psnr") ? "psnr" : metrics[0]);
  const shown = rows.filter((r) => r.metric === metric);
  const sig = shown.filter((r) => r.significant).length;
  return (
    <TableCard
      title="Planned comparisons"
      sub={
        <>
          Each proposed component against the arm it extends, Wilcoxon signed-rank over scenes × conditions, Holm-corrected.{" "}
          <b className="font-medium text-fg-2">
            {sig} of {shown.length} significant
          </b>{" "}
          for {metricLabel(metric)}.
        </>
      }
      rows={shown}
      filename={`planned_tests_${metric}.csv`}
      tools={<Select label="Metric" value={metric} onChange={setMetric} options={metrics.map((m) => [m, metricLabel(m)])} className="w-44" />}
      columns={[
        pipelineCol("arm", "Arm"),
        pipelineCol("against", "Against"),
        numCol("n", "n", 0),
        { key: "mean_delta", header: "Mean Δ", value: (r) => num(r, "mean_delta"), render: (r) => fmtSigned(num(r, "mean_delta")) },
        { key: "median_delta", header: "Median Δ", value: (r) => num(r, "median_delta"), render: (r) => fmtSigned(num(r, "median_delta")) },
        {
          key: "wins",
          header: "Wins",
          title: "pairs where the arm beat its comparator",
          value: (r) => num(r, "wins"),
          render: (r) => <WinBar wins={num(r, "wins")} n={num(r, "n")} />,
        },
        { key: "p_holm", header: "p (Holm)", value: (r) => num(r, "p_holm"), render: (r) => fmtP(num(r, "p_holm")) },
        { key: "significant", header: "", align: "left", sortable: false, render: sigBadge },
      ]}
    />
  );
}

function WinBar({ wins, n }: { wins: number | null; n: number | null }) {
  if (wins === null || !n) return <>—</>;
  const share = wins / n;
  return (
    <span className="inline-flex items-center gap-2">
      <span className="relative h-1.5 w-14 overflow-hidden rounded-full bg-surface-3" aria-hidden>
        <span className={clsx("absolute inset-y-0 left-0 rounded-full", share >= 0.5 ? "bg-good" : "bg-bad")} style={{ width: `${share * 100}%` }} />
      </span>
      <span>
        {fmt(wins, 0)}/{fmt(n, 0)}
      </span>
    </span>
  );
}

/* ------------------------------------------------------------- arm means --- */

function MeansSection({ main, ablation, directions }: { main?: Row[]; ablation?: Row[]; directions: Record<string, Direction> }) {
  const sets = ([["main", "Main arms", main], ["ablation", "Ablations", ablation]] as [string, string, Row[] | undefined][]).filter(
    ([, , rows]) => rows?.length,
  ) as [string, string, Row[]][];
  const [which, setWhich] = useState(sets[0][0]);
  const rows = sets.find(([key]) => key === which)![2];
  const metrics = MEAN_COLUMNS.filter((m) => m in rows[0]);
  return (
    <TableCard
      title="Arm means over the whole sweep"
      sub="Each value averages every scene and every condition. The best value in each column is marked."
      rows={rows}
      rowKey={(r) => str(r, "pipeline")}
      filename={`${which}_means.csv`}
      tools={sets.length > 1 ? <Segmented size="sm" label="Set" value={which} onChange={setWhich} options={sets.map(([key, label]) => [key, label])} /> : null}
      columns={[
        pipelineCol("pipeline"),
        ...metrics.map((m) => ({
          key: m,
          header: metricLabel(m),
          title: `${m}${metricUnit(m) ? ` (${metricUnit(m)})` : ""} · ${directionNote(directions[m] ?? true)}`,
          best: directions[m] === undefined ? undefined : directions[m],
          value: (r: Row) => num(r, m),
          render: (r: Row) => fmt(num(r, m)),
        })),
      ]}
    />
  );
}

/* ------------------------------------------------------------- divergence --- */

function DivergenceSection({ divergence }: { divergence: Divergence }) {
  const pooled = divergence.pooled;
  const rows = Object.values(divergence.per_pipeline ?? {}) as unknown as Row[];
  return (
    <Card
      title="Fidelity vs perception"
      sub={`Does ${metricLabel(pooled?.fidelity_metric ?? "psnr")} move with ${metricLabel(pooled?.perceptual_metric ?? "nr_quality")}? Spearman ρ across scenes × conditions; a significant negative ρ means an arm trades one for the other.`}
      flush
    >
      {pooled && (
        <div className="px-4 pb-3 sm:px-5">
          <Callout tone={pooled.diverges ? "warn" : "good"}>
            {pooled.diverges ? <Badge tone="warn">diverges</Badge> : <Badge tone="good">no divergence</Badge>}
            <span>
              All arms pooled: ρ = {fmt(pooled.spearman_rho, 2)}, p = {fmtP(pooled.p_value)}, n = {fmt(pooled.n, 0)}. {pooled.note ?? ""}
            </span>
          </Callout>
        </div>
      )}
      <DataTable
        rows={rows}
        rowKey={(r) => str(r, "pipeline")}
        columns={[
          pipelineCol("pipeline"),
          numCol("n", "n", 0),
          { key: "spearman_rho", header: "ρ", value: (r) => num(r, "spearman_rho"), render: (r) => <RhoBar rho={num(r, "spearman_rho")} /> },
          { key: "p_value", header: "p", value: (r) => num(r, "p_value"), render: (r) => fmtP(num(r, "p_value")) },
          { key: "diverges", header: "", align: "left", sortable: false, render: (r) => (r.diverges ? <Badge tone="warn">diverges</Badge> : <Badge tone="neutral">no</Badge>) },
        ]}
      />
    </Card>
  );
}

function RhoBar({ rho }: { rho: number | null }) {
  if (!isNum(rho)) return <>—</>;
  const width = Math.min(50, Math.abs(rho) * 50);
  return (
    <span className="inline-flex items-center gap-2.5">
      <span className="relative h-1.5 w-24 rounded-full bg-surface-3" aria-hidden>
        <span className="absolute inset-y-[-2px] left-1/2 w-px bg-line-strong" />
        <span
          className={clsx("absolute inset-y-0 rounded-full", rho >= 0 ? "bg-accent" : "bg-warn")}
          style={{ left: rho >= 0 ? "50%" : `${50 - width}%`, width: `${width}%` }}
        />
      </span>
      <span className="w-10 text-right">{fmtSigned(rho, 2)}</span>
    </span>
  );
}

/* --------------------------------------------------------------- training --- */

const TERM_LABELS: Record<string, string> = { total: "Total", noise: "Noise", exp: "Exposure", col: "Colour", spa: "Spatial", tv: "Smoothness" };

function TrainingSection({ rows }: { rows: Row[] }) {
  const terms = ["total", "noise", "exp", "col", "spa", "tv"].filter((k) => k in rows[0]);
  const [term, setTerm] = useState("total");
  const variants = [...new Set(rows.map((r) => str(r, "variant")))];
  const lengths = variants.map((v) => rows.filter((r) => r.variant === v).length);
  const stopped = variants.filter((_, i) => lengths[i] < Math.max(...lengths));
  return (
    <ChartCard
      title="Training curves"
      sub="Zero-reference training of NC-DCE and its ablations, one point per epoch."
      filename={`training_${term}.csv`}
      controls={
        <Field label="Loss term">
          <Segmented label="Loss term" value={term} onChange={setTerm} options={terms.map((k) => [k, TERM_LABELS[k] ?? k])} />
        </Field>
      }
      foot={stopped.length ? `Stopped early: ${stopped.map((v) => pipelineInfo(v).label).join(", ")}. See the training record in models/.` : undefined}
      chart={{
        series: variants.map((variant) => {
          const info = pipelineInfo(variant);
          return {
            name: variant,
            label: info.label,
            color: info.color,
            dash: info.dash,
            points: rows.filter((r) => r.variant === variant).map((r) => ({ x: num(r, "epoch")!, y: num(r, term) })),
          };
        }),
        height: 320,
        xTitle: "Epoch",
        yTitle: `${TERM_LABELS[term] ?? term} loss`,
        xFormat: (x) => fmt(x, 0),
        yFormat: (y) => fmt(y),
        direction: false,
      }}
    />
  );
}

/* ------------------------------------------------------------ noise model --- */

function NoiseSection({ validation, photon }: { validation?: Row[]; photon?: Row[] }) {
  const ratios = (validation ?? []).map((r) => num(r, "ratio")).filter(isNum).sort((a, b) => a - b);
  const median = ratios.length ? ratios[Math.floor(ratios.length / 2)] : null;
  return (
    <Card
      title="Noise model check"
      sub="Noise σ measured on degraded flat fields against the Poisson–Gaussian prediction, var = μ/g + σ_read². A ratio of 1 means the model is exact."
      flush
      tools={validation?.length ? <ExportButton columns={noiseColumns} rows={validation} filename="noise_validation.csv" /> : null}
    >
      {median !== null && (
        <div className="px-4 pb-3 sm:px-5">
          <Callout tone={Math.abs(median - 1) < 0.05 ? "good" : "warn"}>
            <Badge tone={Math.abs(median - 1) < 0.05 ? "good" : "warn"}>measured / predicted</Badge>
            <span className="num">
              {ratios.length} checks · median {fmt(median, 3)} · range {fmt(ratios[0], 3)}–{fmt(ratios[ratios.length - 1], 3)}
            </span>
          </Callout>
        </div>
      )}
      {validation?.length ? <DataTable columns={noiseColumns} rows={validation} rowKey={(_, i) => String(i)} maxHeight={420} dense /> : null}
      {photon?.length ? (
        <details className="group border-t border-line">
          <summary className="cursor-pointer px-4 py-3 text-[13px] font-medium text-fg-2 hover:text-fg sm:px-5">Photon transfer points ({photon.length})</summary>
          <DataTable
            rows={photon}
            rowKey={(_, i) => String(i)}
            maxHeight={360}
            dense
            columns={[
              { key: "source", header: "Source", align: "left", value: (r) => str(r, "source") },
              numCol("mean", "Mean", 4),
              { key: "variance", header: "Variance", value: (r) => num(r, "variance"), render: (r) => num(r, "variance")?.toExponential(3) ?? "—" },
            ]}
          />
        </details>
      ) : null}
    </Card>
  );
}

const noiseColumns: Column<Row>[] = [
  { key: "alpha", header: "Light", value: (r) => num(r, "alpha"), render: (r) => fmtPercent(num(r, "alpha")) },
  { key: "sigma_r_dn", header: "Read noise", value: (r) => num(r, "sigma_r_dn"), render: (r) => `${fmt(num(r, "sigma_r_dn"), 1)} DN` },
  { key: "binning", header: "Bin", value: (r) => num(r, "binning"), render: (r) => `${fmt(num(r, "binning"), 0)}×` },
  { key: "bit_depth", header: "ADC", value: (r) => num(r, "bit_depth"), render: (r) => `${fmt(num(r, "bit_depth"), 0)}-bit` },
  numCol("level", "Level", 2),
  { key: "channel", header: "Ch.", align: "left", value: (r) => str(r, "channel") },
  numCol("measured", "Measured σ", 5),
  numCol("predicted", "Predicted σ", 5),
  { key: "ratio", header: "Ratio", value: (r) => num(r, "ratio"), best: null, render: (r) => fmt(num(r, "ratio"), 3) },
];

/* ---------------------------------------------------------------- figures --- */

function FiguresSection({ figures }: { figures: { name: string; url: string }[] }) {
  const kinds = useMemo(() => {
    const kind = (name: string) => name.split("_")[0];
    const counts = new Map<string, number>();
    for (const f of figures) counts.set(kind(f.name), (counts.get(kind(f.name)) ?? 0) + 1);
    return { kind, list: [...counts].sort((a, b) => b[1] - a[1]) };
  }, [figures]);
  const [filter, setFilter] = useState("all");
  const [open, setOpen] = useState<number | null>(null);
  const shown = filter === "all" ? figures : figures.filter((f) => kinds.kind(f.name) === filter);
  const items = shown.map((f) => ({ src: f.url, caption: f.name.replaceAll("_", " ") }));

  return (
    <Card
      title={`Figures (${figures.length})`}
      sub="Rendered by the analysis stage for the paper. Click one to view it full size; arrow keys step through them."
      tools={
        kinds.list.length > 1 ? (
          <Segmented
            size="sm"
            label="Figure kind"
            value={filter}
            onChange={setFilter}
            options={[["all", `All ${figures.length}`], ...kinds.list.map(([k, n]) => [k, `${k} ${n}`] as [string, string])]}
          />
        ) : null
      }
    >
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-5">
        {shown.map((figure, i) => (
          <button
            key={figure.url}
            type="button"
            title={figure.name}
            onClick={() => setOpen(i)}
            className="group flex min-w-0 flex-col overflow-hidden rounded-lg border border-line bg-surface text-left transition-all hover:-translate-y-0.5 hover:border-line-strong hover:shadow-pop"
          >
            <span className="flex aspect-[4/3] items-center justify-center bg-white p-1.5">
              <img src={figure.url} alt={figure.name} loading="lazy" className="max-h-full max-w-full object-contain" />
            </span>
            <span className="truncate border-t border-line px-2.5 py-2 text-xs text-fg-2 group-hover:text-fg">{figure.name.replaceAll("_", " ")}</span>
          </button>
        ))}
      </div>
      <Lightbox items={items} index={open} onIndex={setOpen} onClose={() => setOpen(null)} />
    </Card>
  );
}
