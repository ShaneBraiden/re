/* One enhancement run: before/after comparison, what was done, the
 * no-reference metrics, the estimated noise map and every file it wrote. */

import { ArrowLeft } from "lucide-react";
import { useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router";
import { CompareViewer } from "../components/CompareViewer";
import { DataTable, ExportButton, type Column } from "../components/DataTable";
import { PageHeader } from "../components/Layout";
import { Lightbox } from "../components/Lightbox";
import { ChartCard } from "../components/LineChart";
import { Card, CardFoot, EmptyState, ErrorBanner, KeyValues, Select, Skeleton, Tag } from "../components/ui";
import { fmt, fmtDate, fmtDuration } from "../lib/format";
import { useEnhancements } from "../lib/queries";
import { useRun } from "../lib/run";
import type { Direction, Enhancement } from "../lib/types";

const OUTPUTS: Record<string, string> = {
  original: "Original upload",
  input_bicubic: "Bicubic upscale",
  classical: "Classical pipeline",
  nag: "NAG (ours)",
  gate: "Wiener gate map",
  // Runs made before enhancing stopped writing paper figures still have one.
  comparison: "Comparison figure",
};

const METRIC_COLUMNS: [string, string, Direction | undefined, number][] = [
  ["sharpness", "Sharpness", true, 2],
  ["noise_sigma", "Noise σ̂", false, 3],
  ["nr_quality", "NR quality", true, 2],
  ["colorfulness", "Colorfulness", undefined, 1],
  ["consistency_psnr", "Consistency PSNR", true, 2],
  ["consistency_ssim", "Consistency SSIM", true, 4],
];
const METRIC_ROWS: [string, string][] = [
  ["input_bicubic", "Bicubic upscale"],
  ["classical", "Classical pipeline"],
  ["nag", "NAG (ours)"],
];

export default function EnhanceResult() {
  const { id = "" } = useParams();
  const { link } = useRun();
  const history = useEnhancements();
  const back = (
    <Link
      to={link("/enhance")}
      className="inline-flex h-9 items-center gap-2 rounded-lg border border-line bg-surface px-3.5 text-[13px] font-medium text-fg shadow-card hover:border-line-strong"
    >
      <ArrowLeft size={15} /> All runs
    </Link>
  );

  if (history.isLoading) {
    return (
      <>
        <PageHeader title="Enhancement" actions={back} />
        <div className="grid gap-5 lg:grid-cols-12">
          <Skeleton className="h-[520px] lg:col-span-8" />
          <Skeleton className="h-[520px] lg:col-span-4" />
        </div>
      </>
    );
  }
  if (history.error) {
    return (
      <>
        <PageHeader title="Enhancement" actions={back} />
        <ErrorBanner>{String((history.error as Error).message)}</ErrorBanner>
      </>
    );
  }
  const run = history.data!.runs.find((r) => r.id === id);
  if (!run) {
    return (
      <>
        <PageHeader title="Enhancement" actions={back} />
        <Card>
          <EmptyState title="No enhancement run with that id" detail={`Nothing in experiments/enhance is called “${id}”.`} />
        </Card>
      </>
    );
  }
  return <Result run={run} back={back} />;
}

function Result({ run, back }: { run: Enhancement; back: ReactNode }) {
  const [ih, iw] = run.input_shape ?? [];
  const [oh, ow] = run.output_shape ?? [];

  const leftChoices = ["original", "input_bicubic", "classical", "gate"].filter((k) => run.urls[k]);
  const rightChoices = ["nag", "classical", "input_bicubic"].filter((k) => run.urls[k]);
  const [left, setLeft] = useState(run.urls.input_bicubic ? "input_bicubic" : leftChoices[0]);
  const [right, setRight] = useState(rightChoices[0]);
  const [lightbox, setLightbox] = useState<number | null>(null);
  const wrap = useRef<HTMLDivElement>(null);

  const files = Object.entries(run.urls).map(([key, url]) => ({ key, url, name: url.split("/").pop() ?? key }));
  const items = files.map((f) => ({ src: f.url, caption: `${OUTPUTS[f.key] ?? f.key} · ${f.name}` }));

  return (
    <>
      <PageHeader
        eyebrow="Enhancement run"
        title={run.name ?? run.id}
        sub={
          <span className="num">
            {run.kind ?? "image"} · {iw}×{ih} → {ow}×{oh} px (×{fmt(run.scale, 1)}) · {fmtDuration(run.seconds)} · {fmtDate(run.modified)}
          </span>
        }
        actions={back}
      />
      <div className="flex flex-col gap-5">
        <div className="grid gap-5 lg:grid-cols-12">
          <Card
            className="lg:col-span-8"
            title="Before and after"
            sub="Drag the divider to wipe · scroll or double-click to zoom into detail"
            tools={
              <div className="flex w-full items-center gap-1.5 sm:w-auto">
                <Select label="Left image" value={left} onChange={setLeft} options={leftChoices.map((k) => [k, OUTPUTS[k]])} className="min-w-0 flex-1 sm:w-44 sm:flex-none" />
                <span className="text-xs text-fg-3">vs</span>
                <Select label="Right image" value={right} onChange={setRight} options={rightChoices.map((k) => [k, OUTPUTS[k]])} className="min-w-0 flex-1 sm:w-44 sm:flex-none" />
              </div>
            }
          >
            <div ref={wrap} className="flex items-center justify-center [&:fullscreen]:bg-black">
              <CompareViewer
                left={{ src: run.urls[left], label: OUTPUTS[left] }}
                right={{ src: run.urls[right], label: OUTPUTS[right] }}
                aspect={ow && oh ? `${ow} / ${oh}` : "4 / 3"}
                pixelated="zoom"
                maxZoom={24}
                className="[:fullscreen_&]:h-full [:fullscreen_&]:w-auto [:fullscreen_&]:max-w-full [:fullscreen_&]:rounded-none"
                onFullscreen={() => (document.fullscreenElement ? document.exitFullscreen() : wrap.current?.requestFullscreen())}
              />
            </div>
          </Card>
          <div className="flex flex-col gap-5 lg:col-span-4">
            <Steps run={run} />
            <Classifier run={run} />
          </div>
        </div>

        <Metrics run={run} />

        <div className="grid gap-5 xl:grid-cols-2">
          <NoiseMap run={run} />
          <Card title="Files" sub={<code>experiments/enhance/{run.id}/</code>} className={run.noise_level_function?.centres?.length ? "" : "xl:col-span-2"}>
            <ul className="grid gap-2 sm:grid-cols-2">
              {files.map((file, i) => (
                <li key={file.key}>
                  <button
                    type="button"
                    onClick={() => setLightbox(i)}
                    className="flex w-full items-center gap-3 rounded-lg border border-line p-2 text-left transition-colors hover:border-line-strong hover:bg-surface-2"
                  >
                    <img src={file.url} alt="" loading="lazy" className="size-11 shrink-0 rounded-md bg-surface-2 object-cover" />
                    <span className="flex min-w-0 flex-col">
                      <span className="truncate text-[13px] font-medium text-fg">{OUTPUTS[file.key] ?? file.key}</span>
                      <span className="truncate text-xs text-fg-3">{file.name}</span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      </div>
      <Lightbox items={items} index={lightbox} onIndex={setLightbox} onClose={() => setLightbox(null)} />
    </>
  );
}

function Steps({ run }: { run: Enhancement }) {
  const steps = run.steps ?? [];
  return (
    <Card title="What was done" sub={`${steps.length} steps, in order`}>
      <ol className="flex flex-col">
        {steps.map((step, i) => (
          <li key={i} className="relative flex gap-3 pb-4 last:pb-0">
            {i < steps.length - 1 && <span className="absolute left-[5px] top-4 h-[calc(100%-10px)] w-px bg-line" aria-hidden />}
            <span className="mt-1.5 size-[11px] shrink-0 rounded-full border-2 border-accent bg-surface" aria-hidden />
            <div className="flex min-w-0 flex-col">
              <b className="text-[13px] font-semibold text-fg">{step.step}</b>
              <span className="break-words text-xs leading-relaxed text-fg-2">{step.detail}</span>
            </div>
          </li>
        ))}
      </ol>
    </Card>
  );
}

function Classifier({ run }: { run: Enhancement }) {
  const features = run.features ?? {};
  const rows: [string, ReactNode][] = [];
  if ("palette_top8" in features) rows.push(["Top-8 colour share", fmt(features.palette_top8, 2)]);
  if ("flat_share" in features) rows.push(["Flat area share", fmt(features.flat_share, 2)]);
  if ("edge_density" in features) rows.push(["Edge density", fmt(features.edge_density, 3)]);
  for (const [name, sha] of Object.entries(run.checkpoints ?? {})) {
    rows.push([
      name,
      <code key={name} title={sha} className="text-[12px]">
        {sha.slice(0, 12)}…
      </code>,
    ]);
  }
  return (
    <Card title="Classifier" sub={`Treated as a ${run.kind ?? "image"}`}>
      {rows.length ? <KeyValues rows={rows} /> : <p className="text-[13px] text-fg-3">No classifier features recorded.</p>}
    </Card>
  );
}

function Metrics({ run }: { run: Enhancement }) {
  type R = { key: string; label: string } & Record<string, number | null | string>;
  const rows: R[] = METRIC_ROWS.filter(([key]) => run.metrics?.[key]).map(([key, label]) => ({ key, label, ...run.metrics![key] }));
  const columns: Column<R>[] = useMemo(
    () => [
      {
        key: "label",
        header: "Output",
        align: "left",
        value: (r) => r.label,
        render: (r) => (
          <span className="inline-flex items-center gap-2">
            {r.label}
            {r.key === "nag" && <Tag tone="accent">ours</Tag>}
          </span>
        ),
      },
      ...METRIC_COLUMNS.filter(([key]) => rows.some((r) => key in r)).map(([key, label, direction, digits]) => ({
        key,
        header: label,
        best: direction,
        title: direction === undefined ? "no better direction" : direction ? "higher is better" : "lower is better",
        value: (r: R) => (typeof r[key] === "number" ? (r[key] as number) : null),
        render: (r: R) => fmt(typeof r[key] === "number" ? (r[key] as number) : null, digits),
      })),
    ],
    [run],
  );
  if (!rows.length) return null;
  return (
    <Card
      title="Metrics"
      sub="No ground truth exists for an upload, so these are no-reference proxies plus a consistency check."
      tools={<ExportButton columns={columns} rows={rows} filename={`${run.id}_metrics.csv`} />}
      flush
    >
      <DataTable columns={columns} rows={rows} rowKey={(r) => r.key} />
      <div className="px-4 pb-4 sm:px-5">
        <CardFoot>
          Consistency: the output is downscaled to the input size, tone-matched per channel and compared with the upload, so invented detail lowers it. Bicubic
          scores highest by construction.
        </CardFoot>
      </div>
    </Card>
  );
}

function NoiseMap({ run }: { run: Enhancement }) {
  const nlf = run.noise_level_function;
  if (!nlf?.centres?.length || !nlf.sigma?.length) return null;
  const channels: [string, string][] =
    nlf.sigma.length === 3
      ? [
          ["R", "--series-5"],
          ["G", "--series-6"],
          ["B", "--series-1"],
        ]
      : nlf.sigma.map((_, i) => [`ch ${i}`, `--series-${i + 1}`]);
  return (
    <ChartCard
      title="Noise map"
      sub={`${nlf.estimated ? "Estimated blindly from the upload" : "From the capture record"}: the per-pixel σ every NAG stage is steered by`}
      filename={`${run.id}_noise_level_function.csv`}
      chart={{
        series: channels.map(([label, color], i) => ({
          name: label,
          label,
          color,
          points: nlf.centres.map((x, j) => ({ x, y: nlf.sigma[i][j] === null ? null : nlf.sigma[i][j]! * 255 })),
        })),
        height: 260,
        xTitle: "Intensity",
        yTitle: "Noise σ (8-bit levels)",
        xFormat: (x) => fmt(x, 2),
        yFormat: (y) => fmt(y, 2),
        direction: false,
      }}
    />
  );
}
