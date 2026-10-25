/* Enhance: run NAG on an uploaded image (when the server was started with
 * --upload), and browse every enhancement run already on disk. The history is
 * read-only and shows without the flag. */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { CheckCircle2, ImageIcon, ImageUp, Lock, Sparkles } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate, useSearchParams } from "react-router";
import { PageHeader } from "../components/Layout";
import { Badge, Button, Card, CopyCommand, EmptyState, ErrorBanner, Field, Segmented, Skeleton, Spinner } from "../components/ui";
import { api } from "../lib/api";
import { fmtBytes, fmtDate, fmtDuration } from "../lib/format";
import { useEnhancements } from "../lib/queries";
import { useRun } from "../lib/run";
import type { Enhancement } from "../lib/types";

const MAX_BYTES = 20 * 1024 * 1024;
const ACCEPT = ".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff";

export default function Enhance() {
  const [params] = useSearchParams();
  const { link, upload } = useRun();
  const history = useEnhancements();

  // Links from the classic dashboard used ?id=<run>.
  const legacyId = params.get("id");
  if (legacyId) return <Navigate to={link(`/enhance/${encodeURIComponent(legacyId)}`)} replace />;

  return (
    <>
      <PageHeader
        title="Enhance"
        sub="Run NAG on your own image: noise-aware curves, VP-denoise and guarded Real-ESRGAN, on this machine."
      />
      <div className="flex flex-col gap-5">
        <div className="grid gap-5 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          {upload ? <UploadCard /> : <UploadOff />}
          <MethodCard />
        </div>
        <Card
          title={history.data ? `Past runs (${history.data.runs.length})` : "Past runs"}
          sub="Every enhancement on disk, newest first. Open one to compare it with the baselines."
        >
          {history.isLoading ? (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {Array.from({ length: 3 }, (_, i) => (
                <Skeleton key={i} className="h-64" />
              ))}
            </div>
          ) : history.error ? (
            <ErrorBanner>{String((history.error as Error).message)}</ErrorBanner>
          ) : history.data!.runs.length === 0 ? (
            <EmptyState
              icon={<ImageIcon size={18} />}
              title="No enhancement runs yet"
              detail="Upload an image above, or run one from the command line."
              command="python -m sensorsim.cli enhance image.png"
            />
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {history.data!.runs.map((run) => (
                <RunTile key={run.id} run={run} />
              ))}
            </div>
          )}
        </Card>
      </div>
    </>
  );
}

function RunTile({ run }: { run: Enhancement }) {
  const { link } = useRun();
  const [ih, iw] = run.input_shape ?? [];
  const [oh, ow] = run.output_shape ?? [];
  const thumb = run.urls.nag ?? run.urls.original;
  return (
    <Link
      to={link(`/enhance/${encodeURIComponent(run.id)}`)}
      className="group flex min-w-0 flex-col overflow-hidden rounded-xl border border-line bg-surface transition-all hover:-translate-y-0.5 hover:border-line-strong hover:shadow-pop"
    >
      <div className="relative aspect-[16/10] overflow-hidden bg-surface-2">
        {thumb ? (
          <img src={thumb} alt="" loading="lazy" className="size-full object-cover transition-transform duration-500 group-hover:scale-[1.03]" />
        ) : (
          <div className="flex size-full items-center justify-center text-fg-3">
            <ImageIcon size={28} />
          </div>
        )}
        {run.scale && (
          <span className="absolute right-2 top-2 rounded-md bg-black/60 px-1.5 py-0.5 text-[11px] font-semibold text-white backdrop-blur">×{run.scale}</span>
        )}
      </div>
      <div className="flex flex-col gap-1.5 p-3">
        <span className="truncate text-[13.5px] font-semibold text-fg" title={run.name}>
          {run.name ?? run.id}
        </span>
        <span className="flex flex-wrap items-center gap-2 text-xs text-fg-3">
          <Badge tone="neutral" icon={false}>
            {run.kind ?? "image"}
          </Badge>
          {iw && (
            <span className="num">
              {iw}×{ih} → {ow}×{oh}
            </span>
          )}
        </span>
        <span className="text-xs text-fg-3">
          {fmtDate(run.modified)} · {fmtDuration(run.seconds)}
        </span>
      </div>
    </Link>
  );
}

/* ------------------------------------------------------------ upload --- */

type Kind = "auto" | "photo" | "diagram";

function UploadCard() {
  const navigate = useNavigate();
  const { link } = useRun();
  const queryClient = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [size, setSize] = useState<[number, number] | null>(null);
  const [over, setOver] = useState(false);
  const [kind, setKind] = useState<Kind>("auto");
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => () => void (preview && URL.revokeObjectURL(preview)), [preview]);

  const mutation = useMutation({
    mutationFn: (form: FormData) => api.enhance(form),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: ["enhancements"] });
      navigate(link(`/enhance/${encodeURIComponent(result.id)}`));
    },
  });

  useEffect(() => {
    if (!mutation.isPending) return;
    const started = performance.now();
    setElapsed(0);
    const timer = setInterval(() => setElapsed((performance.now() - started) / 1000), 500);
    return () => clearInterval(timer);
  }, [mutation.isPending]);

  const pick = (chosen: File | undefined) => {
    if (!chosen) return;
    setFile(chosen);
    setSize(null);
    setPreview(URL.createObjectURL(chosen));
    mutation.reset();
  };

  const tooBig = Boolean(file && file.size > MAX_BYTES);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!file || tooBig) return;
    const form = new FormData();
    form.append("file", file);
    form.append("kind", kind);
    mutation.mutate(form);
  };

  return (
    <Card
      title="Upload an image"
      sub="Enhanced on this machine, one image at a time. The result is saved under experiments/enhance/ and nothing else is written."
    >
      <form onSubmit={submit} className="flex flex-col gap-4">
        <label
          onDragEnter={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            pick(e.dataTransfer.files[0]);
          }}
          className={clsx(
            "relative flex min-h-52 cursor-pointer flex-col items-center justify-center gap-2 overflow-hidden rounded-xl border-2 border-dashed p-6 text-center transition-colors",
            over ? "border-accent bg-accent-soft" : "border-line-strong bg-surface-2/50 hover:border-accent/60 hover:bg-surface-2",
            mutation.isPending && "pointer-events-none opacity-60",
          )}
        >
          <input
            ref={input}
            type="file"
            accept={ACCEPT}
            className="sr-only"
            onChange={(event) => pick(event.target.files?.[0])}
            disabled={mutation.isPending}
          />
          {preview ? (
            <>
              <img
                src={preview}
                alt=""
                onLoad={(e) => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
                className="max-h-56 max-w-full rounded-lg object-contain shadow-card"
              />
              <span className="mt-1 max-w-full truncate text-[13px] font-semibold text-fg">{file!.name}</span>
              <span className="num text-xs text-fg-3">
                {size ? `${size[0]}×${size[1]} px · ` : ""}
                {fmtBytes(file!.size)} · click or drop to replace
              </span>
            </>
          ) : (
            <>
              <span className="flex size-12 items-center justify-center rounded-full bg-accent-soft text-accent">
                <ImageUp size={22} />
              </span>
              <span className="text-sm font-semibold text-fg">Drop an image here, or click to choose</span>
              <span className="text-xs text-fg-3">PNG, JPEG, WebP, BMP or TIFF · up to 20 MB</span>
            </>
          )}
        </label>

        <Field label="Image type">
          <Segmented
            label="Image type"
            value={kind}
            onChange={setKind}
            options={[
              ["auto", "Detect"],
              ["photo", "Photo"],
              ["diagram", "Diagram / screenshot"],
            ]}
          />
        </Field>

        <div className="flex flex-wrap items-center gap-3 border-t border-line pt-4">
          <Button variant="primary" type="submit" disabled={!file || tooBig || mutation.isPending}>
            {mutation.isPending ? <Spinner size={14} /> : <Sparkles size={15} />}
            {mutation.isPending ? "Enhancing…" : "Enhance image"}
          </Button>
          <div role="status" aria-live="polite" className="text-[13px] text-fg-2">
            {mutation.isPending && (
              <span>
                <b className="num font-semibold text-fg">{Math.round(elapsed)} s</b> · a large photo takes a minute or two on a laptop GPU. Keep this tab open.
              </span>
            )}
            {mutation.isSuccess && (
              <span className="inline-flex items-center gap-1.5 text-good">
                <CheckCircle2 size={15} /> Done in {fmtDuration(mutation.data.seconds)}.
              </span>
            )}
          </div>
        </div>
        {tooBig && <ErrorBanner>That file is over the 20 MB limit.</ErrorBanner>}
        {mutation.error && <ErrorBanner>{String((mutation.error as Error).message)}</ErrorBanner>}
      </form>
    </Card>
  );
}

function UploadOff() {
  const { defaultExp } = useRun();
  return (
    <Card title="Upload is off" sub="The dashboard is read-only unless you ask for its one write route." tools={<Badge tone="neutral">read-only</Badge>}>
      <div className="flex flex-col gap-3 text-[13px] text-fg-2">
        <div className="flex items-start gap-3 rounded-lg bg-surface-2 p-3">
          <Lock size={16} className="mt-0.5 shrink-0 text-fg-3" />
          <p>Restart the dashboard with <code>--upload</code> to enhance images from the browser:</p>
        </div>
        <CopyCommand command={`python -m sensorsim.cli dashboard experiments/${defaultExp ?? "pilot"} --upload`} />
        <p>Or run one image from the command line:</p>
        <CopyCommand command="python -m sensorsim.cli enhance image.png" />
        <p className="text-xs text-fg-3">Past runs below are already on disk and open either way.</p>
      </div>
    </Card>
  );
}

function MethodCard() {
  const steps = [
    ["NC-DCE", "Zero-DCE's curve network with a per-pixel noise map as a fourth input, trained so its curves do not amplify noise in flat regions."],
    ["VP-denoise", "A guided filter whose strength at each pixel is the noise the curves left there."],
    ["G-ESRGAN", "Real-ESRGAN's added detail scaled by a Wiener gate, kept only where the measurement holds more than noise, then back-projected so the output stays consistent with the input."],
  ];
  return (
    <Card title="What NAG does" sub="Noise-aware guarded enhancement. On uploads the noise map is estimated blindly.">
      <ol className="relative flex flex-col gap-4">
        {steps.map(([name, text], i) => (
          <li key={name} className="relative flex gap-3">
            {i < steps.length - 1 && <span className="absolute left-[13px] top-8 h-[calc(100%-16px)] w-px bg-line" aria-hidden />}
            <span className="num flex size-7 shrink-0 items-center justify-center rounded-full bg-accent-soft text-xs font-semibold text-accent">{i + 1}</span>
            <div className="flex flex-col gap-0.5 pt-0.5">
              <b className="text-[13.5px] font-semibold text-fg">{name}</b>
              <p className="text-[13px] leading-relaxed text-fg-2">{text}</p>
            </div>
          </li>
        ))}
      </ol>
    </Card>
  );
}
