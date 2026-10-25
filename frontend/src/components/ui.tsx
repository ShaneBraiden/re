/* Shared building blocks. Status always ships with an icon and a word, never
 * colour alone; untrusted text (file names, reasons) only ever renders as text. */

import clsx from "clsx";
import { AlertTriangle, Check, ChevronDown, Copy, Info, X } from "lucide-react";
import { useState, type ButtonHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from "react";
import type { OptionGroup } from "../lib/metrics";

/* ------------------------------------------------------------------ card --- */

export function Card({
  title,
  sub,
  tools,
  children,
  className,
  bodyClassName,
  id,
  flush = false,
}: {
  title?: ReactNode;
  sub?: ReactNode;
  tools?: ReactNode;
  children?: ReactNode;
  className?: string;
  bodyClassName?: string;
  id?: string;
  /** Body without padding, for tables and media that run edge to edge. */
  flush?: boolean;
}) {
  return (
    <section id={id} className={clsx("min-w-0 rounded-xl border border-line bg-surface shadow-card", className)}>
      {(title || tools) && (
        <header className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-4 pt-4 sm:px-5">
          <div className="min-w-0">
            {title && <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-fg">{title}</h2>}
            {sub && <p className="mt-0.5 max-w-[72ch] text-[13px] leading-relaxed text-fg-3">{sub}</p>}
          </div>
          {tools && <div className="flex flex-wrap items-center gap-2">{tools}</div>}
        </header>
      )}
      <div className={clsx(flush ? "pt-3" : "p-4 sm:p-5", (title || tools) && !flush && "pt-3 sm:pt-3", bodyClassName)}>{children}</div>
    </section>
  );
}

export function CardFoot({ children }: { children: ReactNode }) {
  return <p className="mt-3 text-xs leading-relaxed text-fg-3">{children}</p>;
}

/* ------------------------------------------------------------------ stats --- */

export function Stat({
  label,
  value,
  unit,
  note,
  tone,
  hero = false,
}: {
  label: ReactNode;
  value: ReactNode;
  unit?: ReactNode;
  note?: ReactNode;
  tone?: "warn" | "good" | "bad" | null;
  hero?: boolean;
}) {
  return (
    <div
      className={clsx(
        "flex min-w-0 flex-col gap-1 rounded-xl border p-4 shadow-card",
        hero ? "border-accent/30 bg-gradient-to-br from-accent-soft to-surface" : "border-line bg-surface",
      )}
    >
      <span className="text-xs font-medium text-fg-2">{label}</span>
      <span className={clsx("num flex items-baseline gap-1 font-semibold tracking-[-0.02em] text-fg", hero ? "text-[32px] leading-9" : "text-2xl")}>
        {value}
        {unit && <span className="text-sm font-medium tracking-normal text-fg-3">{unit}</span>}
      </span>
      {note && (
        <span
          className={clsx(
            "text-xs leading-relaxed",
            tone === "warn" ? "text-warn" : tone === "bad" ? "text-bad" : tone === "good" ? "text-good" : "text-fg-3",
          )}
        >
          {note}
        </span>
      )}
    </div>
  );
}

/* ----------------------------------------------------------------- badges --- */

export type Tone = "good" | "bad" | "warn" | "neutral" | "accent";

const TONE_CLASS: Record<Tone, string> = {
  good: "bg-good-soft text-good",
  bad: "bg-bad-soft text-bad",
  warn: "bg-warn-soft text-warn",
  neutral: "bg-surface-2 text-fg-2",
  accent: "bg-accent-soft text-accent",
};

const TONE_ICON: Record<Tone, typeof Check> = { good: Check, bad: X, warn: AlertTriangle, neutral: Info, accent: Info };

export function Badge({ tone, children, icon = true }: { tone: Tone; children: ReactNode; icon?: boolean }) {
  const Icon = TONE_ICON[tone];
  return (
    <span className={clsx("inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium", TONE_CLASS[tone])}>
      {icon && <Icon size={12} strokeWidth={2.5} aria-hidden />}
      {children}
    </span>
  );
}

export function Tag({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "accent" }) {
  return (
    <span
      className={clsx(
        "rounded px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide",
        tone === "accent" ? "bg-accent-soft text-accent" : "bg-surface-2 text-fg-3",
      )}
    >
      {children}
    </span>
  );
}

export function Delta({ value, tone, children }: { value?: number | null; tone: "good" | "bad" | null; children: ReactNode }) {
  return (
    <span className={clsx("num font-medium", tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : value === null ? "text-fg-3" : "text-fg-2")}>
      {children}
    </span>
  );
}

/* --------------------------------------------------------------- controls --- */

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
  size = "md",
}: {
  options: [T, ReactNode][];
  value: T;
  onChange: (value: T) => void;
  label?: string;
  size?: "sm" | "md";
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex max-w-full flex-wrap gap-0.5 rounded-lg bg-surface-2 p-0.5">
      {options.map(([key, text]) => (
        <button
          key={key}
          type="button"
          aria-pressed={key === value}
          onClick={() => onChange(key)}
          className={clsx(
            "inline-flex items-center gap-1.5 rounded-md font-medium transition-colors",
            size === "sm" ? "px-2 py-1 text-xs" : "px-3 py-1.5 text-[13px]",
            key === value ? "bg-surface text-fg shadow-card" : "text-fg-2 hover:text-fg",
          )}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

type SelectOption = [string, string];

export function Select({
  options,
  value,
  onChange,
  label,
  className,
  ...rest
}: {
  options: (SelectOption | OptionGroup)[];
  value: string;
  onChange: (value: string) => void;
  label: string;
  className?: string;
} & Omit<SelectHTMLAttributes<HTMLSelectElement>, "onChange" | "value">) {
  return (
    <div className={clsx("relative inline-flex min-w-0", className)}>
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-9 w-full min-w-0 cursor-pointer appearance-none truncate rounded-lg border border-line bg-surface py-0 pl-3 pr-8 text-[13px] font-medium text-fg shadow-card transition-colors hover:border-line-strong"
        {...rest}
      >
        {options.map((option) =>
          Array.isArray(option) ? (
            <option key={option[0]} value={option[0]}>
              {option[1]}
            </option>
          ) : (
            <optgroup key={option.group} label={option.group}>
              {option.items.map(([key, text]) => (
                <option key={key} value={key}>
                  {text}
                </option>
              ))}
            </optgroup>
          ),
        )}
      </select>
      <ChevronDown size={15} className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-fg-3" aria-hidden />
    </div>
  );
}

/** A div, not a <label>: a label wrapping a group of buttons would press the
 * first one whenever its caption is clicked. Controls carry aria-label. */
export function Field({ label, children, className, hint }: { label: ReactNode; children: ReactNode; className?: string; hint?: ReactNode }) {
  return (
    <div className={clsx("flex min-w-0 flex-col gap-1.5", className)}>
      <span className="text-[11px] font-semibold uppercase tracking-[0.06em] text-fg-3">{label}</span>
      {children}
      {hint && <span className="text-xs text-fg-3">{hint}</span>}
    </div>
  );
}

export function Button({
  children,
  variant = "ghost",
  className,
  ...rest
}: { variant?: "primary" | "ghost" | "outline" } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={clsx(
        "inline-flex h-9 items-center justify-center gap-2 rounded-lg px-3.5 text-[13px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "bg-accent text-accent-fg shadow-card hover:brightness-110",
        variant === "ghost" && "text-fg-2 hover:bg-surface-2 hover:text-fg",
        variant === "outline" && "border border-line bg-surface text-fg shadow-card hover:border-line-strong",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

export function IconButton({ label, children, className, ...rest }: { label: string } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={clsx("inline-flex size-8 items-center justify-center rounded-lg text-fg-2 transition-colors hover:bg-surface-2 hover:text-fg", className)}
      {...rest}
    >
      {children}
    </button>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-line bg-surface-2 px-1.5 py-px text-[11px] text-fg-2">{children}</kbd>;
}

/* ------------------------------------------------------------ feedback --- */

export function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard blocked: the text is selectable */
    }
  };
  return (
    <div className="group flex max-w-full items-center gap-2 rounded-lg border border-line bg-surface-2 py-1.5 pl-3 pr-1.5 text-left">
      <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-[12.5px] text-fg scroll-thin">
        <span className="select-none text-fg-3">$ </span>
        {command}
      </code>
      <IconButton label={copied ? "Copied" : "Copy command"} onClick={copy} className="size-7 shrink-0">
        {copied ? <Check size={14} className="text-good" /> : <Copy size={14} />}
      </IconButton>
    </div>
  );
}

export function EmptyState({ title, detail, command, icon }: { title: ReactNode; detail?: ReactNode; command?: string; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 px-4 py-10 text-center">
      <div className="mb-1 flex size-10 items-center justify-center rounded-full bg-surface-2 text-fg-3">{icon ?? <Info size={18} />}</div>
      <p className="text-sm font-semibold text-fg">{title}</p>
      {detail && <p className="max-w-[56ch] text-[13px] leading-relaxed text-fg-3">{detail}</p>}
      {command && (
        <div className="mt-2 w-full max-w-xl">
          <CopyCommand command={command} />
        </div>
      )}
    </div>
  );
}

export function ErrorBanner({ children }: { children: ReactNode }) {
  return (
    <div role="alert" className="flex items-start gap-2 rounded-lg border border-bad/30 bg-bad-soft px-3 py-2.5 text-[13px] text-bad">
      <AlertTriangle size={16} className="mt-px shrink-0" aria-hidden />
      <span className="min-w-0 break-words">{children}</span>
    </div>
  );
}

export function Callout({ children, tone = "neutral" }: { children: ReactNode; tone?: Tone }) {
  return (
    <div
      className={clsx(
        "flex flex-wrap items-center gap-2 rounded-lg border px-3 py-2.5 text-[13px] leading-relaxed",
        tone === "warn" ? "border-warn/25 bg-warn-soft" : tone === "good" ? "border-good/25 bg-good-soft" : "border-line bg-surface-2",
        "text-fg-2",
      )}
    >
      {children}
    </div>
  );
}

export function Spinner({ size = 16 }: { size?: number }) {
  return (
    <span
      aria-hidden
      style={{ width: size, height: size }}
      className="inline-block shrink-0 animate-spin rounded-full border-2 border-line-strong border-t-accent"
    />
  );
}

export function Loading({ text = "Loading…" }: { text?: string }) {
  return (
    <div role="status" className="flex items-center justify-center gap-2.5 py-12 text-[13px] text-fg-3">
      <Spinner />
      {text}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={clsx("rounded-lg bg-surface-2", className)}
      style={{
        backgroundImage: "linear-gradient(90deg, transparent 0%, var(--surface-3) 50%, transparent 100%)",
        backgroundSize: "200% 100%",
        animation: "shimmer 1.6s ease-in-out infinite",
      }}
    />
  );
}

export function KeyValues({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(110px,max-content)_1fr] gap-x-5 gap-y-2.5 text-[13px]">
      {rows.map(([key, value], i) => (
        <div key={i} className="contents">
          <dt className="text-fg-3">{key}</dt>
          <dd className="min-w-0 break-words text-fg">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
