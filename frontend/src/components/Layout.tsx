/* The shell: navigation, run switcher and theme in a sidebar on wide screens,
 * a top bar and drawer on narrow ones. */

import clsx from "clsx";
import { BarChart3, ExternalLink, LayoutDashboard, Menu, Monitor, Moon, ScanSearch, Sparkles, Sun, X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router";
import { useRun } from "../lib/run";
import { ErrorBoundary } from "./ErrorBoundary";
import { useTheme, type Theme } from "../lib/theme";
import { Badge, IconButton, Segmented, Select } from "./ui";

const NAV = [
  { path: "/overview", label: "Overview", icon: LayoutDashboard, hint: "The run at a glance" },
  { path: "/explore", label: "Explorer", icon: ScanSearch, hint: "One scene, one condition" },
  { path: "/analysis", label: "Analysis", icon: BarChart3, hint: "Recovery, knees, significance" },
  { path: "/enhance", label: "Enhance", icon: Sparkles, hint: "Run NAG on an image" },
];

export function Layout() {
  const [drawer, setDrawer] = useState(false);
  const location = useLocation();
  const { index } = useRun();

  useEffect(() => setDrawer(false), [location.pathname]);
  useEffect(() => {
    document.documentElement.style.overflow = drawer ? "hidden" : "";
  }, [drawer]);

  return (
    <div className="min-h-dvh lg:pl-64">
      <a
        href="#main"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main")?.focus();
        }}
        className="sr-only z-50 rounded-lg bg-accent px-3 py-2 text-accent-fg focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to content
      </a>

      {/* Wide screens: a fixed sidebar. */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 border-r border-line bg-surface lg:block" aria-label="Navigation">
        <SidebarContent />
      </aside>

      {/* Narrow screens: a top bar and a drawer. */}
      <header className="sticky top-0 z-30 flex h-14 items-center gap-2 border-b border-line bg-surface/85 px-3 backdrop-blur lg:hidden">
        <IconButton label="Open navigation" onClick={() => setDrawer(true)}>
          <Menu size={18} />
        </IconButton>
        <Brand compact />
      </header>
      <div className={clsx("fixed inset-0 z-40 lg:hidden", drawer ? "pointer-events-auto" : "pointer-events-none")} aria-hidden={!drawer}>
        <div
          className={clsx("absolute inset-0 bg-black/40 transition-opacity", drawer ? "opacity-100" : "opacity-0")}
          onClick={() => setDrawer(false)}
        />
        <aside
          className={clsx(
            "absolute inset-y-0 left-0 w-72 max-w-[85vw] border-r border-line bg-surface shadow-pop transition-transform",
            drawer ? "translate-x-0" : "-translate-x-full",
          )}
          aria-label="Navigation"
        >
          <IconButton label="Close navigation" onClick={() => setDrawer(false)} className="absolute right-2 top-3">
            <X size={18} />
          </IconButton>
          {drawer && <SidebarContent />}
        </aside>
      </div>

      <main id="main" tabIndex={-1} className="mx-auto w-full max-w-[1440px] px-4 pb-10 pt-6 outline-none sm:px-6 lg:px-8 lg:pt-8">
        <ErrorBoundary resetKey={location.pathname}>
          <Outlet />
        </ErrorBoundary>
      </main>

      <footer className="mx-auto flex max-w-[1440px] flex-wrap justify-between gap-x-6 gap-y-1 border-t border-line px-4 py-5 text-xs text-fg-3 sm:px-6 lg:px-8">
        <span className="min-w-0 break-all">{index ? `${index.experiment_id} · from ${index.generated_from}` : "SensorSim"}</span>
        <span>Read-only view over cached results: every number here was computed by an earlier run.</span>
      </footer>
    </div>
  );
}

function Brand({ compact = false }: { compact?: boolean }) {
  const { link } = useRun();
  return (
    <NavLink to={link("/overview")} className="flex min-w-0 items-center gap-2.5 rounded-lg">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-fg shadow-card">
        <svg viewBox="0 0 32 32" width="20" height="20" aria-hidden>
          <circle cx="16" cy="16" r="9" fill="none" stroke="currentColor" strokeWidth="2.6" />
          <circle cx="16" cy="16" r="3.4" fill="currentColor" />
        </svg>
      </span>
      <span className="flex min-w-0 flex-col leading-tight">
        <span className="text-[15px] font-semibold tracking-[-0.01em] text-fg">SensorSim</span>
        {!compact && <span className="truncate text-[11.5px] text-fg-3">sensor–algorithm testbench</span>}
      </span>
    </NavLink>
  );
}

function SidebarContent() {
  const { link, upload } = useRun();
  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto px-3 py-4 scroll-thin">
      <div className="px-2">
        <Brand />
      </div>
      <nav className="flex flex-col gap-0.5" aria-label="Pages">
        {NAV.map(({ path, label, icon: Icon, hint }) => (
          <NavLink
            key={path}
            to={link(path)}
            className={({ isActive }) =>
              clsx(
                "group flex items-center gap-3 rounded-lg px-2.5 py-2 text-[13.5px] font-medium transition-colors",
                isActive ? "bg-accent-soft text-accent" : "text-fg-2 hover:bg-surface-2 hover:text-fg",
              )
            }
          >
            <Icon size={17} strokeWidth={2} aria-hidden />
            <span className="flex min-w-0 flex-1 flex-col leading-tight">
              <span>{label}</span>
              <span className="truncate text-[11px] font-normal text-fg-3">{hint}</span>
            </span>
            {path === "/enhance" && upload && (
              <span className="rounded-full bg-good-soft px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-good">on</span>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto flex flex-col gap-4">
        <RunSwitcher />
        <ThemeSwitch />
        <a
          href="/classic"
          className="flex items-center gap-1.5 px-2 text-xs text-fg-3 transition-colors hover:text-fg"
          title="The original no-build dashboard, kept as a fallback"
        >
          Classic dashboard <ExternalLink size={12} />
        </a>
      </div>
    </div>
  );
}

function RunSwitcher() {
  const { experiments, exp, setExp } = useRun();
  const current = experiments.find((run) => run.id === exp);
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-line bg-surface-2/60 p-3">
      <span className="text-[11px] font-semibold uppercase tracking-[0.06em] text-fg-3">Run</span>
      {experiments.length ? (
        <Select
          label="Experiment run"
          value={exp ?? ""}
          onChange={setExp}
          options={experiments.map((run) => [run.id, run.error ? `${run.id} (no index)` : run.id])}
        />
      ) : (
        <span className="text-[13px] text-fg-3">No runs found</span>
      )}
      {current && !current.error && (
        <div className="flex flex-col gap-1.5 text-xs text-fg-3">
          <span className="num">
            {current.available} of {current.pipelines} arms · {current.images} scenes · {current.conditions} conditions
          </span>
          <span>
            {current.validation === true ? (
              <Badge tone="good">calibration passed</Badge>
            ) : current.validation === false ? (
              <Badge tone="bad">calibration failed</Badge>
            ) : (
              <Badge tone="neutral">no calibration record</Badge>
            )}
          </span>
        </div>
      )}
    </div>
  );
}

function ThemeSwitch() {
  const [theme, setTheme] = useTheme();
  const option = (key: Theme, icon: ReactNode, text: string): [Theme, ReactNode] => [
    key,
    <span className="inline-flex items-center gap-1.5" title={text}>
      {icon}
      <span className="text-xs">{text}</span>
    </span>,
  ];
  return (
    <div className="flex flex-col gap-2 px-1">
      <span className="text-[11px] font-semibold uppercase tracking-[0.06em] text-fg-3">Theme</span>
      <Segmented
        label="Theme"
        size="sm"
        value={theme}
        onChange={setTheme}
        options={[option("auto", <Monitor size={13} />, "Auto"), option("light", <Sun size={13} />, "Light"), option("dark", <Moon size={13} />, "Dark")]}
      />
    </div>
  );
}

/* ------------------------------------------------------------ header --- */

export function PageHeader({ title, sub, actions, eyebrow }: { title: ReactNode; sub?: ReactNode; actions?: ReactNode; eyebrow?: ReactNode }) {
  useEffect(() => {
    if (typeof title === "string") document.title = `${title} · SensorSim`;
  }, [title]);
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && <div className="mb-1 text-xs font-medium text-fg-3">{eyebrow}</div>}
        <h1 className="break-words text-2xl font-semibold tracking-[-0.02em] text-fg sm:text-[28px]">{title}</h1>
        {sub && <p className="mt-1.5 max-w-[90ch] text-[13.5px] leading-relaxed text-fg-2">{sub}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
