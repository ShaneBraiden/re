/* A data table: sortable columns, the best value per column marked, optional
 * row selection, and a raw-value CSV export the card can offer. */

import clsx from "clsx";
import { ArrowDown, ArrowUp, Download } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { downloadCSV } from "../lib/csv";
import { fmt } from "../lib/format";
import { bestIndex, byDirection } from "../lib/metrics";
import type { Direction } from "../lib/types";
import { IconButton } from "./ui";

type Raw = string | number | boolean | null | undefined;

export interface Column<T> {
  key: string;
  header: ReactNode;
  /** Plain-text header for CSV and tooltips when `header` is not a string. */
  label?: string;
  title?: string;
  align?: "left" | "right" | "center";
  /** The raw value: drives sorting, the best-cell mark and CSV export. */
  value?: (row: T) => Raw;
  render?: (row: T) => ReactNode;
  /** Mark the best cell of this column, given which way is better. */
  best?: Direction;
  sortable?: boolean;
  className?: string;
}

interface SortState {
  key: string;
  desc: boolean;
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  onRowClick,
  isActive,
  stickyFirst = false,
  dense = false,
  maxHeight,
  caption,
}: {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T, i: number) => string;
  onRowClick?: (row: T) => void;
  isActive?: (row: T) => boolean;
  stickyFirst?: boolean;
  dense?: boolean;
  maxHeight?: number;
  caption?: string;
}) {
  const [sort, setSort] = useState<SortState | null>(null);
  const valueOf = (column: Column<T>, row: T): Raw => (column.value ? column.value(row) : (row as Record<string, Raw>)[column.key]);

  const bests = useMemo(() => {
    const map = new Map<string, number>();
    for (const column of columns) {
      if (column.best === undefined) continue;
      const values = rows.map((row) => {
        const v = valueOf(column, row);
        return typeof v === "number" ? v : null;
      });
      map.set(column.key, bestIndex(values, column.best));
    }
    return map;
  }, [columns, rows]);

  const ordered = useMemo(() => {
    const indexed = rows.map((row, i) => ({ row, i }));
    if (!sort) return indexed;
    const column = columns.find((c) => c.key === sort.key);
    if (!column) return indexed;
    const numeric = byDirection(column.best === undefined ? true : column.best);
    return [...indexed].sort((a, b) => {
      const va = valueOf(column, a.row);
      const vb = valueOf(column, b.row);
      let cmp: number;
      if (typeof va === "number" || typeof vb === "number") {
        cmp = numeric(typeof va === "number" ? va : null, typeof vb === "number" ? vb : null);
      } else {
        cmp = String(va ?? "").localeCompare(String(vb ?? ""));
      }
      return sort.desc ? -cmp : cmp;
    });
  }, [rows, sort, columns]);

  const toggleSort = (column: Column<T>) => {
    setSort((current) => {
      if (!current || current.key !== column.key) return { key: column.key, desc: false };
      if (!current.desc) return { key: column.key, desc: true };
      return null;
    });
  };

  const pad = dense ? "px-2.5 py-1.5" : "px-3 py-2";

  return (
    <div className="overflow-auto scroll-thin" style={maxHeight ? { maxHeight } : undefined}>
      <table className="w-full border-separate border-spacing-0 text-[13px]">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>
            {columns.map((column, ci) => {
              const sortable = column.sortable ?? Boolean(column.value || column.best !== undefined);
              const active = sort?.key === column.key;
              return (
                <th
                  key={column.key}
                  scope="col"
                  title={column.title}
                  aria-sort={active ? ariaSort(column, sort!.desc, rows.length ? valueOf(column, rows[0]) : null) : undefined}
                  className={clsx(
                    "sticky top-0 z-10 whitespace-nowrap border-b border-line bg-surface text-[11px] font-semibold uppercase tracking-[0.05em] text-fg-3",
                    pad,
                    alignClass(column.align),
                    stickyFirst && ci === 0 && "left-0 z-20",
                    ci === 0 && "pl-4 sm:pl-5",
                    ci === columns.length - 1 && "pr-4 sm:pr-5",
                  )}
                >
                  {sortable ? (
                    <button
                      type="button"
                      onClick={() => toggleSort(column)}
                      className={clsx("inline-flex items-center gap-1 uppercase hover:text-fg", active && "text-fg")}
                    >
                      {column.header}
                      {active && (sort!.desc ? <ArrowUp size={11} /> : <ArrowDown size={11} />)}
                    </button>
                  ) : (
                    column.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {ordered.map(({ row, i }) => {
            const active = isActive?.(row) ?? false;
            return (
              <tr
                key={rowKey(row, i)}
                tabIndex={onRowClick ? 0 : undefined}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                onKeyDown={
                  onRowClick
                    ? (event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          onRowClick(row);
                        }
                      }
                    : undefined
                }
                aria-selected={isActive ? active : undefined}
                className={clsx(
                  "group",
                  onRowClick && "cursor-pointer",
                  active ? "[&>td]:bg-accent-soft" : onRowClick && "hover:[&>td]:bg-surface-2",
                )}
              >
                {columns.map((column, ci) => {
                  const isBest = bests.get(column.key) === i;
                  const content = column.render ? column.render(row) : defaultRender(valueOf(column, row));
                  return (
                    <td
                      key={column.key}
                      title={isBest ? "best in column" : undefined}
                      className={clsx(
                        "border-b border-line/70 align-middle text-fg-2 transition-colors group-last:border-b-0",
                        pad,
                        alignClass(column.align),
                        column.align !== "left" && "num whitespace-nowrap",
                        stickyFirst && ci === 0 && "sticky left-0 z-[1] bg-surface",
                        ci === 0 && "pl-4 text-fg sm:pl-5",
                        ci === columns.length - 1 && "pr-4 sm:pr-5",
                        column.className,
                      )}
                    >
                      {isBest ? <span className="rounded bg-good-soft px-1.5 py-0.5 font-semibold text-good">{content}</span> : content}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** The first click sorts best first, so which way that runs depends on the column. */
function ariaSort<T>(column: Column<T>, desc: boolean, sample: Raw): "ascending" | "descending" | "other" {
  if (typeof sample !== "number") return desc ? "descending" : "ascending";
  if (column.best === null) return "other";
  const highFirst = column.best !== false;
  return highFirst !== desc ? "descending" : "ascending";
}

function alignClass(align: Column<unknown>["align"]) {
  if (align === "left") return "text-left";
  if (align === "center") return "text-center";
  return "text-right";
}

function defaultRender(value: Raw): ReactNode {
  if (typeof value === "number") return fmt(value);
  if (typeof value === "boolean") return value ? "yes" : "no";
  return value ?? "—";
}

/** A card-tools button that saves the table's raw values. */
export function ExportButton<T>({ columns, rows, filename }: { columns: Column<T>[]; rows: T[]; filename: string }) {
  const save = () => {
    const usable = columns.filter((c) => c.value || c.key);
    downloadCSV(
      filename,
      usable.map((c) => c.label ?? (typeof c.header === "string" ? c.header : c.key)),
      rows.map((row) => usable.map((c) => (c.value ? c.value(row) : (row as Record<string, Raw>)[c.key]))),
    );
  };
  return (
    <IconButton label="Download as CSV" onClick={save}>
      <Download size={15} />
    </IconButton>
  );
}
