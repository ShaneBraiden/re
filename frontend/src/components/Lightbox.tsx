/* Full-size viewer for figures and outputs. A native <dialog>, which brings its
 * own focus trap and Escape handling; arrow keys step through a gallery. */

import { ChevronLeft, ChevronRight, ExternalLink, X } from "lucide-react";
import { useEffect, useRef } from "react";
import { IconButton } from "./ui";

export interface LightboxItem {
  src: string;
  caption: string;
}

export function Lightbox({
  items,
  index,
  onIndex,
  onClose,
}: {
  items: LightboxItem[];
  index: number | null;
  onIndex?: (index: number) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const open = index !== null && items[index] !== undefined;

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  useEffect(() => {
    if (!open || !onIndex || items.length < 2) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight") onIndex((index! + 1) % items.length);
      if (event.key === "ArrowLeft") onIndex((index! - 1 + items.length) % items.length);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, index, items.length, onIndex]);

  const item = open ? items[index!] : null;
  const many = Boolean(onIndex) && items.length > 1;

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(event) => event.target === ref.current && onClose()}
      aria-label={item?.caption}
      className="lightbox m-auto h-[min(92vh,1000px)] w-[min(94vw,1400px)] max-w-none overflow-hidden rounded-xl border border-line bg-surface p-0 text-fg shadow-pop backdrop:bg-transparent"
    >
      {item && (
        <div className="flex h-full flex-col">
          <div className="flex items-center gap-2 border-b border-line px-3 py-2">
            <span className="min-w-0 flex-1 truncate text-[13px] font-medium">{item.caption}</span>
            {many && (
              <span className="num text-xs text-fg-3">
                {index! + 1} / {items.length}
              </span>
            )}
            <a
              href={item.src}
              target="_blank"
              rel="noopener"
              className="inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[13px] text-fg-2 hover:bg-surface-2 hover:text-fg"
            >
              <ExternalLink size={14} /> Open
            </a>
            <IconButton label="Close" onClick={onClose}>
              <X size={16} />
            </IconButton>
          </div>
          <div className="relative flex min-h-0 flex-1 items-center justify-center bg-[repeating-conic-gradient(var(--surface-2)_0%_25%,var(--surface)_0%_50%)] bg-[length:16px_16px] p-4">
            <img src={item.src} alt={item.caption} className="max-h-full max-w-full object-contain" />
            {many && (
              <>
                <NavButton side="left" onClick={() => onIndex!((index! - 1 + items.length) % items.length)} />
                <NavButton side="right" onClick={() => onIndex!((index! + 1) % items.length)} />
              </>
            )}
          </div>
        </div>
      )}
    </dialog>
  );
}

function NavButton({ side, onClick }: { side: "left" | "right"; onClick: () => void }) {
  const Icon = side === "left" ? ChevronLeft : ChevronRight;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={side === "left" ? "Previous" : "Next"}
      className={`absolute top-1/2 ${side === "left" ? "left-3" : "right-3"} flex size-10 -translate-y-1/2 items-center justify-center rounded-full border border-line bg-surface/90 text-fg shadow-pop backdrop-blur hover:bg-surface`}
    >
      <Icon size={20} />
    </button>
  );
}
