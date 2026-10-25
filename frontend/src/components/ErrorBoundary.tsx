/* Keeps one broken page from blanking the whole app.
 *
 * The common case is benign: the frontend was rebuilt while a tab was open, so
 * the page asks for a chunk whose content hash no longer exists. That reloads
 * once, onto the new build. Anything else is shown where the page would be,
 * with the navigation still working around it. */

import { RefreshCw } from "lucide-react";
import { Component, type ReactNode } from "react";
import { Button, Card, EmptyState } from "./ui";

const RELOADED = "sensorsim-chunk-reload";

export const isChunkError = (error: unknown) =>
  error instanceof Error && /dynamically imported module|Importing a module script failed|error loading dynamically imported/i.test(error.message);

/** Reload onto a fresh build, at most once per tab session, so a genuinely missing file cannot loop. */
export function reloadForNewBuild(): boolean {
  try {
    if (sessionStorage.getItem(RELOADED)) return false;
    sessionStorage.setItem(RELOADED, "1");
  } catch {
    return false;
  }
  window.location.reload();
  return true;
}

export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error) {
    if (isChunkError(error)) reloadForNewBuild();
    else console.error(error);
  }

  componentDidUpdate(previous: { resetKey?: string }) {
    if (this.state.error && previous.resetKey !== this.props.resetKey) this.setState({ error: null });
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const stale = isChunkError(error);
    return (
      <Card>
        <EmptyState
          title={stale ? "The dashboard was rebuilt" : "This page hit an error"}
          detail={stale ? "This tab is still running the previous build. Reload to pick up the new one." : error.message}
        />
        <div className="-mt-4 flex justify-center pb-6">
          <Button variant="outline" onClick={() => window.location.reload()}>
            <RefreshCw size={14} /> Reload
          </Button>
        </div>
      </Card>
    );
  }
}
