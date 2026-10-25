import { lazy, Suspense, type ReactNode } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router";
import { Layout, PageHeader } from "./components/Layout";
import { Card, EmptyState, Loading, Skeleton } from "./components/ui";
import { useRun } from "./lib/run";

const Overview = lazy(() => import("./pages/Overview"));
const Explorer = lazy(() => import("./pages/Explorer"));
const Analysis = lazy(() => import("./pages/Analysis"));
const Enhance = lazy(() => import("./pages/Enhance"));
const EnhanceResult = lazy(() => import("./pages/EnhanceResult"));

export default function App() {
  const location = useLocation();
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to={{ pathname: "/overview", search: location.search }} replace />} />
        <Route path="overview" element={<Page title="Overview" needsIndex><Overview /></Page>} />
        <Route path="explore" element={<Page title="Explorer" needsIndex><Explorer /></Page>} />
        <Route path="analysis" element={<Page title="Analysis" needsIndex><Analysis /></Page>} />
        <Route path="enhance" element={<Page title="Enhance"><Enhance /></Page>} />
        <Route path="enhance/:id" element={<Page title="Enhancement"><EnhanceResult /></Page>} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}

/** Lazy-loads a page, and holds back pages that need the run's index until it is in. */
function Page({ title, needsIndex = false, children }: { title: string; needsIndex?: boolean; children: ReactNode }) {
  const { booting, indexLoading, indexError, index } = useRun();
  if (needsIndex && (booting || indexLoading)) return <PageSkeleton />;
  if (needsIndex && !index) {
    return (
      <>
        <PageHeader title={title} />
        <Card>
          <EmptyState
            title="No finished run to show yet"
            detail={indexError ?? "The dashboard reads the files a finished run wrote."}
            command="python -m sensorsim.cli run configs/experiments/pilot.yaml"
          />
        </Card>
      </>
    );
  }
  return <Suspense fallback={<Loading />}>{children}</Suspense>;
}

function PageSkeleton() {
  return (
    <div role="status" aria-label="Loading" className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <Skeleton className="h-8 w-56" />
        <Skeleton className="h-4 w-96 max-w-full" />
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        {Array.from({ length: 5 }, (_, i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
      <Skeleton className="h-96" />
    </div>
  );
}

function NotFound() {
  const { link } = useRun();
  return (
    <>
      <PageHeader title="Not found" />
      <Card>
        <EmptyState
          title="There is no page at this address"
          detail={
            <>
              Try the <a className="text-accent hover:underline" href={`#${link("/overview")}`}>overview</a>.
            </>
          }
        />
      </Card>
    </>
  );
}
