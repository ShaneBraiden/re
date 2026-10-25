import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router";
import App from "./App";
import { reloadForNewBuild } from "./components/ErrorBoundary";
import { RunProvider } from "./lib/run";
import "./styles.css";

// The upload blueprint serves this page at /upload: open it on the Enhance page.
if (window.location.pathname.startsWith("/upload") && !window.location.hash) {
  window.location.hash = "#/enhance";
}

// A chunk preload failing means the bundle was rebuilt under an open tab.
window.addEventListener("vite:preloadError", (event) => {
  if (reloadForNewBuild()) event.preventDefault();
});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, retry: 1 },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <HashRouter>
        <RunProvider>
          <App />
        </RunProvider>
      </HashRouter>
    </QueryClientProvider>
  </StrictMode>,
);
