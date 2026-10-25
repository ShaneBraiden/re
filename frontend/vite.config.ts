import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/* The Flask app in dashboard/app.py serves the built bundle: index.html at "/"
 * and everything else under "/app/". In development Vite serves the page and
 * forwards every data route to a running `sensorsim dashboard` on port 5000. */

const FLASK = process.env.SENSORSIM_API ?? "http://127.0.0.1:5000";
const proxied = ["/api", "/previews", "/figures", "/enhanced"];

export default defineConfig(({ command }) => ({
  base: command === "build" ? "/app/" : "/",
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(proxied.map((path) => [path, { target: FLASK, changeOrigin: true }])),
  },
  build: {
    outDir: "../dashboard/web",
    emptyOutDir: true,
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router"],
          charts: ["recharts"],
        },
      },
    },
  },
}));
