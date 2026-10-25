# SensorSim dashboard: front end

React 19 + TypeScript, built with Vite into `../dashboard/web`, which the Flask
app in `dashboard/app.py` serves at `/`. It only reads: every number comes from
the dashboard's GET routes over files an earlier run wrote. The one exception
is the Enhance page's upload, which needs `sensorsim dashboard --upload`.

```bash
npm install
npm run build       # typecheck, then bundle into ../dashboard/web
npm run dev         # http://localhost:5173, data proxied to the dashboard on :5000
npm run typecheck
```

`npm run dev` expects `python -m sensorsim.cli dashboard experiments/<run>` to be
running. Set `SENSORSIM_API` to point the proxy somewhere other than
`http://127.0.0.1:5000`.

## Layout

| Path | What lives there |
|------|------------------|
| `src/lib/api.ts`, `types.ts` | the only code that knows URLs, and the shapes the API returns |
| `src/lib/queries.ts` | TanStack Query hooks; a finished run is cached for the session |
| `src/lib/run.tsx` | which run is selected (`?exp=`), links that keep it, view state in the URL |
| `src/lib/pipelines.ts`, `metrics.ts` | names, colours, dash patterns, units and which way is better |
| `src/components/` | layout, cards, tables, the Recharts wrapper, the wipe/zoom viewer, the lightbox |
| `src/pages/` | Overview, Explorer, Analysis, Enhance and one enhancement run |
| `src/styles.css` | design tokens for light and dark, mapped to Tailwind utilities |

## Conventions

- **The URL is the state.** Routes live in the hash (`#/explore?exp=pilot&image=nat_00&sev=4`),
  so every view is a link, and links from the classic dashboard still open.
- **Colour follows the pipeline**, set once by name in `pipelines.ts`. Ablations
  wear their parent's hue with a dash pattern, so filtering never repaints anything.
- **Never colour alone.** Status badges carry an icon and a word, best cells are
  marked and titled, and every chart has a table view and a CSV export.
- **Tokens, not colours.** Components use `bg-surface`, `text-fg-2`, `border-line`
  and so on, so a theme switch is one attribute on `<html>`.
