"""The dashboard backend - a read-only window onto finished runs.

Section 3: "Thin read-only layer over cached results. Its purpose is
demonstration and figure generation, not computation."

That is enforced here rather than merely intended. Every route is a GET, the app
opens files earlier stages already wrote - ``dashboard.json``, the preview JPEGs,
the manifest, the tables ``sensorsim analyze`` left in the figures directory,
the records of past enhancements - and nothing in this module imports the
degrader, the restorer or the evaluator. If the dashboard shows a number, some
earlier run computed it and it is on disk. The one summary it forms itself, the
mean over scenes in ``/api/aggregate``, averages those cached numbers and says
how many scenes each point stands on.

The run named on the command line is the default. Sibling runs in the same
parent directory that also have a ``dashboard.json`` are listed too, so the
frontend can switch between them without a restart; every data route takes an
optional ``?exp=<run directory name>``.

The single exception to read-only is opt-in: ``create_app(..., enable_upload=True)``
(or ``sensorsim dashboard --upload``) registers the blueprint in
``dashboard/upload.py``, which adds one POST route for enhancing an uploaded
image. Without the flag, the app is exactly the read-only app described above.

Serve it with::

    sensorsim dashboard experiments/pilot
"""

from __future__ import annotations

import csv
import json
import math
import mimetypes
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, request, send_from_directory

STATIC_DIR = Path(__file__).parent / "static"

# The frontend is ES modules, and a browser refuses to run a module served as
# text/plain - which is what the Windows registry maps .js to on some machines.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

#: Tables `sensorsim analyze` and `sensorsim paper-nag` write into a figures
#: directory. Whichever of them exist are served; none is required.
ANALYSIS_TABLES = (
    "knees",
    "significance",
    "recovery_aggregate",
    "planned_tests",
    "main_means",
    "ablation_means",
    "nag_training",
    "noise_validation",
    "photon_transfer",
)


def finite(obj: Any) -> Any:
    """NaN and infinity become null.

    ``json.dumps`` writes them as bare ``NaN``/``Infinity``, which is not JSON:
    the browser's parser rejects the whole response over one missing value.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: finite(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(value) for value in obj]
    return obj


def _cell(text: str | None) -> Any:
    if text is None or text == "":
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        number = float(text)
    except ValueError:
        return text
    return number if math.isfinite(number) else None


def read_table(path: Path) -> list[dict[str, Any]]:
    """A CSV as a list of rows, numbers parsed, blanks and NaN as None."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: _cell(value) for key, value in row.items()} for row in csv.DictReader(handle)]


class UnknownExperiment(LookupError):
    pass


class ExperimentStore:
    """Cached read of one experiment's results.

    The index is loaded once and reloaded only when the file's modification time
    changes, so a dashboard left open during a rerun picks up new results
    without a restart and without polling the filesystem on every request.
    """

    def __init__(self, root: str | Path):
        # Absolute: Flask resolves a relative send_from_directory path against
        # the app package directory, not the working directory, so a relative
        # root here silently serves 404s for files that are plainly there.
        self.root = Path(root).resolve()
        self.index_path = self.root / "dashboard.json"
        self._index: dict[str, Any] | None = None
        self._mtime: float = -1.0

    @property
    def id(self) -> str:
        return self.root.name

    @property
    def index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"no dashboard index at {self.index_path}. The dashboard reads a finished run - "
                f"produce one first:  sensorsim run configs/experiments/pilot.yaml"
            )
        mtime = self.index_path.stat().st_mtime
        if self._index is None or mtime != self._mtime:
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._mtime = mtime
        return self._index

    def _json(self, name: str) -> dict[str, Any] | None:
        path = self.root / name
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def manifest(self) -> dict[str, Any] | None:
        return self._json("manifest.json")

    def validation(self) -> dict[str, Any] | None:
        return self._json("validation.json")

    def conditions(self) -> list[dict[str, Any]]:
        return sorted(self.index["conditions"], key=lambda c: c["severity"])

    def values(self, image_id: str, condition_hash: str, pipeline: str) -> dict[str, float]:
        return self.index["values"].get(image_id, {}).get(condition_hash, {}).get(pipeline, {})

    def series(self, image_id: str, metric: str, pipelines: list[str]) -> dict[str, Any]:
        """One metric across the severity sweep, per pipeline - what the chart plots."""
        out: dict[str, list[dict[str, float | None]]] = {}
        for pipeline in pipelines:
            points = []
            for condition in self.conditions():
                value = self.values(image_id, condition["hash"], pipeline).get(metric)
                points.append(
                    {
                        "severity": condition["severity"],
                        "value": value,
                        "hash": condition["hash"],
                    }
                )
            out[pipeline] = points
        return {"image_id": image_id, "metric": metric, "series": out}

    def aggregate(self, metric: str, pipelines: list[str]) -> dict[str, Any]:
        """Mean, min and max of one metric over the scenes, per pipeline and severity.

        A summary of numbers already in the index: nothing is re-evaluated, and
        ``n`` says how many scenes each point stands on, so a point that lost a
        scene to a non-finite value says so instead of quietly averaging fewer.
        """
        images = [image["image_id"] for image in self.index["images"]]
        out: dict[str, list[dict[str, Any]]] = {}
        for pipeline in pipelines:
            points = []
            for condition in self.conditions():
                found = [self.values(image, condition["hash"], pipeline).get(metric) for image in images]
                cached = [v for v in found if isinstance(v, (int, float)) and math.isfinite(v)]
                points.append(
                    {
                        "severity": condition["severity"],
                        "hash": condition["hash"],
                        "mean": sum(cached) / len(cached) if cached else None,
                        "min": min(cached) if cached else None,
                        "max": max(cached) if cached else None,
                        "n": len(cached),
                    }
                )
            out[pipeline] = points
        return {"metric": metric, "n_images": len(images), "series": out}

    def summary(self) -> dict[str, Any]:
        index = self.index
        validation = self.validation()
        pipelines = index.get("pipelines", [])
        return {
            "id": self.id,
            "experiment_id": index.get("experiment_id", self.id),
            "images": len(index.get("images", [])),
            "conditions": len(index.get("conditions", [])),
            "pipelines": len(pipelines),
            "available": sum(1 for p in pipelines if p.get("available")),
            "validation": None if validation is None else bool(validation.get("passed")),
        }

    def figures_dir(self, figures_root: Path) -> Path | None:
        """Where ``sensorsim analyze`` wrote this run's tables, if anywhere.

        A run analysed with ``--figures paper/figures/<run>`` has its own folder.
        The shared default, ``paper/figures``, is claimed only by the run whose
        available pipelines are exactly the ones in its recovery table, so a
        second run never borrows the pilot's knees.
        """
        own = figures_root / self.id
        if own.is_dir():
            return own
        table = figures_root / "recovery_aggregate.csv"
        if table.is_file():
            names = {row["pipeline"] for row in read_table(table)}
            available = {p["name"] for p in self.index.get("pipelines", []) if p.get("available")}
            if names and names == available:
                return figures_root
        return None


class Catalog:
    """The default run plus every sibling run that has a dashboard index."""

    def __init__(self, default: str | Path):
        self.default = ExperimentStore(default)
        self._stores: dict[str, ExperimentStore] = {self.default.id: self.default}

    def discover(self) -> list[ExperimentStore]:
        parent = self.default.root.parent
        if parent.is_dir():
            for path in sorted(parent.iterdir()):
                if path.name not in self._stores and (path / "dashboard.json").is_file():
                    self._stores[path.name] = ExperimentStore(path)
        # A run deleted since the last look drops out; the default always stays.
        self._stores = {
            name: store for name, store in self._stores.items()
            if store is self.default or store.index_path.is_file()
        }
        return list(self._stores.values())

    def get(self, name: str | None) -> ExperimentStore:
        if not name or name == self.default.id:
            return self.default
        if name not in self._stores:
            self.discover()
        try:
            return self._stores[name]
        except KeyError:
            raise UnknownExperiment(f"no run named {name!r} next to {self.default.root}") from None


def _enhancement_view(record: dict[str, Any], folder: Path) -> dict[str, Any]:
    """A past enhancement as the frontend shows it: the record plus file URLs."""
    files = record.get("files", {})
    urls = {
        key: f"/enhanced/{folder.name}/{Path(path).name}"
        for key, path in files.items()
        if (folder / Path(path).name).is_file()
    }
    keep = ("name", "kind", "scale", "input_shape", "output_shape", "steps", "metrics",
            "seconds", "features", "noise_level_function", "checkpoints")
    view = {key: record.get(key) for key in keep}
    view.update(id=folder.name, urls=urls, modified=(folder / "metrics.json").stat().st_mtime)
    return finite(view)


def create_app(experiment: str | Path = "experiments/pilot", enable_upload: bool = False,
               upload_options: dict[str, Any] | None = None,
               figures: str | Path = "paper/figures") -> Flask:
    app = Flask(__name__, static_folder=None)
    catalog = Catalog(experiment)
    store = catalog.default
    figures_root = Path(figures).resolve()
    options = dict(upload_options or {})
    enhance_root = Path(options.get("out_root", "experiments/enhance")).resolve()

    if enable_upload:
        # The one exception to read-only, and only on request: see dashboard/upload.py.
        from dashboard.upload import MAX_BYTES, create_upload_blueprint

        app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
        app.register_blueprint(create_upload_blueprint(**options))

    def resolve() -> ExperimentStore:
        return catalog.get(request.args.get("exp"))

    def pipelines_arg() -> list[str]:
        return [p for p in request.args.get("pipelines", "").split(",") if p]

    @app.errorhandler(FileNotFoundError)
    def missing_run(exc: FileNotFoundError):
        return jsonify({"error": str(exc)}), 404

    @app.errorhandler(UnknownExperiment)
    def unknown_run(exc: UnknownExperiment):
        return jsonify({"error": str(exc)}), 404

    @app.get("/")
    def index_page():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(STATIC_DIR, filename)

    @app.get("/previews/<path:filename>")
    def previews(filename: str):
        preview_root = resolve().root / "previews"
        if not preview_root.exists():
            abort(404, description="this run wrote no previews")
        return send_from_directory(preview_root, filename, max_age=3600)

    @app.get("/figures/<path:filename>")
    def figure_files(filename: str):
        if not figures_root.exists():
            abort(404)
        return send_from_directory(figures_root, filename)

    @app.get("/enhanced/<path:filename>")
    def enhanced(filename: str):
        # Read-only: these are files a finished enhancement run already wrote.
        if not enhance_root.exists():
            abort(404)
        return send_from_directory(enhance_root, filename, max_age=3600)

    @app.get("/api/experiments")
    def api_experiments():
        runs = []
        for run in catalog.discover():
            try:
                runs.append(run.summary() | {"default": run is store})
            except (FileNotFoundError, ValueError, KeyError):
                runs.append({"id": run.id, "default": run is store, "error": "no readable dashboard index"})
        return jsonify({"default": store.id, "experiments": runs})

    @app.get("/api/index")
    def api_index():
        data = resolve().index
        # The values blob can be large and the client fetches slices on demand.
        summary = {k: v for k, v in data.items() if k != "values"}
        summary["has_values"] = bool(data.get("values"))
        return jsonify(finite(summary))

    @app.get("/api/values")
    def api_values():
        run = resolve()
        image = request.args.get("image", "")
        condition = request.args.get("condition", "")
        if not image or not condition:
            return jsonify({"error": "image and condition are required"}), 400
        return jsonify(finite({
            "image_id": image,
            "condition": condition,
            "values": {p: run.values(image, condition, p) for p in pipelines_arg()},
        }))

    @app.get("/api/series")
    def api_series():
        run = resolve()
        image = request.args.get("image", "")
        metric = request.args.get("metric", "psnr")
        pipelines = pipelines_arg()
        if not image or not pipelines:
            return jsonify({"error": "image and pipelines are required"}), 400
        return jsonify(finite(run.series(image, metric, pipelines)))

    @app.get("/api/aggregate")
    def api_aggregate():
        run = resolve()
        metric = request.args.get("metric", "psnr")
        pipelines = pipelines_arg()
        if not pipelines:
            return jsonify({"error": "pipelines are required"}), 400
        return jsonify(finite(run.aggregate(metric, pipelines)))

    @app.get("/api/run")
    def api_run():
        run = resolve()
        manifest = run.manifest() or {}
        gaps: dict[str, dict[str, Any]] = {}
        for gap in run.index.get("gaps", []):
            entry = gaps.setdefault(gap["pipeline"], {"pipeline": gap["pipeline"], "reason": gap["reason"], "rows": 0})
            entry["rows"] += 1
        keep = ("experiment_id", "config_hash", "seed", "started_at", "finished_at", "git_commit",
                "hardware_info", "packages", "stages", "weights")
        return jsonify(finite({
            "id": run.id,
            "manifest": {key: manifest.get(key) for key in keep} if manifest else None,
            "config": manifest.get("config"),
            "validation": run.validation(),
            "gaps": list(gaps.values()),
        }))

    @app.get("/api/analysis")
    def api_analysis():
        run = resolve()
        folder = run.figures_dir(figures_root)
        if folder is None:
            return jsonify({"id": run.id, "figures_dir": None, "tables": {}, "divergence": None, "figures": []})
        tables = {
            name: read_table(folder / f"{name}.csv")
            for name in ANALYSIS_TABLES
            if (folder / f"{name}.csv").is_file()
        }
        divergence_path = folder / "divergence.json"
        divergence = json.loads(divergence_path.read_text(encoding="utf-8")) if divergence_path.is_file() else None
        figures = [
            {"name": png.stem, "url": f"/figures/{png.relative_to(figures_root).as_posix()}"}
            for png in sorted(folder.glob("*.png"))
        ]
        shown = folder.relative_to(Path.cwd()) if folder.is_relative_to(Path.cwd()) else folder
        return jsonify(finite({
            "id": run.id,
            "figures_dir": shown.as_posix(),
            "tables": tables,
            "divergence": divergence,
            "figures": figures,
        }))

    @app.get("/api/enhancements")
    def api_enhancements():
        runs = []
        if enhance_root.is_dir():
            for folder in enhance_root.iterdir():
                record_path = folder / "metrics.json"
                if not record_path.is_file():
                    continue
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                runs.append(_enhancement_view(record, folder))
        runs.sort(key=lambda r: r["modified"], reverse=True)
        return jsonify({"upload": enable_upload, "runs": runs})

    @app.get("/api/health")
    def api_health():
        ready = store.index_path.exists()
        return jsonify(
            {
                "ready": ready,
                "upload": enable_upload,
                "experiment": str(store.root),
                "default": store.id,
                "index": str(store.index_path),
                "note": "" if ready else "run an experiment first: sensorsim run configs/experiments/pilot.yaml",
            }
        ), (200 if ready else 503)

    return app


def main() -> int:  # pragma: no cover - convenience entry point
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", default="experiments/pilot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    app = create_app(args.experiment, enable_upload=args.upload)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
