"""The run orchestrator - the only place the four stages meet.

Stages still do not import each other. This module reads a config, walks the
corpus, calls each stage in turn and writes the files they exchange. Everything
it produces is a file: degraded frames, restored frames, a tidy metrics table, a
run manifest and a dashboard index.

Resumability comes from two mechanisms working together: Stage 2 caches
restored frames by content hash, and this runner skips any
``(image, condition, pipeline)`` triple already present in an existing metrics
table for the same config hash. Interrupt a sweep and rerun it, and it picks up
where it stopped.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import pandas as pd

from sensorsim.config import build_sweep, load_experiment, pipeline_specs
from sensorsim.degrader.model import DegradationParams, degrade
from sensorsim.degrader.noise import SensorNoiseModel
from sensorsim.degrader.validate import ValidationError, self_consistency
from sensorsim.evaluate import (
    DIRECTION,
    FAMILY_OF,
    MetricRow,
    evaluate_frame,
    missing_metrics,
    unavailable_rows,
)
from sensorsim.io.manifest import RunManifest, content_hash
from sensorsim.io.raw import RawImage, load_raw
from sensorsim.restore import RestoreContext, build_bank, run_pipeline

METRICS_COLUMNS = [
    "image_id",
    "degradation_params_hash",
    "pipeline",
    "metric_family",
    "metric_name",
    "value",
    "scene_kind",
    "severity",
    "status",
    "note",
]

PREVIEW_QUALITY = 88


class CorpusEntry:
    """One clean frame plus its ground-truth reference."""

    def __init__(self, raw_path: Path, reference_path: Path, scene_kind: str = "natural"):
        self.raw_path = raw_path
        self.reference_path = reference_path
        self.scene_kind = scene_kind

    def load(self) -> tuple[RawImage, np.ndarray]:
        raw = load_raw(self.raw_path)
        bgr = cv2.imread(str(self.reference_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"reference image not readable: {self.reference_path}")
        return raw, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_corpus(config: dict[str, Any]) -> list[CorpusEntry]:
    """Resolve the corpus described by a config, building it if it is synthetic."""
    spec = config.get("corpus", {})
    kind = spec.get("kind", "synthetic")
    root = Path(spec.get("path", "data/synthetic"))

    if kind == "synthetic":
        index_path = root / "index.json"
        if not index_path.exists() or spec.get("rebuild", False):
            from sensorsim.data.synthetic import build_corpus

            build_corpus(
                root,
                kinds=tuple(spec.get("kinds", ("resolution", "text", "natural"))),
                per_kind=int(spec.get("per_kind", 1)),
                size=tuple(spec.get("size", (512, 512))),  # type: ignore[arg-type]
            )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return [CorpusEntry(root / e["raw"], root / e["reference"], e["scene_kind"]) for e in index]

    if kind == "directory":
        raws = sorted(Path(spec["raw_dir"]).glob(spec.get("glob", "*.npy")))
        ref_dir = Path(spec.get("reference_dir", ""))
        entries = []
        for raw_path in raws:
            reference = ref_dir / f"{raw_path.stem}.png" if ref_dir else raw_path.with_suffix(".png")
            if not reference.exists():
                raise FileNotFoundError(
                    f"no reference image for {raw_path.name}. Every clean frame needs the "
                    "ground-truth rendering the fidelity metrics measure against - for a real "
                    "corpus that is the long-exposure or retouched reference, not a re-render "
                    "of the same RAW."
                )
            entries.append(CorpusEntry(raw_path, reference, spec.get("scene_kind", "natural")))
        if not entries:
            raise FileNotFoundError(f"no RAW frames matched in {spec['raw_dir']}")
        return entries

    raise ValueError(f"unknown corpus kind {kind!r}; expected synthetic or directory")


def run_validation_gate(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Phase 1's gate. Refuses to proceed on a model that fails calibration.

    ``mode: hardware`` is the real gate and needs measured captures.
    ``mode: self`` checks the generator against its own theory - it catches
    implementation bugs but says nothing about whether the model matches any
    real sensor, so a run made under it is marked accordingly in the manifest
    and the paper must not claim calibration.
    """
    spec = config.get("validation", {})
    mode = spec.get("mode", "self")

    if mode == "skip":
        return {
            "mode": "skip",
            "passed": None,
            "note": "VALIDATION SKIPPED - results are not calibrated against any sensor",
        }

    if mode == "self":
        model = SensorNoiseModel(
            full_well=float(spec.get("full_well", 15000.0)),
            sigma_read_dn=float(spec.get("sigma_read_dn", 4.0)),
            dark_current_e=0.0,
            prnu=0.0,
        )
        report = self_consistency(model, seed=int(config.get("seed", 0)))
        report.write(out_dir / "validation.json")
        if not report.passed:
            raise ValidationError(report.summary())
        return {
            "mode": "self",
            "passed": True,
            "summary": report.summary(),
            "note": (
                "self-consistency only. The forward model reproduces its own constants; it has "
                "NOT been compared against a real sensor. Section 3's gate needs mode: hardware."
            ),
        }

    if mode == "hardware":
        from sensorsim.degrader.validate import Tolerance, validate

        calib = Path(spec.get("calibration_dir", ""))
        synthetic = calib / "synthetic_stack.npy"
        measured = calib / "measured_stack.npy"
        if not synthetic.exists() or not measured.exists():
            raise ValidationError(
                f"hardware validation needs {synthetic} and {measured}: stacks of >= 8 captures of "
                "one static scene, normalised to [0, 1]. Shoot them on a tripod at fixed exposure."
            )
        edges = {}
        for name in ("synthetic_edge.npy", "measured_edge.npy"):
            if (calib / name).exists():
                edges[name.split("_")[0]] = np.load(calib / name)

        report = validate(
            np.load(synthetic),
            np.load(measured),
            edges.get("synthetic"),
            edges.get("measured"),
            Tolerance(**spec.get("tolerance", {})),
        )
        report.write(out_dir / "validation.json")
        report.require_pass()
        return {"mode": "hardware", "passed": True, "summary": report.summary()}

    raise ValueError(f"unknown validation mode {mode!r}; expected self, hardware or skip")


def _write_preview(path: Path, image: np.ndarray, max_side: int = 768) -> None:
    """Write a JPEG preview for the dashboard, downscaled if large."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])


def _degraded_preview(degraded: RawImage, target_shape: tuple[int, int]) -> np.ndarray:
    """What the degraded frame looks like with no restoration at all.

    A plain gained demosaic, so the dashboard's "before" panel is the honest
    input rather than the control pipeline's output.
    """
    from sensorsim.degrader.sampling import upsample_to
    from sensorsim.restore.classical import apply_white_balance, demosaic, to_uint8, tone_curve

    rgb = apply_white_balance(demosaic(degraded), degraded.wb_gains)
    return to_uint8(tone_curve(upsample_to(rgb, target_shape, order=1)))


def run(
    config_path: str | Path,
    resume: bool = True,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Run one experiment end to end. Returns a summary dict."""
    config = load_experiment(config_path)
    out_dir = Path(config["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    preview_dir = out_dir / "previews"

    say = progress or (lambda _: None)
    started = time.perf_counter()

    manifest = RunManifest.create(config["experiment_id"], config, int(config["seed"]))
    gate = run_validation_gate(config, out_dir)
    manifest.record_stage("validation", **gate)
    say(f"validation [{gate['mode']}]: {gate.get('note', gate.get('summary', ''))}")

    corpus = load_corpus(config)
    sweep = build_sweep(config["degradation"])
    bank = build_bank(pipeline_specs(config["pipelines"]))
    families = tuple(config.get("metrics", {}).get("families", ("fidelity", "perceptual", "machine_task")))

    say(f"{len(corpus)} images x {len(sweep)} conditions x {len(bank)} pipelines")
    for pipe in bank:
        availability = pipe.availability()
        if not availability:
            say(f"  unavailable: {pipe.name} - {availability.reason}")

    metrics_path = out_dir / "metrics.csv"
    done: set[tuple[str, str, str]] = set()
    existing = pd.DataFrame(columns=METRICS_COLUMNS)
    if resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        if not existing.empty:
            done = set(
                zip(existing["image_id"], existing["degradation_params_hash"], existing["pipeline"])
            )
            say(f"resuming: {len(done)} (image, condition, pipeline) triples already measured")

    rows: list[MetricRow] = []
    records: list[dict[str, Any]] = []

    for entry in corpus:
        clean, reference = entry.load()
        crops = clean.extra.get("crops")
        # PRNU is a property of the sensor, not of the condition, so the field is
        # drawn once per image (per binned resolution) and reused across the
        # sweep. Re-rolling it per condition would turn fixed-pattern noise into
        # another source of random noise, and fixed-pattern noise is precisely
        # the part that does not average away.
        prnu_rng = np.random.default_rng(int(content_hash(clean.image_id, 8), 16))
        prnu_field = prnu_rng.normal(1.0, SensorNoiseModel().prnu, size=clean.shape)

        for params in sweep:
            degraded, record = degrade(
                clean, params, seed=int(config["seed"]), prnu_field=prnu_field
            )
            records.append(record.to_dict())

            context = RestoreContext(
                image_id=clean.image_id,
                target_shape=clean.shape,
                degradation=record.params,
                sensor=record.sensor,
                scene_kind=entry.scene_kind,
            )

            preview_base = preview_dir / clean.image_id
            if not (preview_base / "reference.jpg").exists():
                _write_preview(preview_base / "reference.jpg", reference)
            degraded_preview_path = preview_base / f"{params.params_hash}__degraded.jpg"
            if not degraded_preview_path.exists():
                _write_preview(degraded_preview_path, _degraded_preview(degraded, clean.shape))

            for pipe in bank:
                key = (clean.image_id, params.params_hash, pipe.name)
                if key in done:
                    continue

                result = run_pipeline(pipe, degraded, context, cache_dir=cache_dir)
                if not result.ok:
                    rows.extend(
                        unavailable_rows(
                            image_id=clean.image_id,
                            degradation_params_hash=params.params_hash,
                            pipeline=pipe.name,
                            scene_kind=entry.scene_kind,
                            severity=params.severity,
                            reason=result.error,
                        )
                    )
                    continue

                _write_preview(preview_base / f"{params.params_hash}__{pipe.name}.jpg", result.image)
                rows.extend(
                    evaluate_frame(
                        result.image,
                        reference,
                        image_id=clean.image_id,
                        degradation_params_hash=params.params_hash,
                        pipeline=pipe.name,
                        scene_kind=entry.scene_kind,
                        severity=params.severity,
                        crops=crops,
                        families=families,
                    )
                )
            say(f"  {clean.image_id} severity {params.severity:.2f} done")

    new = pd.DataFrame([r.as_dict() for r in rows], columns=METRICS_COLUMNS)
    # Concatenate only the frames that hold rows: a fully resumed run produces an
    # empty `new`, and pandas warns about inferring dtypes from empty entries.
    parts = [frame for frame in (existing, new) if not frame.empty]
    table = pd.concat(parts, ignore_index=True) if len(parts) > 1 else (parts[0] if parts else new)
    table = table.drop_duplicates(
        subset=["image_id", "degradation_params_hash", "pipeline", "metric_name"], keep="last"
    )
    table.to_csv(metrics_path, index=False)

    with (out_dir / "degradations.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    manifest.record_stage(
        "degrade", conditions=len(sweep), frames=len(corpus) * len(sweep),
        order=sweep[0].order if sweep else "n/a",
    )
    manifest.record_stage(
        "restore", pipelines=[p.name for p in bank],
        unavailable={p.name: p.availability().reason for p in bank if not p.availability()},
    )
    manifest.weights = {p.name: p.weights_version() for p in bank}
    manifest.record_stage(
        "evaluate", rows=int(len(table)), families=list(families),
        metrics_unavailable=missing_metrics(),
    )
    manifest.finish()
    manifest.write(out_dir / "manifest.json")

    index = write_dashboard_index(out_dir, config, sweep, bank, table)

    summary = {
        "experiment_id": config["experiment_id"],
        "output": str(out_dir),
        "images": len(corpus),
        "conditions": len(sweep),
        "pipelines": [p.name for p in bank],
        "rows": int(len(table)),
        "seconds": round(time.perf_counter() - started, 1),
        "metrics_path": str(metrics_path),
        "dashboard_index": str(index),
        "reproducibility_warnings": manifest.reproducibility_report(),
    }
    say(json.dumps(summary, indent=2))
    return summary


def write_dashboard_index(
    out_dir: Path,
    config: dict[str, Any],
    sweep: Iterable[DegradationParams],
    bank: Iterable[Any],
    table: pd.DataFrame,
) -> Path:
    """Everything the dashboard needs, in one JSON file.

    The dashboard is a read-only layer over cached results (section 3), so it
    reads this and never recomputes anything.
    """
    sweep = list(sweep)
    bank = list(bank)

    ok = table[table["status"] == "ok"] if "status" in table.columns else table
    values: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for row in ok.itertuples(index=False):
        values.setdefault(row.image_id, {}).setdefault(row.degradation_params_hash, {}).setdefault(
            row.pipeline, {}
        )[row.metric_name] = None if pd.isna(row.value) else float(row.value)

    gaps = []
    if "status" in table.columns:
        for row in table[table["status"] != "ok"].itertuples(index=False):
            gaps.append(
                {
                    "image_id": row.image_id,
                    "degradation": row.degradation_params_hash,
                    "pipeline": row.pipeline,
                    "reason": str(row.note),
                }
            )

    images = sorted(values) or sorted({e.stem for e in (out_dir / "previews").glob("*")})
    index = {
        "experiment_id": config["experiment_id"],
        "generated_from": str(out_dir / "metrics.csv"),
        "images": [
            {"image_id": image_id, "reference": f"previews/{image_id}/reference.jpg"} for image_id in images
        ],
        "conditions": [
            {
                "hash": p.params_hash,
                "severity": p.severity,
                "tier": p.tier,
                "label": p.describe(),
                "params": asdict(p),
            }
            for p in sorted(sweep, key=lambda p: p.severity)
        ],
        "pipelines": [
            {
                "name": p.name,
                "kind": p.kind,
                "available": bool(p.availability()),
                "reason": p.availability().reason,
            }
            for p in bank
        ],
        "metrics": sorted(set(ok["metric_name"])) if len(ok) else [],
        # Direction and family travel with the data so the dashboard never keeps
        # its own copy of which metrics are better when larger - one source of
        # truth, in sensorsim.evaluate.
        "directions": {k: v for k, v in DIRECTION.items()},
        "families": {k: v for k, v in FAMILY_OF.items()},
        "values": values,
        "gaps": gaps[:200],
    }
    path = out_dir / "dashboard.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    return path
