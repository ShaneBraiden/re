"""Command line entry point.

    sensorsim corpus     build the synthetic corpus
    sensorsim validate   run the calibration gate on its own
    sensorsim run        degrade -> restore -> evaluate, writing a metrics table
    sensorsim analyze    fit recovery curves, locate knees, regenerate figures
    sensorsim reproduce  regenerate every number and figure, and diff them
    sensorsim pipelines  list registered pipeline kinds and what is available
    sensorsim dashboard  serve the read-only dashboard over a finished run
    sensorsim weights    fetch or verify the pinned checkpoints
    sensorsim train      train NC-DCE (the proposed curve estimator)
    sensorsim enhance    enhance one image with NAG; update the paper; write reports
    sensorsim paper-nag  regenerate the NAG section, tables and method report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_corpus(args: argparse.Namespace) -> int:
    from sensorsim.data.synthetic import build_corpus

    index = build_corpus(
        args.out,
        kinds=tuple(args.kinds),
        per_kind=args.per_kind,
        size=(args.size, args.size),
    )
    print(f"wrote {len(index)} frames to {args.out}")
    for entry in index:
        print(f"  {entry['image_id']:10s} {entry['scene_kind']:11s} {entry['shape']}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from sensorsim.config import load_experiment
    from sensorsim.degrader.validate import ValidationError
    from sensorsim.experiment import run_validation_gate

    config = load_experiment(args.config)
    out_dir = Path(config["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        gate = run_validation_gate(config, out_dir)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.figure:
        _write_photon_transfer_figure(config, Path(args.figures))
        print(f"photon transfer figure written to {args.figures}/photon_transfer.png")

    print(json.dumps(gate, indent=2))
    return 0


def _write_photon_transfer_figure(config: dict, out_dir: Path) -> Path:
    """The calibration figure, from measured captures when they exist."""
    import numpy as np

    from sensorsim.analyze.figures import photon_transfer_figure, save_figure
    from sensorsim.degrader.noise import SensorNoiseModel, photon_transfer_curve
    from sensorsim.degrader.validate import synthetic_photon_transfer

    spec = config.get("validation", {})
    model = SensorNoiseModel(
        full_well=float(spec.get("full_well", 15000.0)),
        sigma_read_dn=float(spec.get("sigma_read_dn", 4.0)),
        dark_current_e=0.0,
        prnu=0.0,
    )
    synthetic = synthetic_photon_transfer(model, seed=int(config.get("seed", 0)))

    measured = None
    calibration = Path(spec.get("calibration_dir") or "")
    if calibration and (calibration / "measured_stack.npy").exists():
        mean, variance, _ = photon_transfer_curve(np.load(calibration / "measured_stack.npy"))
        measured = (mean, variance)

    fig, data = photon_transfer_figure(synthetic, measured)
    return save_figure(fig, out_dir / "photon_transfer.png", data, f"sensorsim validate {config['experiment_id']}")


def _cmd_run(args: argparse.Namespace) -> int:
    from sensorsim.degrader.validate import ValidationError
    from sensorsim.experiment import run

    try:
        run(args.config, resume=not args.no_resume)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    import pandas as pd

    from sensorsim.analyze.curves import aggregate, recovery_ratio
    from sensorsim.analyze.figures import generate_all
    from sensorsim.analyze.knee import knee_table
    from sensorsim.analyze.stats import compare_pipelines, divergence_test

    metrics = pd.read_csv(args.metrics)
    provenance = f"sensorsim {args.metrics}"

    recovery = recovery_ratio(metrics, baseline=args.baseline)
    agg = aggregate(recovery, value_column="recovery")

    # The baseline is flat at R = 0 by construction, so it has no knee to find.
    knees = knee_table(agg[agg["pipeline"] != args.baseline], metric=args.metric)

    ok = metrics[metrics["status"] == "ok"]
    tests = compare_pipelines(ok, metric=args.metric, baseline=args.baseline)
    # Per pipeline, not pooled: pooling mixes arms that diverge with arms that
    # do not, and the average of the two says nothing about either.
    divergence = {
        "pooled": divergence_test(ok),
        "per_pipeline": {p: divergence_test(ok, pipeline=p) for p in sorted(set(ok["pipeline"]))},
    }

    out_dir = Path(args.figures)
    written = generate_all(metrics, out_dir=out_dir, metric=args.metric, baseline=args.baseline,
                           provenance=provenance)

    tables = Path(args.tables or out_dir)
    tables.mkdir(parents=True, exist_ok=True)
    agg.to_csv(tables / "recovery_aggregate.csv", index=False)
    knees.to_csv(tables / "knees.csv", index=False)
    tests.to_csv(tables / "significance.csv", index=False)
    (tables / "divergence.json").write_text(json.dumps(divergence, indent=2), encoding="utf-8")

    print(f"figures ({len(written)}):")
    for path in written:
        print(f"  {path}")
    print("\nknees:")
    print(knees[["pipeline", "knee", "agreement", "identifiable", "curve_model", "curve_r2"]].to_string(index=False))
    print("\nH2 divergence test:")
    print(json.dumps(divergence, indent=2))
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    """Regenerate every number and figure for one experiment, from scratch.

    Section 7's requirement. Deletes the restoration cache and the metrics table
    first, because a cache hit would prove only that the cache still works.
    If a previous ``metrics.csv`` was kept, the new one is compared against it
    cell by cell and any drift is reported as a failure - that comparison is the
    whole point, and a run that merely completes has not reproduced anything.
    """
    import shutil

    import pandas as pd

    from sensorsim.config import load_experiment
    from sensorsim.experiment import run

    config_path = Path(args.config)
    if not config_path.exists():
        candidate = Path("configs/experiments") / f"{args.config}.yaml"
        if not candidate.exists():
            print(f"no such experiment config: {args.config}", file=sys.stderr)
            return 1
        config_path = candidate

    config = load_experiment(config_path)
    out_dir = Path(config["output"])
    metrics_path = out_dir / "metrics.csv"

    previous = pd.read_csv(metrics_path) if metrics_path.exists() else None
    if previous is not None:
        keep = out_dir / "metrics.previous.csv"
        shutil.copy(metrics_path, keep)
        print(f"kept the existing table at {keep} for comparison")

    shutil.rmtree(out_dir / "cache", ignore_errors=True)
    metrics_path.unlink(missing_ok=True)

    print(f"== reproducing {config['experiment_id']} from {config_path}")
    run(config_path, resume=False)

    if previous is not None:
        current = pd.read_csv(metrics_path)
        drift = _compare_tables(previous, current)
        if drift:
            print("\nREPRODUCTION FAILED - the numbers changed:", file=sys.stderr)
            for line in drift:
                print(f"  {line}", file=sys.stderr)
            return 1
        print(f"\nreproduced: {len(current)} rows identical to the previous run")

    print("\n== regenerating figures")
    return _cmd_analyze(
        argparse.Namespace(
            metrics=str(metrics_path),
            metric=args.metric,
            baseline=args.baseline,
            figures=args.figures,
            tables=args.tables,
        )
    )


def _compare_tables(previous, current) -> list[str]:
    """Cell-by-cell differences between two metrics tables."""
    import numpy as np
    import pandas as pd

    key = ["image_id", "degradation_params_hash", "pipeline", "metric_name"]
    merged = previous.merge(current, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)

    problems: list[str] = []
    only_old = int((merged["_merge"] == "left_only").sum())
    only_new = int((merged["_merge"] == "right_only").sum())
    if only_old:
        problems.append(f"{only_old} rows disappeared")
    if only_new:
        problems.append(f"{only_new} rows appeared")

    both = merged[merged["_merge"] == "both"]
    old, new = both["value_old"].to_numpy(float), both["value_new"].to_numpy(float)
    finite = np.isfinite(old) & np.isfinite(new)
    changed = ~np.isclose(old[finite], new[finite], rtol=1e-9, atol=1e-12)
    if changed.any():
        rows = both[finite].iloc[np.flatnonzero(changed)]
        problems.append(f"{int(changed.sum())} values changed, e.g.:")
        for row in rows.head(5).itertuples(index=False):
            problems.append(
                f"    {row.image_id} {row.pipeline} {row.metric_name}: "
                f"{row.value_old:.6g} -> {row.value_new:.6g}"
            )
    return problems


def _cmd_pipelines(args: argparse.Namespace) -> int:
    from sensorsim.restore import build_pipeline, registered_kinds

    print(f"{'kind':16s} {'available':10s} reason")
    for kind in registered_kinds():
        pipe = build_pipeline({"kind": kind})
        availability = pipe.availability()
        print(f"{kind:16s} {'yes' if availability else 'no':10s} {availability.reason}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from dashboard.app import create_app

    app = create_app(args.experiment, enable_upload=args.upload)
    print(f"serving {args.experiment} at http://{args.host}:{args.port}")
    if args.upload:
        print(f"image upload enabled at http://{args.host}:{args.port}/upload")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def _cmd_weights(args: argparse.Namespace) -> int:
    from sensorsim.enhance import weights

    if args.action == "fetch":
        for line in weights.fetch(args.only.split(",") if args.only else None):
            print(line)
        return 0
    ok_all = True
    for name in weights.CHECKPOINTS:
        try:
            ok, line = weights.verify(name)
        except weights.MissingWeights as exc:
            ok, line = False, str(exc)
        ok_all &= ok
        print(line)
    return 0 if ok_all else 1


def _cmd_train(args: argparse.Namespace) -> int:
    from sensorsim.train.nc_dce import train_all

    only = args.variant.split(",") if args.variant else None
    train_all(args.config, only=only, epochs=args.epochs, device=args.device)
    return 0


def _cmd_enhance(args: argparse.Namespace) -> int:
    from sensorsim.enhance.pipeline import run_upload

    outcome = run_upload(
        args.image,
        kind=args.kind,
        device=args.device,
        update_paper=not args.no_paper,
        write_report=not args.no_report,
        tex=args.tex,
        report_dir=args.report_dir,
        cover={"title": args.title, "student": args.student, "uid": args.uid},
        rebuild_zip=args.zip,
    )
    print(json.dumps(outcome["summary"], indent=2))
    return 0


def _cmd_paper_nag(args: argparse.Namespace) -> int:
    from sensorsim.analyze.proposed import build_paper_assets
    from sensorsim.enhance.latex import update_paper_with_nag
    from sensorsim.enhance.report import write_method_report

    assets = build_paper_assets(args.metrics, training_dir=args.training)
    for line in assets["written"]:
        print(f"  wrote {line}")
    if not args.no_tex:
        result = update_paper_with_nag(args.tex, rebuild_zip=not args.no_zip)
        print(f"paper updated: {result['tex']} (backup {result['backup']})")
    if not args.no_report:
        uploads = sorted(p for p in Path("experiments/enhance").glob("*") if (p / "metrics.json").exists())
        pdf = write_method_report(assets, out_dir=args.report_dir, upload_dirs=uploads)
        print(f"method report: {pdf}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sensorsim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("corpus", help="build the synthetic corpus")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--kinds", nargs="+", default=["resolution", "text", "natural"])
    p.add_argument("--per-kind", type=int, default=1)
    p.add_argument("--size", type=int, default=512)
    p.set_defaults(func=_cmd_corpus)

    p = sub.add_parser("validate", help="run the calibration gate")
    p.add_argument("config")
    p.add_argument("--figure", action="store_true", help="also write the photon transfer figure")
    p.add_argument("--figures", default="paper/figures")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("run", help="degrade, restore and evaluate")
    p.add_argument("config")
    p.add_argument("--no-resume", action="store_true", help="recompute everything")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("analyze", help="curves, knees, significance, figures")
    p.add_argument("metrics", help="path to metrics.csv")
    p.add_argument("--metric", default="psnr")
    p.add_argument("--baseline", default="classical_isp")
    p.add_argument("--figures", default="paper/figures")
    p.add_argument("--tables", default=None)
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("reproduce", help="regenerate every number and figure from scratch")
    p.add_argument("config", help="path to an experiment config, or its id")
    p.add_argument("--metric", default="psnr")
    p.add_argument("--baseline", default="classical_isp")
    p.add_argument("--figures", default="paper/figures")
    p.add_argument("--tables", default=None)
    p.set_defaults(func=_cmd_reproduce)

    p = sub.add_parser("pipelines", help="list registered pipelines and availability")
    p.set_defaults(func=_cmd_pipelines)

    p = sub.add_parser("dashboard", help="serve the read-only dashboard")
    p.add_argument("experiment", nargs="?", default="experiments/pilot")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--upload", action="store_true",
                   help="also enable the image-upload page (adds one POST route)")
    p.set_defaults(func=_cmd_dashboard)

    p = sub.add_parser("weights", help="fetch or verify pinned checkpoints")
    p.add_argument("action", choices=["fetch", "verify"])
    p.add_argument("--only", default=None, help="comma-separated checkpoint names")
    p.set_defaults(func=_cmd_weights)

    p = sub.add_parser("train", help="train the NC-DCE curve estimator and its ablations")
    p.add_argument("model", choices=["nc-dce"])
    p.add_argument("--config", default="configs/train/nc_dce.yaml")
    p.add_argument("--variant", default=None, help="comma-separated variant names")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("enhance", help="enhance one image with NAG, update the paper, write reports")
    p.add_argument("image")
    p.add_argument("--kind", choices=["auto", "photo", "diagram"], default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-paper", action="store_true", help="do not touch the LaTeX paper")
    p.add_argument("--no-report", action="store_true", help="do not write reports to temp/")
    p.add_argument("--tex", default="paper/overleaf/main.tex")
    p.add_argument("--report-dir", default="temp")
    p.add_argument("--title", default=None, help="project title for the RPT template cover")
    p.add_argument("--student", default=None)
    p.add_argument("--uid", default=None)
    p.add_argument("--zip", action="store_true", help="rebuild paper/SensorSim_overleaf.zip")
    p.set_defaults(func=_cmd_enhance)

    p = sub.add_parser("paper-nag", help="generate the NAG tables/macros, update the paper, write the PDF")
    p.add_argument("--metrics", default="experiments/enhance_study/metrics.csv")
    p.add_argument("--training", default="experiments/enhance_study/training")
    p.add_argument("--tex", default="paper/overleaf/main.tex")
    p.add_argument("--report-dir", default="temp")
    p.add_argument("--no-tex", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--no-report", action="store_true")
    p.set_defaults(func=_cmd_paper_nag)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
