"""Config, sweep construction, the run orchestrator and reproducibility."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from sensorsim.config import build_sweep, load_experiment, pipeline_specs
from sensorsim.degrader.validate import ValidationError
from sensorsim.experiment import run, run_validation_gate
from sensorsim.io.manifest import RunManifest, content_hash, stable_json


# ---------------------------------------------------------------- config ---

def test_compound_sweep_includes_the_identity_condition():
    config = {
        "mode": "compound",
        "levels": [0.5, 1.0],  # deliberately missing 0.0
        "schedule": {"alpha": {"from": 1.0, "to": 0.1, "scale": "log"}},
    }
    sweep = build_sweep(config)
    assert sweep[0].severity == 0.0
    assert sweep[0].alpha == pytest.approx(1.0)
    assert sweep[0].tier == "clean"


def test_compound_sweep_interpolates_each_scale_correctly():
    config = {
        "mode": "compound",
        "levels": [0.0, 0.5, 1.0],
        "schedule": {
            "alpha": {"from": 1.0, "to": 0.01, "scale": "log"},
            "sigma_r_dn": {"from": 0.0, "to": 10.0, "scale": "linear"},
            "binning": {"from": 1, "to": 2, "scale": "step"},
        },
    }
    sweep = build_sweep(config)
    mid = sweep[1]
    assert mid.alpha == pytest.approx(0.1)          # geometric midpoint of 1.0 and 0.01
    assert mid.sigma_r_dn == pytest.approx(5.0)     # arithmetic midpoint
    assert mid.binning in (1, 2)                    # integer, so it steps
    assert sweep[-1].alpha == pytest.approx(0.01)
    assert sweep[-1].binning == 2


def test_log_scale_rejects_a_zero_endpoint():
    config = {"mode": "compound", "levels": [0.0, 1.0],
              "schedule": {"alpha": {"from": 1.0, "to": 0.0, "scale": "log"}}}
    with pytest.raises(ValueError, match="positive endpoints"):
        build_sweep(config)


def test_axis_sweep_prepends_a_clean_condition():
    config = {
        "mode": "axis", "axis": "alpha", "values": [0.5, 0.1],
        "fixed": {"sigma_r_dn": 4.0, "binning": 1, "bit_depth": 14, "r_psf": 0.0},
    }
    sweep = build_sweep(config)
    assert sweep[0].severity == 0.0
    assert len({p.params_hash for p in sweep}) == len(sweep)


def test_unknown_sweep_mode_is_an_error():
    with pytest.raises(ValueError, match="unknown sweep mode"):
        build_sweep({"mode": "vibes"})


def test_pipeline_specs_require_a_kind():
    with pytest.raises(ValueError, match="kind"):
        pipeline_specs({"pipelines": [{"name": "nameless"}]})


def test_shipped_configs_load_and_build(tmp_path):
    config = load_experiment("configs/experiments/pilot.yaml")
    sweep = build_sweep(config["degradation"])
    specs = pipeline_specs(config["pipelines"])
    assert sweep[0].severity == 0.0
    assert sweep[-1].severity == 1.0
    assert len(specs) == 4
    assert {s["kind"] for s in specs} == {"classical_isp", "nlm_denoise", "curve_tone", "rl_deconv"}


def test_full_config_references_the_whole_bank():
    config = load_experiment("configs/experiments/full.yaml")
    kinds = {s["kind"] for s in pipeline_specs(config["pipelines"])}
    assert {"nafnet", "restormer", "real_esrgan", "zero_dce", "learned_isp"} <= kinds


# -------------------------------------------------------------- manifest ---

def test_content_hash_is_order_independent():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_stable_json_normalises_float_noise():
    assert stable_json({"x": 0.1 + 0.2}) == stable_json({"x": 0.3})


def test_manifest_round_trips(tmp_path):
    manifest = RunManifest.create("probe", {"seed": 1}, seed=1)
    manifest.record_stage("degrade", frames=12)
    manifest.finish()
    path = manifest.write(tmp_path / "manifest.json")

    loaded = RunManifest.load(path)
    assert loaded.experiment_id == "probe"
    assert loaded.config_hash == manifest.config_hash
    assert loaded.stages["degrade"]["frames"] == 12
    assert loaded.packages["python"]


def test_manifest_rejects_a_foreign_version(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"experiment_id": "x", "config": {}, "seed": 0, "manifest_version": 99}))
    with pytest.raises(ValueError, match="version"):
        RunManifest.load(path)


def test_manifest_flags_unpinned_weights():
    manifest = RunManifest.create("probe", {}, seed=0)
    manifest.weights = {"nafnet": "unpinned"}
    assert any("nafnet" in problem for problem in manifest.reproducibility_report())


# ------------------------------------------------------- validation gate ---

def test_validation_gate_self_mode_passes(tmp_path):
    gate = run_validation_gate({"validation": {"mode": "self"}, "seed": 0}, tmp_path)
    assert gate["passed"]
    assert "NOT been compared against a real sensor" in gate["note"]
    assert (tmp_path / "validation.json").exists()


def test_validation_gate_skip_is_loudly_recorded(tmp_path):
    gate = run_validation_gate({"validation": {"mode": "skip"}}, tmp_path)
    assert gate["passed"] is None
    assert "SKIPPED" in gate["note"]


def test_hardware_gate_refuses_without_captures(tmp_path):
    config = {"validation": {"mode": "hardware", "calibration_dir": str(tmp_path / "nope")}}
    with pytest.raises(ValidationError, match="stacks of"):
        run_validation_gate(config, tmp_path)


def test_unknown_validation_mode_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="unknown validation mode"):
        run_validation_gate({"validation": {"mode": "hope"}}, tmp_path)


# ------------------------------------------------------------ end to end ---

@pytest.fixture
def tiny_experiment(tmp_path):
    """A two-condition, one-scene, two-pipeline run - complete but quick."""
    root = tmp_path / "run"
    config = {
        "experiment_id": "tiny",
        "seed": 7,
        "output": str(root),
        "corpus": {"kind": "synthetic", "path": str(tmp_path / "corpus"),
                   "kinds": ["natural"], "per_kind": 1, "size": [128, 128]},
        "degradation": {"mode": "compound", "levels": [0.0, 1.0],
                        "schedule": {"alpha": {"from": 1.0, "to": 0.1, "scale": "log"},
                                     "sigma_r_dn": {"from": 0.0, "to": 8.0, "scale": "linear"}}},
        "pipelines": {"pipelines": [{"kind": "classical_isp"}, {"kind": "nafnet"}]},
        "metrics": {"families": ["fidelity", "perceptual"]},
        "validation": {"mode": "self"},
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, root


def test_run_produces_every_artefact(tiny_experiment):
    import pandas as pd

    path, root = tiny_experiment
    summary = run(path, progress=None)

    for name in ("metrics.csv", "manifest.json", "dashboard.json", "degradations.jsonl", "validation.json"):
        assert (root / name).exists(), f"{name} missing"

    table = pd.read_csv(root / "metrics.csv")
    assert summary["rows"] == len(table)
    assert set(table["pipeline"]) == {"classical_isp", "nafnet"}

    # The unavailable arm is present as explicit gap rows, never absent.
    gaps = table[table["pipeline"] == "nafnet"]
    assert len(gaps) > 0
    assert (gaps["status"] == "unavailable").all()
    # The reason names what is missing: torch where it is not installed, the
    # checkpoint everywhere (this spec gives no weights).
    assert gaps["note"].str.contains("weights").all()

    ok = table[table["status"] == "ok"]
    assert set(ok["pipeline"]) == {"classical_isp"}
    assert set(ok["metric_family"]) == {"fidelity", "perceptual"}


def test_run_is_resumable(tiny_experiment):
    import pandas as pd

    path, root = tiny_experiment
    first = run(path, progress=None)
    before = pd.read_csv(root / "metrics.csv")

    second = run(path, resume=True, progress=None)
    after = pd.read_csv(root / "metrics.csv")

    assert second["rows"] == first["rows"]
    assert len(after) == len(before)
    pd.testing.assert_frame_equal(
        before.sort_values(list(before.columns)).reset_index(drop=True),
        after.sort_values(list(after.columns)).reset_index(drop=True),
    )


def test_dashboard_index_describes_the_run(tiny_experiment):
    path, root = tiny_experiment
    run(path, progress=None)
    index = json.loads((root / "dashboard.json").read_text(encoding="utf-8"))

    assert index["experiment_id"] == "tiny"
    assert [c["severity"] for c in index["conditions"]] == [0.0, 1.0]
    assert {p["name"]: p["available"] for p in index["pipelines"]} == {
        "classical_isp": True, "nafnet": False,
    }
    assert index["directions"]["psnr"] is True
    assert index["families"]["psnr"] == "fidelity"
    assert index["gaps"], "an unavailable pipeline must appear in the gap list"


def test_previews_exist_for_every_condition(tiny_experiment):
    path, root = tiny_experiment
    run(path, progress=None)
    previews = list((root / "previews").rglob("*.jpg"))
    names = {p.name for p in previews}
    assert any(n == "reference.jpg" for n in names)
    assert sum("__degraded" in n for n in names) == 2
    assert sum("__classical_isp" in n for n in names) == 2


def test_reruns_reproduce_the_same_numbers(tiny_experiment, tmp_path):
    """Same config, same seed, same numbers - section 7's whole point."""
    import pandas as pd

    path, root = tiny_experiment
    run(path, progress=None)
    first = pd.read_csv(root / "metrics.csv")

    for stale in ("metrics.csv",):
        (root / stale).unlink()
    import shutil

    shutil.rmtree(root / "cache", ignore_errors=True)

    run(path, resume=False, progress=None)
    second = pd.read_csv(root / "metrics.csv")

    key = ["image_id", "degradation_params_hash", "pipeline", "metric_name"]
    merged = first.merge(second, on=key, suffixes=("_a", "_b"))
    assert len(merged) == len(first)
    finite = merged[np.isfinite(merged["value_a"]) & np.isfinite(merged["value_b"])]
    assert np.allclose(finite["value_a"], finite["value_b"])
