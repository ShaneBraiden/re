"""The dashboard: its API contract, and the read-only guarantee.

The read-only guarantee is tested rather than asserted in a docstring, because
"the dashboard does not compute" is the kind of claim that stays true only for
as long as someone checks.
"""

from __future__ import annotations

import json

import pytest
import yaml

from dashboard.app import ExperimentStore, create_app
from sensorsim.experiment import run


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dash")
    root = tmp_path / "run"
    config = {
        "experiment_id": "dash",
        "seed": 3,
        "output": str(root),
        "corpus": {"kind": "synthetic", "path": str(tmp_path / "corpus"),
                   "kinds": ["natural"], "per_kind": 1, "size": [128, 128]},
        "degradation": {"mode": "compound", "levels": [0.0, 0.5, 1.0],
                        "schedule": {"alpha": {"from": 1.0, "to": 0.1, "scale": "log"}}},
        "pipelines": {"pipelines": [{"kind": "classical_isp"}, {"kind": "rl_deconv"},
                                    {"kind": "real_esrgan"}]},
        "metrics": {"families": ["fidelity"]},
        "validation": {"mode": "self"},
    }
    path = tmp_path / "dash.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run(path, progress=None)
    return root


@pytest.fixture
def client(finished_run):
    app = create_app(finished_run)
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_page_and_assets_are_served(client):
    assert client.get("/").status_code == 200
    for asset in ("/static/app.js", "/static/styles.css"):
        response = client.get(asset)
        assert response.status_code == 200
        assert response.data


def test_health_reports_ready(client, finished_run):
    payload = client.get("/api/health").get_json()
    assert payload["ready"]
    assert "dash" in payload["experiment"]


def test_health_reports_not_ready_without_a_run(tmp_path):
    app = create_app(tmp_path / "nothing-here")
    response = app.test_client().get("/api/health")
    assert response.status_code == 503
    assert "sensorsim run" in response.get_json()["note"]


def test_index_api_omits_the_values_blob(client):
    payload = client.get("/api/index").get_json()
    assert "values" not in payload, "the index response must stay small"
    assert payload["has_values"]
    assert payload["experiment_id"] == "dash"
    assert [c["severity"] for c in payload["conditions"]] == [0.0, 0.5, 1.0]
    assert payload["directions"]["psnr"] is True


def test_unavailable_pipelines_are_advertised_with_a_reason(client):
    payload = client.get("/api/index").get_json()
    by_name = {p["name"]: p for p in payload["pipelines"]}
    assert by_name["classical_isp"]["available"]
    assert not by_name["real_esrgan"]["available"]
    assert by_name["real_esrgan"]["reason"]


def test_values_endpoint_returns_cached_numbers(client):
    index = client.get("/api/index").get_json()
    condition = index["conditions"][-1]["hash"]
    payload = client.get(
        f"/api/values?image=nat_00&condition={condition}&pipelines=classical_isp,rl_deconv"
    ).get_json()
    assert set(payload["values"]) == {"classical_isp", "rl_deconv"}
    assert payload["values"]["classical_isp"]["psnr"] > 0


def test_values_endpoint_validates_its_arguments(client):
    assert client.get("/api/values?image=nat_00").status_code == 400
    assert client.get("/api/series?image=nat_00").status_code == 400


def test_series_endpoint_spans_the_whole_sweep(client):
    payload = client.get("/api/series?image=nat_00&metric=psnr&pipelines=classical_isp").get_json()
    points = payload["series"]["classical_isp"]
    assert [p["severity"] for p in points] == [0.0, 0.5, 1.0]
    assert all(p["value"] is not None for p in points)
    # Fidelity must fall as the hardware gets worse - if not, the run is broken.
    assert points[0]["value"] > points[-1]["value"]


def test_series_for_an_unavailable_pipeline_is_all_nulls(client):
    payload = client.get("/api/series?image=nat_00&metric=psnr&pipelines=real_esrgan").get_json()
    assert all(p["value"] is None for p in payload["series"]["real_esrgan"])


def test_previews_are_served(client):
    assert client.get("/previews/nat_00/reference.jpg").status_code == 200
    assert client.get("/previews/nat_00/does-not-exist.jpg").status_code == 404


def test_preview_paths_cannot_escape_the_run_directory(client):
    """send_from_directory should refuse traversal; pin it so a refactor cannot
    turn the preview route into an arbitrary file reader."""
    for attempt in ("../manifest.json", "..%2Fmanifest.json", "....//manifest.json"):
        assert client.get(f"/previews/{attempt}").status_code in (400, 403, 404)


def test_every_route_is_read_only(client):
    """No POST, PUT, PATCH or DELETE anywhere. The dashboard demonstrates
    results; it never produces them."""
    app = create_app("experiments/pilot")
    for rule in app.url_map.iter_rules():
        assert rule.methods <= {"GET", "HEAD", "OPTIONS"}, f"{rule} accepts writes"


def test_dashboard_module_does_not_import_the_stages():
    """A dashboard that can import the degrader is a dashboard that will
    eventually compute something on demand."""
    import inspect

    import dashboard.app as module

    source = inspect.getsource(module)
    for stage in ("sensorsim.degrader", "sensorsim.restore", "sensorsim.evaluate"):
        assert stage not in source, f"the dashboard reaches into {stage}"


def test_store_reloads_when_the_index_changes(finished_run):
    store = ExperimentStore(finished_run)
    first = store.index["experiment_id"]

    data = json.loads(store.index_path.read_text(encoding="utf-8"))
    data["experiment_id"] = "renamed"
    store.index_path.write_text(json.dumps(data), encoding="utf-8")

    assert first == "dash"
    assert store.index["experiment_id"] == "renamed"

    data["experiment_id"] = "dash"
    store.index_path.write_text(json.dumps(data), encoding="utf-8")


def test_store_explains_a_missing_run(tmp_path):
    store = ExperimentStore(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="sensorsim run"):
        _ = store.index


# --------------------------------------------------------------------------
# The frontend's read-only endpoints
# --------------------------------------------------------------------------

def test_frontend_modules_are_served_as_javascript(client):
    """Browsers refuse to run an ES module served as text/plain, which is what
    the Windows registry maps .js to on some machines."""
    for asset in ("/static/app.js", "/static/js/ui.js", "/static/js/pages/explorer.js"):
        response = client.get(asset)
        assert response.status_code == 200
        assert "javascript" in response.mimetype


def test_experiments_lists_the_default_run(client):
    payload = client.get("/api/experiments").get_json()
    assert payload["default"] == "run"
    run = next(r for r in payload["experiments"] if r["id"] == "run")
    assert run["default"] and run["experiment_id"] == "dash"
    assert run["images"] == 1 and run["conditions"] == 3
    assert run["pipelines"] == 3 and run["available"] == 2


def test_sibling_runs_are_selectable_with_exp(finished_run, tmp_path):
    import shutil

    shutil.copytree(finished_run, tmp_path / "a")
    (tmp_path / "b").mkdir()
    index = json.loads((finished_run / "dashboard.json").read_text(encoding="utf-8"))
    index["experiment_id"] = "second"
    (tmp_path / "b" / "dashboard.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "not-a-run").mkdir()

    client = create_app(tmp_path / "a").test_client()
    ids = [r["id"] for r in client.get("/api/experiments").get_json()["experiments"]]
    assert sorted(ids) == ["a", "b"]
    assert client.get("/api/index?exp=b").get_json()["experiment_id"] == "second"
    assert client.get("/api/index").get_json()["experiment_id"] == "dash"


def test_an_unknown_run_is_a_json_404(client):
    response = client.get("/api/index?exp=../../etc")
    assert response.status_code == 404
    assert "error" in response.get_json()
    assert client.get("/previews/nat_00/reference.jpg?exp=nope").status_code == 404


def test_aggregate_is_the_mean_of_the_cached_values(client):
    index = client.get("/api/index").get_json()
    payload = client.get("/api/aggregate?metric=psnr&pipelines=classical_isp,real_esrgan").get_json()
    assert payload["n_images"] == 1
    points = payload["series"]["classical_isp"]
    assert [p["severity"] for p in points] == [0.0, 0.5, 1.0]
    # One scene: the mean is exactly the cached value, and n says one scene.
    single = client.get("/api/series?image=nat_00&metric=psnr&pipelines=classical_isp").get_json()
    assert [p["mean"] for p in points] == [p["value"] for p in single["series"]["classical_isp"]]
    assert all(p["n"] == 1 and p["min"] == p["max"] == p["mean"] for p in points)
    # A pipeline that never ran has no mean, and n = 0 says why.
    assert all(p["mean"] is None and p["n"] == 0 for p in payload["series"]["real_esrgan"])
    assert index["conditions"][0]["hash"] == points[0]["hash"]
    assert client.get("/api/aggregate?metric=psnr").status_code == 400


def test_run_endpoint_reports_config_validation_and_gaps(client):
    payload = client.get("/api/run").get_json()
    assert payload["manifest"]["experiment_id"] == "dash"
    assert payload["config"]["degradation"]["levels"] == [0.0, 0.5, 1.0]
    assert "passed" in payload["validation"]
    gaps = {g["pipeline"]: g for g in payload["gaps"]}
    assert gaps["real_esrgan"]["rows"] == 3 and gaps["real_esrgan"]["reason"]


def test_analysis_is_empty_until_the_run_is_analysed(finished_run, tmp_path):
    client = create_app(finished_run, figures=tmp_path / "figures").test_client()
    payload = client.get("/api/analysis").get_json()
    assert payload["figures_dir"] is None and payload["tables"] == {}


def test_analysis_reads_the_runs_own_figures_folder(finished_run, tmp_path):
    folder = tmp_path / "figures" / "run"
    folder.mkdir(parents=True)
    (folder / "knees.csv").write_text(
        "pipeline,metric_name,knee,identifiable,notes\nrl_deconv,psnr,0.5,False,\nclassical_isp,psnr,nan,True,x\n",
        encoding="utf-8")
    (folder / "knee_psnr_rl_deconv.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    client = create_app(finished_run, figures=tmp_path / "figures").test_client()

    response = client.get("/api/analysis")
    assert response.status_code == 200
    payload = json.loads(response.data)  # strict: NaN would not parse in a browser
    knees = payload["tables"]["knees"]
    assert knees[0] == {"pipeline": "rl_deconv", "metric_name": "psnr", "knee": 0.5,
                        "identifiable": False, "notes": None}
    assert knees[1]["knee"] is None
    assert payload["figures"] == [{"name": "knee_psnr_rl_deconv", "url": "/figures/run/knee_psnr_rl_deconv.png"}]
    assert client.get("/figures/run/knee_psnr_rl_deconv.png").status_code == 200


def test_shared_figures_folder_is_claimed_only_by_the_matching_run(finished_run, tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    table = "pipeline,metric_name,severity,n,mean,median,std,ci_low,ci_high\n"
    (figures / "recovery_aggregate.csv").write_text(
        table + "classical_isp,psnr,1.0,1,0,0,0,0,0\nnlm_denoise,psnr,1.0,1,0,0,0,0,0\n", encoding="utf-8")
    client = create_app(finished_run, figures=figures).test_client()
    assert client.get("/api/analysis").get_json()["figures_dir"] is None

    (figures / "recovery_aggregate.csv").write_text(
        table + "classical_isp,psnr,1.0,1,0,0,0,0,0\nrl_deconv,psnr,1.0,1,0,0,0,0,0\n", encoding="utf-8")
    assert client.get("/api/analysis").get_json()["figures_dir"] is not None


def test_past_enhancements_are_listed_read_only(finished_run, tmp_path):
    folder = tmp_path / "enhance" / "photo-abc"
    folder.mkdir(parents=True)
    (folder / "nag.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    record = {"name": "photo.jpg", "kind": "photo", "scale": 2.0, "seconds": 1.5,
              "metrics": {"nag": {"sharpness": float("nan")}},
              "files": {"nag": str(folder / "nag.png"), "gate": str(folder / "gate.png")}}
    (folder / "metrics.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "enhance" / "half-written").mkdir()

    app = create_app(finished_run, upload_options={"out_root": tmp_path / "enhance"})
    client = app.test_client()
    response = client.get("/api/enhancements")
    payload = json.loads(response.data)
    assert payload["upload"] is False
    [run] = payload["runs"]
    assert run["id"] == "photo-abc" and run["name"] == "photo.jpg"
    assert run["metrics"]["nag"]["sharpness"] is None
    # Only files that exist get a URL.
    assert run["urls"] == {"nag": "/enhanced/photo-abc/nag.png"}
    assert client.get("/enhanced/photo-abc/nag.png").status_code == 200
    assert client.get("/enhanced/../photo-abc/metrics.json").status_code in (400, 403, 404)


# --------------------------------------------------------------------------
# Which frontend is served
# --------------------------------------------------------------------------

def test_classic_frontend_stands_in_until_the_react_app_is_built(finished_run, tmp_path):
    """A checkout with no Node toolchain still gets a working dashboard."""
    client = create_app(finished_run, web=tmp_path / "unbuilt").test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b"/static/app.js" in page.data
    assert client.get("/classic").status_code == 200


def test_built_react_app_is_served_at_the_root(finished_run, tmp_path):
    web = tmp_path / "web"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text(
        '<div id="root"></div><script type="module" src="/app/assets/index-abc.js"></script>', encoding="utf-8")
    (web / "assets" / "index-abc.js").write_text("export {};", encoding="utf-8")
    client = create_app(finished_run, web=web).test_client()

    page = client.get("/")
    assert b'id="root"' in page.data
    assert page.headers["Cache-Control"] == "no-cache"

    asset = client.get("/app/assets/index-abc.js")
    assert asset.status_code == 200
    assert "javascript" in asset.mimetype
    assert "max-age=31536000" in asset.headers["Cache-Control"]

    # The classic frontend stays one link away.
    assert b"/static/app.js" in client.get("/classic").data


def test_built_app_paths_cannot_escape_the_build_directory(finished_run, tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<div id=root></div>", encoding="utf-8")
    client = create_app(finished_run, web=web).test_client()
    for attempt in ("../dash.yaml", "..%2Fdash.yaml"):
        assert client.get(f"/app/{attempt}").status_code in (400, 403, 404)
