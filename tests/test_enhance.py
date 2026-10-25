"""The plumbing around NAG: classification, the LaTeX writer that paper-nag
uses, and enhancing an uploaded image.

The neural stages are replaced by stubs here. Their behaviour is covered in
``test_nag.py``. What these tests pin down is the plumbing:
- the paper is never edited twice in the same place, and is always backed up;
- enhancing an image writes its run folder and nothing else - no paper, no reports;
- the dashboard has a write route only when one was asked for.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sensorsim.data.synthetic import linear_to_srgb, natural_scene
from sensorsim.enhance.classify import classify_image
from sensorsim.enhance.latex import escape, update_paper_with_nag, upsert_block

MINI_TEX = r"""\documentclass{article}
\newcommand{\NRQ}{\texttt{nr\_quality}}
\begin{document}
\begin{abstract}
Abstract.
\end{abstract}
(iv)~an end-to-end reproducible pipeline in which every figure and number in this
paper is regenerated from one command.

\section{Results}
Text.
\section{Discussion}\label{sec:discussion}
More.
\section*{Acknowledgment}
\end{document}
"""


def _diagram() -> np.ndarray:
    img = np.full((240, 480, 3), 250, np.uint8)
    for i in range(4):
        cv2.rectangle(img, (20 + i * 115, 60), (110 + i * 115, 150), (220, 235, 250), -1)
        cv2.rectangle(img, (20 + i * 115, 60), (110 + i * 115, 150), (40, 40, 40), 2)
        cv2.putText(img, f"Box {i}", (30 + i * 115, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def _photo() -> np.ndarray:
    rgb = linear_to_srgb(natural_scene((256, 256), seed=3).linear_rgb)
    noisy = np.clip(rgb + np.random.default_rng(0).normal(0, 0.02, rgb.shape), 0, 1)
    return (noisy * 255).astype(np.uint8)


def test_classifier_separates_diagrams_from_photos():
    assert classify_image(_diagram())[0] == "diagram"
    assert classify_image(_photo())[0] == "photo"


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------

def test_escape_handles_latex_specials():
    assert escape("a_b & 50% #1") == r"a\_b \& 50\% \#1"


def test_upsert_inserts_once_then_replaces():
    once = upsert_block(MINI_TEX, "demo", "first", r"\section{Discussion}")
    twice = upsert_block(once, "demo", "second", r"\section{Discussion}")
    assert twice.count("%% AUTOGEN:demo BEGIN") == 1
    assert "second" in twice and "first" not in twice
    assert twice.index("second") < twice.index(r"\section{Discussion}")


def test_upsert_reports_a_missing_anchor():
    with pytest.raises(ValueError, match="anchor not found"):
        upsert_block(MINI_TEX, "demo", "x", "no such anchor")


def test_nag_paper_update_is_idempotent_and_backed_up(tmp_path):
    tex = tmp_path / "main.tex"
    tex.write_text(MINI_TEX, encoding="utf-8")
    (tmp_path / "refs.bib").write_text("@misc{x, title={x}}\n", encoding="utf-8")

    first = update_paper_with_nag(tex, rebuild_zip=False)
    assert Path(first["backup"]).read_text(encoding="utf-8") == MINI_TEX
    text = tex.read_text(encoding="utf-8")
    for key in ("nag-preamble", "nag-abstract", "nag-contribution", "nag-section", "nag-conclusion"):
        assert text.count(f"%% AUTOGEN:{key} BEGIN") == 1, key
    assert text.index(r"\section{Proposed Method") < text.index(r"\section{Discussion}")
    assert text.index("AUTOGEN:nag-abstract") < text.index(r"\end{abstract}")
    assert "he2013guided" in (tmp_path / "refs.bib").read_text(encoding="utf-8")

    second = update_paper_with_nag(tex, rebuild_zip=False)
    assert second["changed"] == "no"
    assert tex.read_text(encoding="utf-8") == text


def _record(tmp_path: Path, stem: str = "my_fig") -> dict:
    files = {}
    for key in ("original", "input_bicubic", "classical", "nag", "gate"):
        p = tmp_path / f"{key}.png"
        cv2.imwrite(str(p), _diagram())
        files[key] = p.as_posix()
    metric = {"sharpness": 1.0, "noise_sigma": 0.5, "colorfulness": 2.0, "nr_quality": 1.5,
              "consistency_psnr": 35.0, "consistency_ssim": 0.98}
    return {"name": f"{stem}.png", "stem": stem, "id": f"{stem}-abc", "kind": "diagram",
            "input_shape": [240, 480], "output_shape": [720, 1440], "scale": 3.0,
            "steps": [{"step": "classify", "detail": "diagram"}], "checkpoints": {"real_esrgan_general": "f" * 64},
            "metrics": {"input_bicubic": metric, "classical": metric, "nag": metric},
            "seconds": 1.0, "out_dir": tmp_path.as_posix(), "files": files}


# --------------------------------------------------------------------------
# Enhancing an image writes its run folder and nothing else
# --------------------------------------------------------------------------

def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_run_upload_touches_nothing_but_its_run_folder(tmp_path, monkeypatch):
    from sensorsim.enhance import pipeline

    work = tmp_path / "project"
    (work / "paper" / "overleaf").mkdir(parents=True)
    (work / "paper" / "overleaf" / "main.tex").write_text(MINI_TEX, encoding="utf-8")
    (work / "temp").mkdir()
    out_root = work / "experiments" / "enhance"
    monkeypatch.chdir(work)
    before = _snapshot(work)

    def fake_enhance(source, kind, device, out_root, original_name):
        folder = Path(out_root) / "my_fig-abc"
        folder.mkdir(parents=True)
        record = _record(folder)
        (folder / "metrics.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    monkeypatch.setattr(pipeline, "enhance_image", fake_enhance)
    outcome = pipeline.run_upload(work / "in.png", out_root=out_root)

    after = _snapshot(work)
    added = {k for k in after if k not in before}
    assert added and all(k.startswith("experiments/enhance/my_fig-abc/") for k in added), added
    assert all(after[k] == v for k, v in before.items()), "an existing file was changed"
    assert set(outcome["summary"]["outputs"]) == set(outcome["record"]["files"])


# --------------------------------------------------------------------------
# Dashboard upload route
# --------------------------------------------------------------------------

def _png_bytes() -> bytes:
    ok, buf = cv2.imencode(".png", _diagram())
    assert ok
    return buf.tobytes()


@pytest.fixture
def upload_client(tmp_path):
    from dashboard.app import create_app

    calls = []

    def fake_runner(path, **kwargs):
        calls.append((Path(path).name, kwargs))
        record = _record(tmp_path)
        return {"record": record, "summary": {"outputs": record["files"]}}

    app = create_app(tmp_path / "no-run", enable_upload=True,
                     upload_options={"out_root": tmp_path, "runner": fake_runner})
    app.config.update(TESTING=True)
    client = app.test_client()
    client.calls = calls
    return client


def test_upload_adds_exactly_one_write_route(tmp_path):
    from dashboard.app import create_app

    app = create_app(tmp_path, enable_upload=True)
    writes = [r.rule for r in app.url_map.iter_rules() if r.methods - {"GET", "HEAD", "OPTIONS"}]
    assert writes == ["/api/enhance"]
    assert all(not (r.methods - {"GET", "HEAD", "OPTIONS"}) for r in create_app(tmp_path).url_map.iter_rules())


def test_upload_rejects_non_images(upload_client):
    r = upload_client.post("/api/enhance", data={"file": (io.BytesIO(b"not an image"), "x.png")},
                           content_type="multipart/form-data")
    assert r.status_code == 400
    r = upload_client.post("/api/enhance", data={"file": (io.BytesIO(_png_bytes()), "x.exe")},
                           content_type="multipart/form-data")
    assert r.status_code == 400
    assert not upload_client.calls


def test_upload_rejects_oversized_files(upload_client):
    big = io.BytesIO(b"\0" * (21 * 1024 * 1024))
    r = upload_client.post("/api/enhance", data={"file": (big, "big.png")}, content_type="multipart/form-data")
    assert r.status_code == 413


def test_upload_happy_path(upload_client):
    r = upload_client.post("/api/enhance",
                           data={"file": (io.BytesIO(_png_bytes()), "My Fig.png"), "kind": "auto"},
                           content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    payload = r.get_json()
    assert payload["kind"] == "diagram"
    assert payload["urls"]["nag"].startswith("/enhanced/my_fig-abc/")
    name, kwargs = upload_client.calls[0]
    assert name == "My_Fig.png"
    assert upload_client.get("/upload").status_code == 200
    assert upload_client.get("/api/health").get_json()["upload"] is True


def test_upload_only_enhances(upload_client, tmp_path):
    """Whatever a client sends, the route asks for an enhancement and nothing more."""
    r = upload_client.post("/api/enhance",
                           data={"file": (io.BytesIO(_png_bytes()), "fig.png"), "kind": "photo",
                                 "update_paper": "true", "write_report": "true"},
                           content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    _, kwargs = upload_client.calls[0]
    assert set(kwargs) == {"kind", "original_name", "out_root"}
    assert kwargs["kind"] == "photo"
