"""The upload flow around NAG: classification, the LaTeX writer, the reports and
the opt-in dashboard route.

The neural stages are replaced by stubs here. Their behaviour is covered in
``test_nag.py``. What these tests pin down is the plumbing:
- the paper is never edited twice in the same place, and is always backed up;
- the RPT template is never written to;
- the dashboard has a write route only when one was asked for.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sensorsim.data.synthetic import linear_to_srgb, natural_scene
from sensorsim.enhance.classify import classify_image
from sensorsim.enhance.latex import escape, insert_upload_figure, update_paper_with_nag, upsert_block

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
    for key in ("original", "input_bicubic", "classical", "nag", "gate", "comparison"):
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


def test_upload_figure_lands_after_the_nag_section_and_escapes(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    tex = paper / "main.tex"
    tex.write_text(MINI_TEX, encoding="utf-8")
    update_paper_with_nag(tex, rebuild_zip=False)
    record = _record(tmp_path)

    insert_upload_figure(record, tex=tex)
    insert_upload_figure(record, tex=tex)
    text = tex.read_text(encoding="utf-8")
    assert text.count("%% AUTOGEN:upload-my_fig BEGIN") == 1
    assert text.index("AUTOGEN:nag-section END") < text.index("AUTOGEN:upload-my_fig BEGIN")
    assert (paper / "figures" / "upload_my_fig_comparison.png").exists()
    assert r"\label{fig:upload-my-fig}" in text


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

TEMPLATE = Path("temp/RPT_FinalReport_Template.docx")


@pytest.mark.skipif(not TEMPLATE.exists(), reason="RPT template not present")
def test_rpt_template_is_filled_as_a_copy(tmp_path):
    docx = pytest.importorskip("docx")
    from sensorsim.enhance.report import fill_rpt_template

    before = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    out = fill_rpt_template(_record(tmp_path), "\\begin{figure}x\\end{figure}", out_dir=tmp_path,
                            cover={"title": "Noise-aware enhancement"})
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == before

    doc = docx.Document(out)
    texts = [p.text for p in doc.paragraphs]
    i = next(k for k, t in enumerate(texts) if t.startswith("Output:") and "NAG pipeline" in t)
    assert doc.paragraphs[i + 1]._p.xpath(".//pic:pic"), "the enhanced image follows the Experiment 3 output"
    assert "NOISE-AWARE ENHANCEMENT" in texts
    assert any(t.startswith("Output: Insert your PubMed") for t in texts), "other experiments untouched"


def test_upload_report_writes_html_and_markdown(tmp_path):
    from sensorsim.enhance.report import write_upload_report

    out = write_upload_report(_record(tmp_path), "\\begin{figure}x\\end{figure}", out_dir=tmp_path)
    html = Path(out["report_html"]).read_text(encoding="utf-8")
    assert "def wiener_gate" in html, "code excerpts come from the source"
    assert "\\begin{figure}" in html
    assert Path(out["report_md"]).exists()


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
        return {"record": record, "summary": {"outputs": {"report_pdf": "temp/x.pdf"}}}

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
                           data={"file": (io.BytesIO(_png_bytes()), "My Fig.png"), "kind": "auto",
                                 "update_paper": "false"},
                           content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    payload = r.get_json()
    assert payload["kind"] == "diagram"
    assert payload["urls"]["nag"].startswith("/enhanced/my_fig-abc/")
    name, kwargs = upload_client.calls[0]
    assert name == "My_Fig.png" and kwargs["update_paper"] is False
    assert upload_client.get("/upload").status_code == 200
    assert upload_client.get("/api/health").get_json()["upload"] is True
