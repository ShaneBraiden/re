"""Opt-in image upload: the dashboard's one write route.

The dashboard proper is a read-only window onto finished runs, and
``tests/test_dashboard.py`` keeps it that way. Uploading and enhancing an image
is computation, so it lives here, in a blueprint that ``create_app`` registers
only when asked (``sensorsim dashboard --upload``). With the flag off, the app
has exactly the routes it always had.

Routes::

    GET  /upload               the frontend, opened on its Enhance page
    POST /api/enhance          multipart "file" (+ optional kind: auto, photo or diagram)

An upload is enhanced and that is all: the run's own folder under
``experiments/enhance/`` is the only thing written. The paper and the reports
are never touched from here. Those files are served by the read-only app at
``/enhanced/<path>`` (it lists past runs too), so this blueprint adds
computation and nothing else.

Uploads are checked by extension *and* decoded with Pillow before anything
runs, capped by ``MAX_CONTENT_LENGTH``, and processed one at a time - a laptop
GPU running two Real-ESRGAN jobs at once runs out of memory, not faster.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from dashboard.app import finite, frontend_page

ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_BYTES = 20 * 1024 * 1024


def _default_runner(path: Path, **kwargs: Any) -> dict[str, Any]:
    from sensorsim.enhance.pipeline import run_upload

    return run_upload(path, **kwargs)


def create_upload_blueprint(
    out_root: str | Path = "experiments/enhance",
    runner: Callable[..., dict[str, Any]] | None = None,
) -> Blueprint:
    bp = Blueprint("upload", __name__)
    root = Path(out_root).resolve()
    lock = threading.Lock()
    run = runner or _default_runner

    @bp.get("/upload")
    def upload_page():
        # The same page as "/": it reads the path and opens on the Enhance page.
        return frontend_page()

    @bp.post("/api/enhance")
    def api_enhance():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify({"error": "no file in the request (field name: file)"}), 400
        name = secure_filename(file.filename) or "upload.png"
        if Path(name).suffix.lower() not in ALLOWED:
            return jsonify({"error": f"unsupported file type; allowed: {', '.join(sorted(ALLOWED))}"}), 400

        kind = request.form.get("kind", "auto")
        if kind not in ("auto", "photo", "diagram"):
            return jsonify({"error": "kind must be auto, photo or diagram"}), 400

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            file.save(path)
            try:
                from PIL import Image

                with Image.open(path) as im:
                    im.verify()
            except Exception:  # noqa: BLE001 - any decode failure is the client's problem
                return jsonify({"error": "the file is not a readable image"}), 400

            if not lock.acquire(timeout=1):
                return jsonify({"error": "another image is being enhanced; try again shortly"}), 429
            try:
                outcome = run(path, kind=kind, original_name=file.filename, out_root=root)
            except FileNotFoundError as exc:
                return jsonify({"error": str(exc)}), 503
            except Exception as exc:  # noqa: BLE001
                current_app.logger.exception("enhancement failed")
                return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
            finally:
                lock.release()

        record = outcome["record"]
        urls = {key: f"/enhanced/{record['id']}/{Path(p).name}" for key, p in record["files"].items()}
        return jsonify(finite({
            "id": record["id"],
            "name": record.get("name", file.filename),
            "kind": record["kind"],
            "scale": record["scale"],
            "input_shape": record["input_shape"],
            "output_shape": record["output_shape"],
            "steps": record["steps"],
            "metrics": record["metrics"],
            "seconds": record["seconds"],
            "features": record.get("features"),
            "noise_level_function": record.get("noise_level_function"),
            "urls": urls,
        }))

    return bp
