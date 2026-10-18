"""Enhance one uploaded image with NAG, and record everything that was done.

An upload has no capture record, so the noise map is the *blind* estimate from
:func:`sensorsim.enhance.noise.blind_sigma_map`. Everything downstream is the
same method the restoration-bank arms run:

photo    NC-DCE (only if the frame is dark) -> VP-denoise -> G-ESRGAN (x4plus)
diagram  G-ESRGAN (general-x4v3, kinder to text) -> white-point snap -> mild unsharp

A classical baseline runs alongside for comparison: bicubic upscale, non-local
means at the estimated noise level, unsharp mask. Plain bicubic is the
reference "input" panel.

Everything lands in ``experiments/enhance/<stem>-<hash>/``. The run is
content-addressed: the same file always lands in the same folder.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sensorsim.io.manifest import content_hash, stable_json

OUT_ROOT = Path("experiments/enhance")
MAX_INPUT_SIDE = 1536
MAX_OUTPUT_SIDE = 4096
DIAGRAM_TARGET_WIDTH = 2150   # 7.16 in IEEE text width at 300 dpi
DARK_LUMA = 0.35


def safe_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem[:40] or "upload"


def load_image(source: str | Path | bytes) -> np.ndarray:
    """Decode to uint8 RGB, honouring EXIF orientation, alpha over white."""
    from PIL import Image, ImageOps

    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            background = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(background, im)
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def _luma(img: np.ndarray) -> np.ndarray:
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def _u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _unsharp(x: np.ndarray, sigma: float = 1.0, amount: float = 0.5) -> np.ndarray:
    return np.clip(x + amount * (x - cv2.GaussianBlur(x, (0, 0), sigma)), 0.0, 1.0)


def target_scale(shape: tuple[int, int], kind: str) -> float:
    h, w = shape
    if kind == "diagram":
        s = max(2.0, float(np.ceil(DIAGRAM_TARGET_WIDTH / max(w, 1))))
    else:
        s = 4.0 if max(h, w) <= 1024 else 2.0
    return float(max(1.0, min(4.0, s, MAX_OUTPUT_SIDE / max(h, w))))


def snap_background(x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Map a light background to pure white: per-channel white point at the mode."""
    q = (_u8(x) >> 3).reshape(-1, 3).astype(np.int32)
    codes = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    mode = int(np.bincount(codes).argmax())
    bg = np.array([(mode >> 10) & 31, (mode >> 5) & 31, mode & 31], np.float32) * 8 + 4
    bg = bg / 255.0
    if float(_luma(bg[None, None])[0, 0]) < 0.7:
        return x, bg.tolist()
    out = np.clip(x / np.maximum(bg, 0.5), 0.0, 1.0)
    out[np.all(out > 0.97, axis=-1)] = 1.0
    return out, bg.tolist()


def tone_matched_consistency(output: np.ndarray, original: np.ndarray) -> dict[str, float]:
    """PSNR / SSIM of the downscaled output against the upload, after a
    per-channel affine tone match.

    The match removes global brightness and contrast changes (which NC-DCE makes
    on purpose), so what is left measures *structural* agreement with what was
    actually in the file: invented detail lowers it, faithful sharpening barely
    moves it.
    """
    from sensorsim.evaluate.fidelity import psnr, ssim

    h, w = original.shape[:2]
    down = cv2.resize(output, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
    ref = original.astype(np.float64)
    matched = np.empty_like(down)
    for c in range(3):
        a, b = np.polyfit(down[..., c].ravel(), ref[..., c].ravel(), 1)
        matched[..., c] = a * down[..., c] + b
    matched_u8 = np.clip(matched, 0, 255).round().astype(np.uint8)
    return {"consistency_psnr": psnr(matched_u8, original), "consistency_ssim": ssim(matched_u8, original)}


def _metrics(image_u8: np.ndarray, original: np.ndarray) -> dict[str, float]:
    from sensorsim.evaluate.perceptual import compute as perceptual

    out = {k: float(v) for k, v in perceptual(image_u8, include_optional=False).items()}
    out.update(tone_matched_consistency(image_u8, original))
    return out


def _detail_crop(image: np.ndarray, size: int) -> tuple[int, int, int, int]:
    """The crop with the most edge energy - where enhancement differences show."""
    g = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F) ** 2
    k = max(8, size)
    energy = cv2.blur(g, (k, k))
    h, w = energy.shape
    half = k // 2
    inner = energy[half:max(half + 1, h - half), half:max(half + 1, w - half)]
    y, x = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return int(x), int(y), min(k, w), min(k, h)


def comparison_figure(panels: list[tuple[str, np.ndarray]], path: Path, crop_frac: float = 0.3) -> Path:
    """Full frame of the result plus matched zoomed crops of every panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sensorsim.analyze.figures import THEME, apply_style

    apply_style()
    ref = panels[-1][1]
    h, w = ref.shape[:2]
    size = int(min(h, w) * crop_frac)
    x, y, cw, ch = _detail_crop(panels[0][1], size)

    fig = plt.figure(figsize=(7.16, 4.2), dpi=200)
    grid = fig.add_gridspec(2, len(panels), height_ratios=[1.25, 1.0])
    top = fig.add_subplot(grid[0, :])
    top.imshow(ref)
    top.add_patch(plt.Rectangle((x, y), cw, ch, fill=False, ec=THEME.ink, lw=1.2))
    top.set_title(f"{panels[-1][0]}: full frame, box marks the zoomed region", fontsize=8)
    top.axis("off")
    for i, (name, img) in enumerate(panels):
        ax = fig.add_subplot(grid[1, i])
        ax.imshow(img[y:y + ch, x:x + cw], interpolation="nearest")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=THEME.surface)
    plt.close(fig)
    return path


def enhance_image(
    source: str | Path,
    kind: str = "auto",
    device: str = "auto",
    out_root: str | Path = OUT_ROOT,
    original_name: str | None = None,
) -> dict[str, Any]:
    """Run NAG on one image. Returns a JSON-able record of the run."""
    from sensorsim.enhance import weights
    from sensorsim.enhance.classify import classify_image
    from sensorsim.enhance.guard import guarded_sr
    from sensorsim.enhance.noise import blind_sigma_map, luma_sigma
    from sensorsim.enhance.sr import SuperResolver
    from sensorsim.enhance.vpdenoise import guided_filter_varying

    started = time.perf_counter()
    raw_bytes = Path(source).read_bytes()
    name = original_name or Path(source).name
    stem = safe_stem(name)
    digest = content_hash({"bytes": hashlib.sha256(raw_bytes).hexdigest(), "kind": kind}, 8)
    out_dir = Path(out_root) / f"{stem}-{digest}"
    out_dir.mkdir(parents=True, exist_ok=True)

    original = load_image(raw_bytes)
    steps: list[dict[str, str]] = [{"step": "decode", "detail": f"{name}: {original.shape[1]}x{original.shape[0]} px"}]
    h0, w0 = original.shape[:2]
    if max(h0, w0) > MAX_INPUT_SIDE:
        f = MAX_INPUT_SIDE / max(h0, w0)
        original = cv2.resize(original, (round(w0 * f), round(h0 * f)), interpolation=cv2.INTER_AREA)
        steps.append({"step": "resize", "detail": f"downscaled to {original.shape[1]}x{original.shape[0]} for processing"})

    detected, feats = classify_image(original)
    kind = detected if kind == "auto" else kind
    steps.append({"step": "classify", "detail": f"{kind} (palette top-8 {feats['palette_top8']:.2f}, "
                                                f"flat {feats['flat_share']:.2f}, edges {feats['edge_density']:.3f})"})

    x = original.astype(np.float32) / 255.0
    sigma, nlf = blind_sigma_map(x)
    steps.append({"step": "noise map", "detail": f"blind noise level function; median luma std "
                                                 f"{float(np.median(luma_sigma(sigma))) * 255:.2f}/255 (estimated)"})

    h, w = x.shape[:2]
    scale = target_scale((h, w), kind)
    target = (int(round(h * scale)), int(round(w * scale)))
    used: list[str] = []
    gate_map = np.ones(target, np.float32)

    if kind == "photo":
        y, s = x, sigma
        if float(_luma(x).mean()) < DARK_LUMA:
            from sensorsim.enhance.ncdce import CurveEnhancer

            y, s, _ = CurveEnhancer(weights.path("nc_dce"), device=device)(x, sigma)
            used.append("nc_dce")
            steps.append({"step": "NC-DCE", "detail": f"mean luma {float(_luma(x).mean()):.2f} -> {float(_luma(y).mean()):.2f}; "
                                                      f"noise carried through the curves"})
        y, s = guided_filter_varying(y, s)
        steps.append({"step": "VP-denoise", "detail": "guided filter, eps = (1.5 sigma)^2 per pixel"})
        sr = SuperResolver(weights.path("real_esrgan_x4plus"), "x4plus", device=device)
        used.append("real_esrgan_x4plus")
        result = guarded_sr(y, s, sr, target)
        enhanced, gate_map = result.image, result.gate
        steps.append({"step": "G-ESRGAN", "detail": f"RealESRGAN_x4plus to {target[1]}x{target[0]} (x{scale:g}); "
                                                    f"mean gate {float(gate_map.mean()):.2f}; back-projection residual "
                                                    f"{result.consistency[0]:.4f} -> {result.consistency[-1]:.4f}"})
    else:
        sr = SuperResolver(weights.path("real_esrgan_general"), "general", device=device)
        used.append("real_esrgan_general")
        result = guarded_sr(x, sigma, sr, target)
        gate_map = result.gate
        enhanced, bg = snap_background(result.image)
        enhanced = _unsharp(enhanced, 0.8, 0.35)
        steps.append({"step": "G-ESRGAN", "detail": f"realesr-general-x4v3 to {target[1]}x{target[0]} (x{scale:g}); "
                                                    f"mean gate {float(gate_map.mean()):.2f}"})
        steps.append({"step": "finish", "detail": f"background {np.round(np.array(bg) * 255).astype(int).tolist()} -> white; unsharp 0.35"})

    bicubic = cv2.resize(x, (target[1], target[0]), interpolation=cv2.INTER_CUBIC)
    nlm_h = float(np.clip(np.median(luma_sigma(sigma)) * 255.0 * 1.2, 1.0, 15.0))
    classical = cv2.fastNlMeansDenoisingColored(_u8(bicubic), None, nlm_h, nlm_h, 7, 21).astype(np.float32) / 255.0
    classical = _unsharp(classical, 1.0, 0.6)

    panels = {"input_bicubic": _u8(bicubic), "classical": _u8(classical), "nag": _u8(enhanced)}
    for key, img in panels.items():
        cv2.imwrite(str(out_dir / f"{key}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "original.png"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "gate.png"), _u8(gate_map))
    comparison_figure(
        [("Bicubic", panels["input_bicubic"]), ("Classical", panels["classical"]), ("NAG (ours)", panels["nag"])],
        out_dir / "comparison.png",
    )

    metrics = {key: _metrics(img, original) for key, img in panels.items()}
    record = {
        "name": name,
        "stem": stem,
        "id": out_dir.name,
        "kind": kind,
        "features": feats,
        "input_shape": [int(h), int(w)],
        "output_shape": list(target),
        "scale": scale,
        "noise_level_function": nlf,
        "steps": steps,
        "checkpoints": {n: weights.expected_sha(n) for n in used},
        "metrics": metrics,
        "seconds": round(time.perf_counter() - started, 2),
        "out_dir": out_dir.as_posix(),
        "files": {k: (out_dir / f"{k}.png").as_posix() for k in ("original", "input_bicubic", "classical", "nag", "gate", "comparison")},
    }
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(stable_json({k: record[k] for k in ("name", "kind", "scale", "checkpoints", "steps")}),
                                           encoding="utf-8")
    return record


def run_upload(
    source: str | Path,
    kind: str = "auto",
    device: str = "auto",
    update_paper: bool = True,
    write_report: bool = True,
    tex: str | Path = "paper/overleaf/main.tex",
    report_dir: str | Path = "temp",
    cover: dict[str, str | None] | None = None,
    rebuild_zip: bool = False,
    original_name: str | None = None,
    out_root: str | Path = OUT_ROOT,
) -> dict[str, Any]:
    """Enhance, then (optionally) update the paper and write both reports."""
    record = enhance_image(source, kind=kind, device=device, out_root=out_root, original_name=original_name)
    outputs: dict[str, str] = {}

    latex_block = None
    if update_paper:
        from sensorsim.enhance.latex import insert_upload_figure

        res = insert_upload_figure(record, tex=tex, rebuild_zip=rebuild_zip)
        outputs["tex"] = res["tex"]
        outputs["tex_backup"] = res["backup"]
        latex_block = res["block"]
    else:
        from sensorsim.enhance.latex import upload_block

        latex_block = upload_block(record)
    snippet = Path(record["out_dir"]) / "latex_snippet.tex"
    snippet.write_text(latex_block, encoding="utf-8")
    outputs["latex_snippet"] = snippet.as_posix()

    if write_report:
        from sensorsim.enhance.report import fill_rpt_template, write_upload_report

        outputs.update(write_upload_report(record, latex_block, out_dir=report_dir))
        try:
            outputs["rpt_docx"] = fill_rpt_template(record, latex_block, out_dir=report_dir, cover=cover or {})
        except FileNotFoundError as exc:
            outputs["rpt_docx_error"] = str(exc)

    summary = {
        "kind": record["kind"],
        "scale": record["scale"],
        "output_shape": record["output_shape"],
        "seconds": record["seconds"],
        "metrics": {k: {m: round(v, 3) for m, v in d.items()} for k, d in record["metrics"].items()},
        "outputs": {**record["files"], **outputs},
    }
    return {"record": record, "summary": summary}
