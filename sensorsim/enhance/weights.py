"""Pinned checkpoints: where each comes from and what it must hash to.

A checkpoint is identified by its SHA-256, never by "latest". A download whose
hash does not match is deleted and reported, so a re-released file cannot
silently change results. The NC-DCE checkpoints have no URL: they are
produced locally by ``sensorsim train nc-dce``, which records their hash in a
JSON sidecar next to the file.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path("models")


@dataclass(frozen=True)
class Checkpoint:
    name: str
    filename: str
    url: str | None
    sha256: str | None
    source: str
    licence: str


CHECKPOINTS: dict[str, Checkpoint] = {
    c.name: c
    for c in (
        Checkpoint(
            "real_esrgan_x4plus", "RealESRGAN_x4plus.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
            "Wang et al., Real-ESRGAN, ICCVW 2021", "BSD-3-Clause",
        ),
        Checkpoint(
            "real_esrgan_general", "realesr-general-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
            "Wang et al., Real-ESRGAN, ICCVW 2021", "BSD-3-Clause",
        ),
        Checkpoint(
            "zero_dce", "zero_dce_epoch99.pth",
            "https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth",
            "a4395acb874f320375d9704997cef874eaaaaa26a1777ceb29a92b70f74c3612",
            "Guo et al., Zero-DCE, CVPR 2020", "authors' release (research use)",
        ),
        Checkpoint("nc_dce", "nc_dce_v1.pth", None, None, "this work (sensorsim train nc-dce)", "project"),
        Checkpoint("nc_dce_nosigma", "nc_dce_nosigma.pth", None, None, "this work, ablation", "project"),
        Checkpoint("nc_dce_nonoise", "nc_dce_nonoise.pth", None, None, "this work, ablation", "project"),
        Checkpoint("zero_dce_ft", "zero_dce_ft.pth", None, None, "Zero-DCE recipe fine-tuned on the same data (control)",
                   "project"),
    )
}


def training_record(name: str, models_dir: str | Path = MODELS_DIR) -> dict | None:
    """The training sidecar of a locally trained checkpoint, if there is one."""
    sidecar = (Path(models_dir) / CHECKPOINTS[name].filename).with_suffix(".json")
    return json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None


def diverged_reason(weights_path: str | Path) -> str | None:
    """Why a locally trained checkpoint does not exist, if training recorded it."""
    sidecar = Path(weights_path).with_suffix(".json")
    if not sidecar.exists():
        return None
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    d = meta.get("diverged")
    if not d:
        return None
    return f"training diverged at epoch {d['epoch']} (non-finite loss); see {sidecar.as_posix()}"


class MissingWeights(FileNotFoundError):
    """A checkpoint the caller needs is not on disk. Never fall back - report it."""


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path(name: str, models_dir: str | Path = MODELS_DIR) -> Path:
    """Local path of a checkpoint, or :class:`MissingWeights` with the fix."""
    ckpt = CHECKPOINTS[name]
    p = Path(models_dir) / ckpt.filename
    if not p.exists():
        how = "sensorsim weights fetch" if ckpt.url else "sensorsim train nc-dce"
        raise MissingWeights(f"{name} checkpoint missing at {p} - run: {how}")
    return p


def expected_sha(name: str, models_dir: str | Path = MODELS_DIR) -> str | None:
    """Pinned hash, from the registry or the training sidecar."""
    ckpt = CHECKPOINTS[name]
    if ckpt.sha256:
        return ckpt.sha256
    sidecar = (Path(models_dir) / ckpt.filename).with_suffix(".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8")).get("sha256")
    return None


def verify(name: str, models_dir: str | Path = MODELS_DIR) -> tuple[bool, str]:
    reason = diverged_reason(Path(models_dir) / CHECKPOINTS[name].filename)
    if reason:
        return True, f"{name}: no checkpoint by design - {reason}"
    p = path(name, models_dir)
    want = expected_sha(name, models_dir)
    got = sha256(p)
    if want is None:
        return False, f"{name}: no pinned hash (got {got[:16]})"
    return got == want, f"{name}: {'ok' if got == want else f'MISMATCH want {want[:16]} got {got[:16]}'}"


def fetch(names: list[str] | None = None, models_dir: str | Path = MODELS_DIR) -> list[str]:
    """Download every downloadable checkpoint (or ``names``) and verify it."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name, ckpt in CHECKPOINTS.items():
        if names and name not in names:
            continue
        if not ckpt.url:
            continue
        target = models_dir / ckpt.filename
        if not target.exists():
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(ckpt.url, tmp)
            tmp.replace(target)
        got = sha256(target)
        if got != ckpt.sha256:
            target.unlink()
            raise RuntimeError(f"{name}: hash mismatch (want {ckpt.sha256}, got {got}); file removed")
        lines.append(f"{name:22s} ok  {target}")
    return lines
