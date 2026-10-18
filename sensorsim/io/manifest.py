"""Run manifests - the audit trail that makes a number defensible.

Section 7 of ARCHITECTURE.md asks every run to record config hash, git commit,
weights checksums, seeds and hardware. This module is the whole of that
requirement: build a :class:`RunManifest`, write it beside the outputs, and
``make reproduce`` has everything it needs to regenerate the numbers.

Hashing is over *canonical* JSON (sorted keys, fixed separators, floats
normalised) so the same config always hashes the same, whatever order the YAML
happened to be written in.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def _canonical(obj: Any) -> Any:
    """Normalise a structure so equal configs hash equally."""
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        # 12 significant digits: below float64 noise, above anything we tune by.
        if obj != obj or obj in (float("inf"), float("-inf")):
            return str(obj)
        return float(f"{obj:.12g}")
    if isinstance(obj, Path):
        return obj.as_posix()
    if hasattr(obj, "item") and hasattr(obj, "dtype"):  # numpy scalar
        return _canonical(obj.item())
    return obj


def stable_json(obj: Any) -> str:
    """Canonical JSON text for hashing or for writing a config back out."""
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), default=str)


def content_hash(obj: Any, length: int = 12) -> str:
    """Short, stable content hash of any JSON-able structure."""
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()[:length]


def file_checksum(path: str | Path, length: int = 16) -> str:
    """SHA-256 of a file, truncated. Used to pin model weights by content."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def git_commit(repo: str | Path = ".") -> str:
    """Current commit, with a ``-dirty`` suffix when the tree has changes."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, timeout=10,
        )
        if rev.returncode != 0:
            # An initialised repo with no commits is a different problem from no
            # repo at all, and it has a different fix, so do not report them the
            # same way.
            inside = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=str(repo), capture_output=True, text=True, timeout=10,
            )
            return "no-commits-yet" if inside.returncode == 0 else "not-a-git-repo"
        commit = rev.stdout.strip()[:12]
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo), capture_output=True, text=True, timeout=10,
        )
        return commit + ("-dirty" if status.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    """Versions of the libraries that can change a number if they change."""
    names = ["numpy", "scipy", "pandas", "cv2", "matplotlib", "PIL", "torch", "rawpy"]
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for name in names:
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            out[name] = "absent"
    return out


def hardware() -> dict[str, str]:
    """Enough about the machine to explain a timing difference."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_impl": platform.python_implementation(),
    }
    try:  # pragma: no cover - only meaningful with torch installed
        import torch  # type: ignore

        info["cuda"] = torch.version.cuda or "cpu-only"
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        info["cuda"] = "torch-absent"
        info["gpu"] = "none"
    return info


@dataclass
class RunManifest:
    """Everything needed to reproduce one experiment run."""

    experiment_id: str
    config: dict[str, Any]
    seed: int
    started_at: str = ""
    finished_at: str = ""
    config_hash: str = ""
    git_commit: str = ""
    packages: dict[str, str] = field(default_factory=dict)
    hardware_info: dict[str, str] = field(default_factory=dict)
    weights: dict[str, str] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)
    manifest_version: int = MANIFEST_VERSION

    @classmethod
    def create(
        cls,
        experiment_id: str,
        config: dict[str, Any],
        seed: int,
        repo: str | Path = ".",
    ) -> "RunManifest":
        return cls(
            experiment_id=experiment_id,
            config=config,
            seed=seed,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            config_hash=content_hash(config),
            git_commit=git_commit(repo),
            packages=package_versions(),
            hardware_info=hardware(),
        )

    def record_stage(self, name: str, **info: Any) -> None:
        """Note what a stage did - counts, durations, output paths."""
        self.stages[name] = {"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_canonical(asdict(self)), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        version = raw.pop("manifest_version", 0)
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"manifest at {path} is version {version}, this build reads {MANIFEST_VERSION}"
            )
        known = {f for f in cls.__dataclass_fields__}
        return cls(manifest_version=version, **{k: v for k, v in raw.items() if k in known})

    def reproducibility_report(self) -> list[str]:
        """Human-readable lines flagging anything that weakens reproducibility."""
        problems: list[str] = []
        if self.git_commit.endswith("-dirty"):
            problems.append("git tree was dirty at run time - the commit does not describe the code")
        if self.git_commit == "no-commits-yet":
            problems.append("the repository has no commits - commit before quoting these numbers")
        elif self.git_commit in {"unknown", "not-a-git-repo"}:
            problems.append("no git commit recorded - the code version is unpinned")
        for name, spec in self.weights.items():
            if not spec or spec == "unpinned":
                problems.append(f"weights for {name} are unpinned")
        return problems
