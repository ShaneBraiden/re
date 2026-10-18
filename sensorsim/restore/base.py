"""Stage 2 - the restoration pipeline interface and registry.

Every pipeline satisfies exactly one interface::

    class Pipeline:
        name: str
        def restore(self, degraded_raw, meta) -> rgb_uint8

Pipelines are added by a config entry, not a code change: a config names a
registered ``kind`` and the parameters to build it with, so a sweep can add an
arm without touching the runner.

Two rules this module enforces, because breaking either quietly corrupts the
study:

1. **A pipeline that cannot run reports itself unavailable.** It never
   substitutes a different algorithm. A missing weights file must show up as a
   gap in the results table, never as a number produced by something else.
2. **Outputs are cached by content hash** of (input frame, pipeline identity,
   parameters, weights version). Re-running a sweep recomputes nothing that has
   not changed, and a changed weights file invalidates its own cache entries.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from sensorsim.io.manifest import content_hash, file_checksum
from sensorsim.io.raw import RawImage


@dataclass
class RestoreContext:
    """What a pipeline is told about the frame it has been handed.

    A pipeline may use any of this - a real ISP knows its own sensor - but it
    must not reach for the clean reference. ``target_shape`` is the reference
    resolution the output must be returned at, so a binned frame has to be
    resampled by whoever restores it.
    """

    image_id: str
    target_shape: tuple[int, int]
    degradation: dict[str, Any] = field(default_factory=dict)
    sensor: dict[str, float] = field(default_factory=dict)
    scene_kind: str = "natural"

    @property
    def alpha(self) -> float:
        """Photon scale the frame was captured at - the gain a pipeline must undo."""
        return float(self.degradation.get("alpha", 1.0))

    @property
    def binning(self) -> int:
        return int(self.degradation.get("binning", 1))


@dataclass(frozen=True)
class Availability:
    """Whether a pipeline can actually run here, and if not, what is missing."""

    ok: bool
    reason: str = ""
    missing: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok


class Pipeline(ABC):
    """Base class for every restoration arm."""

    #: Registry key. Set by :func:`register`.
    kind: str = "abstract"

    def __init__(self, name: str | None = None, **params: Any):
        self.name = name or self.kind
        self.params = params

    @abstractmethod
    def restore(self, degraded_raw: RawImage, meta: RestoreContext) -> np.ndarray:
        """Reconstruct an 8-bit RGB image at ``meta.target_shape``."""

    def availability(self) -> Availability:
        """Can this pipeline run? Override where dependencies or weights matter."""
        return Availability(True)

    def weights_version(self) -> str:
        """Identity of the weights in use. Pinned by checksum, never 'latest'."""
        path = self.params.get("weights")
        if path and Path(path).exists():
            return file_checksum(path)
        return "no-weights"

    def identity(self) -> dict[str, Any]:
        """Everything that changes this pipeline's output. Feeds the cache key."""
        return {
            "kind": self.kind,
            "name": self.name,
            "params": {k: v for k, v in self.params.items() if k != "weights"},
            "weights_version": self.weights_version(),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type[Pipeline]] = {}


def register(kind: str) -> Callable[[type[Pipeline]], type[Pipeline]]:
    """Class decorator that makes a pipeline addressable from a config."""

    def decorate(cls: type[Pipeline]) -> type[Pipeline]:
        if kind in _REGISTRY:
            raise ValueError(f"pipeline kind {kind!r} is already registered by {_REGISTRY[kind].__name__}")
        cls.kind = kind
        _REGISTRY[kind] = cls
        return cls

    return decorate


def registered_kinds() -> list[str]:
    return sorted(_REGISTRY)


def build_pipeline(spec: dict[str, Any]) -> Pipeline:
    """Instantiate one pipeline from a config entry.

    ``{"name": "classical_isp", "kind": "classical_isp", "params": {...}}``
    """
    if "kind" not in spec:
        raise ValueError(f"pipeline spec needs a 'kind': {spec}")
    kind = spec["kind"]
    if kind not in _REGISTRY:
        raise ValueError(f"unknown pipeline kind {kind!r}; registered: {registered_kinds()}")
    return _REGISTRY[kind](name=spec.get("name", kind), **spec.get("params", {}))


def build_bank(specs: Iterable[dict[str, Any]]) -> list[Pipeline]:
    """Instantiate a whole bank, rejecting duplicate names early.

    Duplicate names would collide in the metrics table and silently overwrite
    each other's rows, so they are an error rather than a warning.
    """
    bank: list[Pipeline] = []
    seen: set[str] = set()
    for spec in specs:
        pipe = build_pipeline(spec)
        if pipe.name in seen:
            raise ValueError(f"duplicate pipeline name {pipe.name!r} in bank")
        seen.add(pipe.name)
        bank.append(pipe)
    return bank


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

@dataclass
class RestoreResult:
    """One restored frame plus how it came to be."""

    image: np.ndarray
    pipeline: str
    cache_key: str
    cached: bool
    seconds: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def frame_hash(raw: RawImage) -> str:
    """Content hash of a frame: pixels and the metadata that changes their meaning."""
    h = content_hash(
        {
            "sha": _array_digest(raw.data),
            "pattern": raw.cfa_pattern,
            "bit_depth": raw.bit_depth,
            "wb": list(raw.wb_gains),
            "full_well": raw.full_well,
        }
    )
    return h


def _array_digest(a: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


def cache_key(raw: RawImage, pipeline: Pipeline, meta: RestoreContext) -> str:
    """Identity of a restoration result. Changing any input changes this."""
    return content_hash(
        {
            "frame": frame_hash(raw),
            "pipeline": pipeline.identity(),
            "target_shape": list(meta.target_shape),
        }
    )


def run_pipeline(
    pipeline: Pipeline,
    raw: RawImage,
    meta: RestoreContext,
    cache_dir: str | Path | None = None,
) -> RestoreResult:
    """Run one pipeline on one frame, honouring the cache.

    A pipeline that raises is reported as an error result rather than crashing
    the sweep - one broken arm should not cost the other arms their run - but
    the error is recorded and shows up as a gap in the table.
    """
    import cv2

    availability = pipeline.availability()
    if not availability:
        return RestoreResult(
            image=np.zeros((*meta.target_shape, 3), np.uint8),
            pipeline=pipeline.name,
            cache_key="",
            cached=False,
            seconds=0.0,
            error=f"unavailable: {availability.reason}",
        )

    key = cache_key(raw, pipeline, meta)
    path = Path(cache_dir) / f"{pipeline.name}_{key}.png" if cache_dir else None

    if path is not None and path.exists():
        cached = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if cached is not None:
            return RestoreResult(
                image=cv2.cvtColor(cached, cv2.COLOR_BGR2RGB),
                pipeline=pipeline.name,
                cache_key=key,
                cached=True,
                seconds=0.0,
            )

    started = time.perf_counter()
    try:
        # Validation is inside the try on purpose: an arm that violates the
        # interface is a broken arm, and a broken arm must not end the sweep
        # any more than one that raises. Both come back as an error result.
        image = _validate_output(pipeline.restore(raw, meta), meta, pipeline.name)
    except Exception as exc:  # noqa: BLE001
        return RestoreResult(
            image=np.zeros((*meta.target_shape, 3), np.uint8),
            pipeline=pipeline.name,
            cache_key=key,
            cached=False,
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    elapsed = time.perf_counter() - started

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return RestoreResult(image=image, pipeline=pipeline.name, cache_key=key, cached=False, seconds=elapsed)


def _validate_output(image: np.ndarray, meta: RestoreContext, name: str) -> np.ndarray:
    """Hold every pipeline to the interface: uint8 RGB at the target shape."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} returned shape {image.shape}; expected (H, W, 3)")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} returned dtype {image.dtype}; the interface is rgb_uint8")
    if image.shape[:2] != tuple(meta.target_shape):
        raise ValueError(
            f"{name} returned {image.shape[:2]} but the reference is {tuple(meta.target_shape)}. "
            "A pipeline handed a binned frame must resample to reference resolution - "
            "the evaluator compares pixel to pixel."
        )
    return image
