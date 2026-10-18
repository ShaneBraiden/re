"""Linear RAW representation and loaders.

Everything upstream of Stage 2 works on *linear* sensor data with the colour
filter array still in place. Demosaicing is a restoration decision, not a
loading one, so nothing here demosaics.

Storage format
--------------
The native format is a ``.npy`` array of float32 normalised linear values in
[0, 1] plus a ``.json`` sidecar holding the sensor metadata. It is boring on
purpose: it round-trips exactly, needs no third-party decoder, and lets the
synthetic corpus and real captures travel through identical code.

Real camera RAW (``.dng``, ``.arw``, ``.cr2``, ...) is read through ``rawpy``
when it is installed. Without it, :func:`load_raw` raises with an actionable
message rather than guessing at a decode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

# CFA patterns are named by their top-left 2x2 tile, reading row-major.
CFA_PATTERNS: dict[str, tuple[str, str, str, str]] = {
    "RGGB": ("R", "G", "G", "B"),
    "BGGR": ("B", "G", "G", "R"),
    "GRBG": ("G", "R", "B", "G"),
    "GBRG": ("G", "B", "R", "G"),
}

RAWPY_SUFFIXES = {".dng", ".arw", ".cr2", ".cr3", ".nef", ".raf", ".rw2", ".orf", ".srw"}


@dataclass
class RawImage:
    """A linear, mosaiced sensor frame plus the metadata needed to interpret it.

    Attributes
    ----------
    data:
        float32, shape (H, W), normalised linear scene radiance in [0, 1] where
        1.0 is full well. Black level is already subtracted.
    cfa_pattern:
        Key into :data:`CFA_PATTERNS`. ``data[0, 0]`` is the first letter.
    black_level, white_level:
        The DN values the normalisation was derived from. Kept so a pipeline can
        reason in DN and so quantisation steps land where the real sensor's do.
    bit_depth:
        Bits in the ADC that produced (or last requantised) this frame.
    wb_gains:
        Per-channel (R, G, B) multipliers that neutralise the illuminant. The
        camera as-shot values; a pipeline may ignore or re-estimate them.
    full_well:
        Electrons at saturation. The photon-scale conversion in the degrader
        needs this; it is the one number that makes shot noise physical rather
        than decorative.
    """

    data: np.ndarray
    cfa_pattern: str = "RGGB"
    black_level: float = 512.0
    white_level: float = 16383.0
    bit_depth: int = 14
    wb_gains: tuple[float, float, float] = (2.0, 1.0, 1.5)
    full_well: float = 15000.0
    image_id: str = "unknown"
    source: str = "unknown"
    scene_kind: str = "natural"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 2:
            raise ValueError(f"RAW data must be 2-D mosaiced, got shape {self.data.shape}")
        if self.cfa_pattern not in CFA_PATTERNS:
            raise ValueError(f"unknown CFA pattern {self.cfa_pattern!r}")
        if self.data.shape[0] % 2 or self.data.shape[1] % 2:
            raise ValueError(f"RAW dimensions must be even for a 2x2 CFA, got {self.data.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    def meta(self) -> dict[str, Any]:
        """Metadata only - everything except the pixels."""
        d = asdict(self)
        d.pop("data")
        d["wb_gains"] = list(self.wb_gains)
        d["shape"] = list(self.data.shape)
        return d

    def replace(self, data: np.ndarray, **overrides: Any) -> "RawImage":
        """A copy carrying new pixels (and optionally new metadata)."""
        fields = {k: v for k, v in self.meta().items() if k != "shape"}
        fields["wb_gains"] = tuple(fields["wb_gains"])
        fields.update(overrides)
        return RawImage(data=data, **fields)


def bayer_to_planes(raw: np.ndarray, pattern: str = "RGGB") -> dict[str, np.ndarray]:
    """Split a mosaic into its four half-resolution sub-planes.

    Returns a dict keyed ``R``, ``G1``, ``G2``, ``B``. The two greens are kept
    apart because they sit on different rows and a real sensor's two green
    channels genuinely differ (crosstalk, microlens shift); averaging them here
    would quietly discard that.
    """
    names = CFA_PATTERNS[pattern]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    out: dict[str, np.ndarray] = {}
    green_seen = 0
    for name, (r, c) in zip(names, positions):
        if name == "G":
            green_seen += 1
            key = f"G{green_seen}"
        else:
            key = name
        out[key] = raw[r::2, c::2]
    return out


def planes_to_bayer(planes: dict[str, np.ndarray], pattern: str = "RGGB") -> np.ndarray:
    """Inverse of :func:`bayer_to_planes`."""
    names = CFA_PATTERNS[pattern]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    h, w = next(iter(planes.values())).shape
    out = np.zeros((h * 2, w * 2), dtype=np.float32)
    green_seen = 0
    for name, (r, c) in zip(names, positions):
        if name == "G":
            green_seen += 1
            key = f"G{green_seen}"
        else:
            key = name
        out[r::2, c::2] = planes[key]
    return out


def cfa_masks(shape: tuple[int, int], pattern: str = "RGGB") -> dict[str, np.ndarray]:
    """Boolean full-resolution masks for R, G and B sample positions."""
    names = CFA_PATTERNS[pattern]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    h, w = shape
    masks = {c: np.zeros((h, w), dtype=bool) for c in "RGB"}
    for name, (r, c) in zip(names, positions):
        masks[name][r::2, c::2] = True
    return masks


def save_raw(path: str | Path, raw: RawImage) -> Path:
    """Write ``<path>.npy`` plus a ``.json`` sidecar. Returns the array path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    np.save(path, raw.data.astype(np.float32))
    path.with_suffix(".json").write_text(json.dumps(raw.meta(), indent=2), encoding="utf-8")
    return path


def load_raw(path: str | Path) -> RawImage:
    """Load a RAW frame from the native format or, with rawpy, a camera file."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        data = np.load(path)
        sidecar = path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if sidecar.exists():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            meta.pop("shape", None)
            if "wb_gains" in meta:
                meta["wb_gains"] = tuple(meta["wb_gains"])
        return RawImage(data=data, **meta)

    if suffix in RAWPY_SUFFIXES:
        return _load_with_rawpy(path)

    known = ", ".join(sorted(RAWPY_SUFFIXES))
    raise ValueError(
        f"unsupported RAW file {path.name!r}. Native format is .npy + .json sidecar; "
        f"camera RAW ({known}) needs: pip install rawpy"
    )


def _load_with_rawpy(path: Path) -> RawImage:
    try:
        import rawpy  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            f"reading {path.name} needs rawpy (pip install sensorsim[raw]). "
            "The synthetic corpus and any .npy frame load without it."
        ) from exc

    with rawpy.imread(str(path)) as rp:  # pragma: no cover - needs a real camera file
        visible = rp.raw_image_visible.astype(np.float32)
        black = float(np.mean(rp.black_level_per_channel))
        white = float(rp.white_level)
        pattern = "".join("RGBG"[i] for i in rp.raw_pattern.flatten())
        wb = rp.camera_whitebalance[:3]
        g = float(wb[1]) or 1.0
        data = np.clip((visible - black) / max(white - black, 1.0), 0.0, 1.0)
        # Trim to an even 2x2 tile count so CFA indexing stays valid.
        h, w = data.shape
        data = data[: h - h % 2, : w - w % 2]
        return RawImage(
            data=data,
            cfa_pattern=pattern if pattern in CFA_PATTERNS else "RGGB",
            black_level=black,
            white_level=white,
            bit_depth=int(np.ceil(np.log2(max(white, 2)))),
            wb_gains=(float(wb[0]) / g, 1.0, float(wb[2]) / g),
            image_id=path.stem,
            source=str(path),
        )
