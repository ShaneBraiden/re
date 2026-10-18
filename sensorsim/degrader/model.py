"""Stage 1 - the forward sensor model.

    degrade(raw_linear, params) -> (degraded_raw, DegradationRecord)

Order of operations
-------------------
The default order is ``optical``, which follows the physical light path::

    PSF blur -> pixel binning -> photon scaling -> shot noise
             -> read noise -> quantisation

ARCHITECTURE.md's diagram lists blur *after* the noise. That order is available
as ``order="listed"`` for comparison, but it is not the default and should not
be used for headline numbers: convolving after the noise is added low-pass
filters the noise itself, so the degraded frame comes out cleaner than the
hardware it is meant to emulate. Recovery ratios measured against an
artificially clean baseline are inflated, which is the same class of error as
using signal-independent Gaussian noise. Every :class:`DegradationRecord`
carries the order it used, so a run made under either convention is
self-describing.

Everything here runs on linear, mosaiced RAW - before demosaicing, as Stage 1
requires. The output is deliberately left *dark* when ``alpha < 1``: the photon
deficit is the hardware deficit under study, and making a pipeline earn the
brightness back with gain is the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np

from sensorsim.degrader.noise import SensorNoiseModel, apply_photon_noise
from sensorsim.degrader.optics import make_psf, apply_psf_mosaic, mtf_from_psf, mtf50
from sensorsim.degrader.sampling import bin_bayer, quantize
from sensorsim.io.manifest import content_hash
from sensorsim.io.raw import RawImage

Order = Literal["optical", "listed"]


@dataclass(frozen=True)
class DegradationParams:
    """One point on the degradation grid.

    Ranges follow the table in ARCHITECTURE.md section 3.

    ============  =========  =====================================
    Parameter     Range      Emulates
    ============  =========  =====================================
    alpha         1.0-0.02   smaller aperture / area / exposure
    sigma_r_dn    0-12 DN    cheaper readout electronics
    r_psf         0-3.0 px   diffraction and inferior optics
    binning       1, 2, 3    smaller effective pixel pitch
    bit_depth     14-8       cheaper ADC
    ============  =========  =====================================
    """

    alpha: float = 1.0
    sigma_r_dn: float = 0.0
    r_psf: float = 0.0
    binning: int = 1
    bit_depth: int = 14
    psf_kind: str = "gaussian"
    order: Order = "optical"
    severity: float = 0.0
    tier: str = "clean"

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.sigma_r_dn < 0:
            raise ValueError(f"sigma_r_dn must be >= 0, got {self.sigma_r_dn}")
        if self.r_psf < 0:
            raise ValueError(f"r_psf must be >= 0, got {self.r_psf}")
        if self.binning < 1:
            raise ValueError(f"binning must be >= 1, got {self.binning}")
        if not 1 <= self.bit_depth <= 16:
            raise ValueError(f"bit_depth must be in 1..16, got {self.bit_depth}")
        if self.order not in ("optical", "listed"):
            raise ValueError(f"order must be optical or listed, got {self.order!r}")

    @property
    def params_hash(self) -> str:
        """Stable identity of this condition - the join key for every table."""
        return content_hash(asdict(self))

    def describe(self) -> str:
        return (
            f"alpha={self.alpha:g} sigma_r={self.sigma_r_dn:g}DN "
            f"psf={self.r_psf:g}px({self.psf_kind}) bin={self.binning} bits={self.bit_depth}"
        )


@dataclass
class DegradationRecord:
    """What was done to a frame, and what it cost in measurable terms."""

    params: dict[str, Any]
    params_hash: str
    image_id: str
    seed: int
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    order: str
    sensor: dict[str, float]
    measured: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        d["output_shape"] = list(self.output_shape)
        return d


def degrade(
    raw: RawImage,
    params: DegradationParams,
    sensor: SensorNoiseModel | None = None,
    seed: int = 0,
    prnu_field: np.ndarray | None = None,
) -> tuple[RawImage, DegradationRecord]:
    """Apply the forward sensor model to one clean linear RAW frame.

    Parameters
    ----------
    raw:
        Clean reference frame, linear and mosaiced.
    params:
        The condition to emulate.
    sensor:
        Sensor constants. Defaults are derived from the frame's own metadata so
        a corpus with per-image full-well values stays consistent.
    seed:
        Combined with the image id and the parameter hash, so the same condition
        on the same image always draws the same noise, and different conditions
        never share a noise realisation by accident.
    prnu_field:
        Fixed-pattern gain field at the *clean* frame's resolution. Pass the same
        field across a sweep to keep PRNU fixed rather than re-rolled per
        condition. It is binned alongside the signal, which is also physically
        right: a binned pixel averages its constituents' gain errors, so its
        PRNU is smaller by the same factor its shot noise is.

    Returns
    -------
    (degraded_raw, record)
    """
    sensor = sensor or SensorNoiseModel(
        full_well=raw.full_well,
        black_level=raw.black_level,
        white_level=raw.white_level,
        sigma_read_dn=params.sigma_r_dn,
    )
    # Read noise is a property of the condition, not of the reference sensor.
    sensor = SensorNoiseModel(
        full_well=sensor.full_well,
        black_level=sensor.black_level,
        white_level=sensor.white_level,
        sigma_read_dn=params.sigma_r_dn,
        dark_current_e=sensor.dark_current_e,
        prnu=sensor.prnu,
    )

    # Deterministic per (image, condition, seed): reruns reproduce bit for bit.
    stream = int(content_hash(f"{raw.image_id}:{params.params_hash}:{seed}", 8), 16)
    rng = np.random.default_rng(stream)

    signal = np.clip(raw.data.astype(np.float64), 0.0, 1.0)
    psf = make_psf(params.psf_kind, params.r_psf)
    # Negative headroom from the black level pedestal. Read noise must be able to
    # swing below zero, or the darkest frames come out biased bright.
    floor = -sensor.black_level / max(sensor.white_level - sensor.black_level, 1.0)

    if params.order == "optical":
        blurred = apply_psf_mosaic(signal, psf, raw.cfa_pattern)
        binned = bin_bayer(blurred, params.binning, raw.cfa_pattern)
        binned_prnu = (
            bin_bayer(prnu_field, params.binning, raw.cfa_pattern)
            if prnu_field is not None
            else None
        )
        noisy = apply_photon_noise(
            binned,
            sensor,
            photon_scale=params.alpha,
            well_multiplier=float(params.binning**2),
            rng=rng,
            prnu_field=binned_prnu,
            floor=floor,
        )
    else:  # "listed" - the diagram's order, kept for comparison only
        noisy_full = apply_photon_noise(
            signal,
            sensor,
            photon_scale=params.alpha,
            well_multiplier=1.0,
            rng=rng,
            prnu_field=prnu_field,
            floor=floor,
        )
        blurred = apply_psf_mosaic(noisy_full, psf, raw.cfa_pattern)
        noisy = bin_bayer(blurred, params.binning, raw.cfa_pattern)

    degraded = quantize(noisy, params.bit_depth, floor=floor)

    out = raw.replace(
        degraded,
        bit_depth=params.bit_depth,
        full_well=raw.full_well * params.binning**2,
        source=f"degraded:{params.params_hash}",
        extra={**raw.extra, "degradation": params.params_hash, "clean_shape": list(raw.shape)},
    )

    freqs, mtf = mtf_from_psf(psf)
    record = DegradationRecord(
        params=asdict(params),
        params_hash=params.params_hash,
        image_id=raw.image_id,
        seed=stream,
        input_shape=raw.shape,
        output_shape=out.shape,
        order=params.order,
        sensor={
            "full_well": sensor.full_well,
            "gain_e_per_dn": sensor.gain_e_per_dn,
            "sigma_read_dn": sensor.sigma_read_dn,
            "sigma_read_e": sensor.sigma_read_e,
            "dark_current_e": sensor.dark_current_e,
            "prnu": sensor.prnu,
        },
        measured={
            "mtf50_psf": mtf50(freqs, mtf),
            "mean_signal_e": float(np.mean(signal) * sensor.full_well * params.alpha),
            "mean_level_degraded": float(np.mean(degraded)),
            "mean_level_clean": float(np.mean(signal)),
            "shot_limited_snr": _shot_limited_snr(signal, sensor, params),
        },
    )
    return out, record


def _shot_limited_snr(signal: np.ndarray, sensor: SensorNoiseModel, params: DegradationParams) -> float:
    """Predicted SNR in dB at the frame's mean level, from theory alone.

    Useful as a sanity check on a run: the measured SNR of a degraded frame
    should land near this, and a large gap means the model was misconfigured.
    """
    e = float(np.mean(signal)) * sensor.full_well * params.alpha * params.binning**2
    if e <= 0:
        return float("-inf")
    read_e = sensor.sigma_read_e
    noise_e = np.sqrt(e + read_e**2)
    return float(20.0 * np.log10(e / noise_e))


def sweep(
    base: DegradationParams,
    axis: str,
    values: list[float] | list[int],
    severity_from: str | None = None,
) -> list[DegradationParams]:
    """Vary one axis of the grid, holding the rest fixed.

    ``severity_from`` names the axis whose normalised position becomes the
    ``severity`` label - the x-coordinate every recovery curve is plotted
    against. It defaults to ``axis``.
    """
    if not hasattr(base, axis):
        raise ValueError(f"{axis!r} is not a degradation parameter")
    severity_axis = severity_from or axis
    ref = [float(getattr(_replace(base, axis, v), severity_axis)) for v in values]
    lo, hi = min(ref), max(ref)

    out: list[DegradationParams] = []
    for value, r in zip(values, ref):
        frac = 0.0 if hi == lo else (r - lo) / (hi - lo)
        # alpha runs the other way: less light is more severe.
        sev = 1.0 - frac if severity_axis == "alpha" else frac
        out.append(_replace(base, axis, value, severity=round(sev, 4), tier=f"{axis}={value:g}"))
    return out


def _replace(params: DegradationParams, axis: str, value: Any, **extra: Any) -> DegradationParams:
    fields = asdict(params)
    fields[axis] = int(value) if axis in ("binning", "bit_depth") else value
    fields.update(extra)
    return DegradationParams(**fields)


def params_from_config(entry: dict[str, Any], defaults: dict[str, Any] | None = None) -> DegradationParams:
    """Build params from a config dict, ignoring keys the dataclass does not own."""
    merged = {**(defaults or {}), **entry}
    known = set(DegradationParams.__dataclass_fields__)
    unknown = set(merged) - known
    if unknown:
        raise ValueError(f"unknown degradation keys {sorted(unknown)}; known: {sorted(known)}")
    return DegradationParams(**merged)
