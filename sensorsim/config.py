"""Config loading and sweep construction.

Configs are YAML and compose by reference: an experiment names a degradation
sweep file and a pipeline bank file rather than inlining them, so a sweep can be
reused across experiments and a bank can gain an arm without touching any
experiment.

Sweep modes
-----------
``compound``
    Every parameter moves together along one severity axis from 0 (a perfect
    sensor) to 1 (the worst hardware in the study). This is the default and the
    one the recovery ratio is defined against, because severity 0 is then
    *exactly* the identity condition - the clean ceiling is a real measurement
    rather than an extrapolation.
``axis``
    One parameter varies, the rest stay fixed. For ablations: which deficit is
    doing the damage. The identity condition is prepended automatically so the
    ceiling still exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sensorsim.degrader.model import DegradationParams

SCALES = ("linear", "log", "step")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file, with a readable error when it is not a mapping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level, got {type(data).__name__}")
    return data


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and resolve its file references.

    ``degradation`` and ``pipelines`` may each be a path or an inline mapping.
    Paths resolve relative to the repository root, then relative to the config.
    """
    path = Path(path)
    config = load_yaml(path)

    for key in ("degradation", "pipelines"):
        value = config.get(key)
        if isinstance(value, str):
            candidate = Path(value)
            if not candidate.exists():
                candidate = path.parent / value
            config[key] = load_yaml(candidate)
            config[key]["_source"] = str(value)
        elif value is None:
            raise ValueError(f"experiment config {path} is missing required key {key!r}")

    config.setdefault("experiment_id", path.stem)
    config.setdefault("seed", 0)
    config.setdefault("output", f"experiments/{config['experiment_id']}")
    return config


def _interpolate(spec: dict[str, Any], t: float) -> float | int:
    """One parameter's value at severity ``t`` in [0, 1]."""
    start, end = float(spec["from"]), float(spec["to"])
    scale = spec.get("scale", "linear")
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {SCALES}")

    if scale == "log":
        if start <= 0 or end <= 0:
            raise ValueError(f"log scale needs positive endpoints, got from={start} to={end}")
        # Geometric interpolation: alpha halving per step is the natural spacing
        # for an exposure axis, where the perceptual step is the ratio.
        value = float(np.exp(np.log(start) + t * (np.log(end) - np.log(start))))
    elif scale == "step":
        value = float(np.round(start + t * (end - start)))
    else:
        value = float(start + t * (end - start))
    return value


def build_sweep(config: dict[str, Any]) -> list[DegradationParams]:
    """Turn a degradation config into the list of conditions to run."""
    mode = config.get("mode", "compound")
    defaults = dict(config.get("defaults", {}))
    defaults.pop("_source", None)

    if mode == "compound":
        return _compound_sweep(config, defaults)
    if mode == "axis":
        return _axis_sweep(config, defaults)
    raise ValueError(f"unknown sweep mode {mode!r}; expected compound or axis")


def _compound_sweep(config: dict[str, Any], defaults: dict[str, Any]) -> list[DegradationParams]:
    levels = [float(v) for v in config.get("levels", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])]
    schedule = config.get("schedule")
    if not schedule:
        raise ValueError("a compound sweep needs a 'schedule' mapping parameter -> {from, to, scale}")
    if not any(np.isclose(levels, 0.0)):
        # The ceiling in the recovery ratio is a measurement at severity 0. A
        # sweep without it would have to extrapolate one, so add it instead.
        levels = [0.0, *levels]

    integer_fields = {"binning", "bit_depth"}
    out: list[DegradationParams] = []
    for t in sorted(set(levels)):
        fields = dict(defaults)
        for name, spec in schedule.items():
            value = _interpolate(spec, t)
            fields[name] = int(round(value)) if name in integer_fields else value
        fields["severity"] = round(t, 4)
        fields["tier"] = "clean" if t == 0 else f"s{t:.2f}"
        out.append(DegradationParams(**fields))
    return out


def _axis_sweep(config: dict[str, Any], defaults: dict[str, Any]) -> list[DegradationParams]:
    axis = config.get("axis")
    values = config.get("values")
    if not axis or not values:
        raise ValueError("an axis sweep needs 'axis' and 'values'")

    fixed = {**defaults, **config.get("fixed", {})}
    integer_fields = {"binning", "bit_depth"}

    numeric = [float(v) for v in values]
    lo, hi = min(numeric), max(numeric)
    inverted = axis == "alpha"  # less light is more severe

    out = [DegradationParams(**{**{k: v for k, v in fixed.items() if k != axis},
                                "severity": 0.0, "tier": "clean"})]
    for value in values:
        frac = 0.0 if hi == lo else (float(value) - lo) / (hi - lo)
        severity = 1.0 - frac if inverted else frac
        fields = dict(fixed)
        fields[axis] = int(round(float(value))) if axis in integer_fields else float(value)
        fields["severity"] = round(severity, 4)
        fields["tier"] = f"{axis}={value}"
        out.append(DegradationParams(**fields))

    # Drop a duplicate identity condition if the swept values already contain it.
    seen: dict[str, DegradationParams] = {}
    for params in out:
        seen.setdefault(params.params_hash, params)
    return sorted(seen.values(), key=lambda p: p.severity)


def pipeline_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Pipeline entries from a bank config."""
    specs = config.get("pipelines")
    if not isinstance(specs, list) or not specs:
        raise ValueError("pipeline bank config needs a non-empty 'pipelines' list")
    out = []
    for spec in specs:
        if not isinstance(spec, dict) or "kind" not in spec:
            raise ValueError(f"each pipeline entry needs a 'kind', got {spec!r}")
        out.append({"name": spec.get("name", spec["kind"]), "kind": spec["kind"], "params": spec.get("params", {})})
    return out
