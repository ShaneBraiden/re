"""Stage 3 - the evaluator.

Emits one tidy table, long format, one row per
``image x degradation x pipeline x metric``::

    image_id, degradation_params_hash, pipeline, metric_family, metric_name, value

Long, not wide, because every downstream plot is a groupby on this and a wide
table would need reshaping before any of them. Adding a metric adds rows, never
columns, so nothing downstream breaks when the metric bank grows.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from sensorsim.evaluate import fidelity, machine_task, perceptual

FAMILIES = ("fidelity", "perceptual", "machine_task")

#: metric name -> True (higher better), False (lower better), None (target-valued).
DIRECTION: dict[str, bool | None] = {
    **fidelity.HIGHER_IS_BETTER,
    **perceptual.HIGHER_IS_BETTER,
    **machine_task.HIGHER_IS_BETTER,
}

FAMILY_OF: dict[str, str] = {
    **{k: "fidelity" for k in fidelity.HIGHER_IS_BETTER},
    **{k: "perceptual" for k in perceptual.HIGHER_IS_BETTER},
    **{k: "machine_task" for k in machine_task.HIGHER_IS_BETTER},
}


@dataclass(frozen=True)
class MetricRow:
    """One cell of the tidy table."""

    image_id: str
    degradation_params_hash: str
    pipeline: str
    metric_family: str
    metric_name: str
    value: float
    scene_kind: str = "natural"
    severity: float = 0.0
    status: str = "ok"          # ok | unavailable | error
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_frame(
    restored: np.ndarray,
    reference: np.ndarray,
    *,
    image_id: str,
    degradation_params_hash: str,
    pipeline: str,
    scene_kind: str = "natural",
    severity: float = 0.0,
    crops: dict[str, Any] | None = None,
    scene_text: str | None = None,
    families: tuple[str, ...] = FAMILIES,
    include_optional: bool = True,
) -> list[MetricRow]:
    """Every metric in ``families`` for one restored frame."""
    rows: list[MetricRow] = []

    def emit(family: str, values: dict[str, float]) -> None:
        for name, value in values.items():
            rows.append(
                MetricRow(
                    image_id=image_id,
                    degradation_params_hash=degradation_params_hash,
                    pipeline=pipeline,
                    metric_family=family,
                    metric_name=name,
                    value=float(value),
                    scene_kind=scene_kind,
                    severity=severity,
                )
            )

    if "fidelity" in families:
        emit("fidelity", fidelity.compute(restored, reference))
    if "perceptual" in families:
        emit("perceptual", perceptual.compute(restored, reference, include_optional=include_optional))
    if "machine_task" in families:
        emit("machine_task", machine_task.compute(restored, reference, crops=crops, scene_text=scene_text))

    return rows


def unavailable_rows(
    *,
    image_id: str,
    degradation_params_hash: str,
    pipeline: str,
    scene_kind: str = "natural",
    severity: float = 0.0,
    reason: str = "",
) -> list[MetricRow]:
    """Explicit rows for a pipeline that could not run.

    A gap must be *in* the table. An absent row reads as "not measured yet"; a
    row with status ``unavailable`` reads as "measured, and it could not run" -
    and only the second one survives a reviewer asking why the bank has holes.
    """
    return [
        MetricRow(
            image_id=image_id,
            degradation_params_hash=degradation_params_hash,
            pipeline=pipeline,
            metric_family=family,
            metric_name="__all__",
            value=float("nan"),
            scene_kind=scene_kind,
            severity=severity,
            status="unavailable",
            note=reason,
        )
        for family in FAMILIES
    ]


def missing_metrics() -> dict[str, str]:
    """Optional metrics that cannot run in this environment, and why."""
    return {**perceptual.unavailable(), **machine_task.unavailable()}


def to_dataframe(rows: list[MetricRow]):
    """Tidy long DataFrame. Import of pandas is deferred so the core stays light."""
    import pandas as pd

    return pd.DataFrame([r.as_dict() for r in rows])


__all__ = [
    "DIRECTION",
    "FAMILIES",
    "FAMILY_OF",
    "MetricRow",
    "evaluate_frame",
    "fidelity",
    "machine_task",
    "missing_metrics",
    "perceptual",
    "to_dataframe",
    "unavailable_rows",
]
