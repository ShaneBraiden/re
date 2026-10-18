"""The read-only dashboard: a window onto a finished run, not a compute layer."""

from dashboard.app import ExperimentStore, create_app

__all__ = ["ExperimentStore", "create_app"]
