"""Stage 2 - the restoration pipeline bank.

Importing this package registers every pipeline kind, so a config can name any
of them without the runner knowing they exist.
"""

from sensorsim.restore.base import (
    Availability,
    Pipeline,
    RestoreContext,
    RestoreResult,
    build_bank,
    build_pipeline,
    cache_key,
    register,
    registered_kinds,
    run_pipeline,
)
from sensorsim.restore import classical as _classical  # noqa: F401  (registration)
from sensorsim.restore import learned as _learned  # noqa: F401  (registration)

__all__ = [
    "Availability",
    "Pipeline",
    "RestoreContext",
    "RestoreResult",
    "build_bank",
    "build_pipeline",
    "cache_key",
    "register",
    "registered_kinds",
    "run_pipeline",
]
