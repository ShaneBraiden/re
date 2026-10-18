# Notebooks

Exploration only. Nothing here is load-bearing.

If a result matters, it belongs in `sensorsim/` with a test, and its figure
belongs in `analyze/figures.py` so `sensorsim reproduce` regenerates it. A
number that exists only in a notebook cannot be reproduced, reviewed, or
defended when someone asks where it came from.

Reasonable uses:

- looking at a single degraded frame while tuning a config
- plotting a metrics table by hand before deciding it deserves a real figure
- sanity-checking a new pipeline before registering it
