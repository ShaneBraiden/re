"""SensorSim - a controlled testbench for the sensor-algorithm substitution rate.

Four stages that communicate only through files and a manifest:

    degrade -> restore -> evaluate -> analyze

No stage imports another. That is what makes runs resumable and results
auditable; keep it that way.
"""

__version__ = "0.1.0"
