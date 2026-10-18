"""Stage 1 - forward sensor model.

Degrades clean linear RAW to emulate weaker hardware, before demosaicing.
"""

from sensorsim.degrader.model import (
    DegradationParams,
    DegradationRecord,
    degrade,
    params_from_config,
    sweep,
)
from sensorsim.degrader.noise import (
    SensorNoiseModel,
    apply_photon_noise,
    fit_noise_params,
    photon_transfer_curve,
)
from sensorsim.degrader.optics import make_psf, apply_psf_mosaic, mtf_from_psf, mtf50, slanted_edge_mtf
from sensorsim.degrader.sampling import bin_bayer, quantize, upsample_to
from sensorsim.degrader.validate import (
    Tolerance,
    ValidationError,
    ValidationReport,
    calibrate_from_stack,
    self_consistency,
    validate,
)

__all__ = [
    "DegradationParams",
    "DegradationRecord",
    "degrade",
    "params_from_config",
    "sweep",
    "SensorNoiseModel",
    "apply_photon_noise",
    "fit_noise_params",
    "photon_transfer_curve",
    "make_psf",
    "apply_psf_mosaic",
    "mtf_from_psf",
    "mtf50",
    "slanted_edge_mtf",
    "bin_bayer",
    "quantize",
    "upsample_to",
    "Tolerance",
    "ValidationError",
    "ValidationReport",
    "calibrate_from_stack",
    "self_consistency",
    "validate",
]
