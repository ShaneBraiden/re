"""Poisson-Gaussian sensor noise.

The one thing this file exists to get right: **noise scales with signal**.
A photon count is Poisson, so its variance equals its mean; read noise is a
signal-independent Gaussian added afterwards by the readout electronics. Adding
a single uniform Gaussian to the whole frame - the usual shortcut - makes dark
regions too clean and bright regions too noisy, which is exactly backwards for a
low-light study and invalidates every recovery number downstream.

The photon transfer relationship this produces is the one a real sensor obeys::

    var(DN) = mu(DN) / g + sigma_read^2

with ``g`` the conversion gain in electrons per DN. :mod:`sensorsim.degrader.validate`
fits that line on synthetic and measured frames and refuses to let a run proceed
if they disagree.

Reference: Foi et al., *Practical Poissonian-Gaussian Noise Modeling and Fitting
for Single-Image Raw-Data*, IEEE TIP 2008.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorNoiseModel:
    """Physical constants of the emulated sensor.

    Attributes
    ----------
    full_well:
        Electrons held by a saturated pixel at binning 1.
    black_level, white_level:
        DN endpoints of the ADC range the signal is expressed against.
    sigma_read_dn:
        Read noise standard deviation in DN. Signal-independent by definition.
    dark_current_e:
        Thermally generated electrons per pixel for this exposure. Poisson like
        photons, and the reason a long dark exposure is not actually black.
    prnu:
        Photo-response non-uniformity as a fraction (0.005 = 0.5%). A fixed
        per-pixel gain error, so it is multiplicative and does not average out
        across frames the way shot noise does.
    """

    full_well: float = 15000.0
    black_level: float = 512.0
    white_level: float = 16383.0
    sigma_read_dn: float = 2.0
    dark_current_e: float = 2.0
    prnu: float = 0.005

    @property
    def gain_e_per_dn(self) -> float:
        """Electrons per DN: how many photoelectrons one ADC step represents."""
        span = self.white_level - self.black_level
        if span <= 0:
            raise ValueError("white_level must exceed black_level")
        return self.full_well / span

    @property
    def sigma_read_e(self) -> float:
        """Read noise expressed in electrons - the number a datasheet quotes."""
        return self.sigma_read_dn * self.gain_e_per_dn


def apply_photon_noise(
    signal: np.ndarray,
    model: SensorNoiseModel,
    photon_scale: float = 1.0,
    well_multiplier: float = 1.0,
    rng: np.random.Generator | None = None,
    prnu_field: np.ndarray | None = None,
    floor: float | None = None,
) -> np.ndarray:
    """Take normalised linear signal to noisy normalised linear signal.

    Parameters
    ----------
    signal:
        Normalised linear radiance in [0, 1]; 1.0 is full well at
        ``photon_scale = 1``.
    model:
        Sensor constants.
    photon_scale:
        ``alpha`` from the degradation grid. Fewer photons reach the pixel -
        smaller aperture, smaller sensor area, shorter exposure. The output stays
        *dark*: the deficit is real and a pipeline has to earn it back with gain,
        which is the entire point of the study.
    well_multiplier:
        Effective full-well multiplier, ``b^2`` after binning ``b x b`` pixels.
        Binned pixels hold more charge, which is why binning buys SNR.
    prnu_field:
        Optional pre-drawn per-pixel gain field. Pass the same field across a
        sweep so PRNU stays *fixed pattern* rather than re-rolling per condition.
    floor:
        Lowest representable normalised value. Defaults to
        ``-black_level / span``, the negative headroom a real sensor's black
        level pedestal provides. **Do not raise this to 0.** Clipping read noise
        at zero rectifies its lower half, which biases the mean upward and
        shrinks the measured sigma to ~0.58 of its true value - and it does so
        exactly where the study lives, in the darkest frames.

    Returns
    -------
    np.ndarray
        Noisy signal, normalised to the same scale as the input (still dark by
        ``photon_scale``), clipped at the saturation ceiling.
    """
    rng = rng or np.random.default_rng()
    signal = np.clip(np.asarray(signal, dtype=np.float64), 0.0, None)

    well = model.full_well * well_multiplier

    # 1. Signal to photoelectrons. This is where the exposure deficit is applied.
    electrons = signal * well * photon_scale

    # 2. Fixed-pattern gain error, before the shot noise it modulates.
    if model.prnu > 0:
        if prnu_field is None:
            prnu_field = rng.normal(1.0, model.prnu, size=electrons.shape)
        electrons = electrons * prnu_field

    # 3. Dark current: thermal electrons, Poisson like the photons.
    electrons = electrons + model.dark_current_e * well_multiplier

    # 4. Shot noise. Poisson variance == Poisson mean; this is the whole model.
    #    numpy Poisson overflows above ~2^63, and the Gaussian limit is
    #    indistinguishable by then, so switch over well before that.
    noisy_e = np.empty_like(electrons)
    large = electrons > 1e6
    if np.any(~large):
        noisy_e[~large] = rng.poisson(electrons[~large]).astype(np.float64)
    if np.any(large):
        noisy_e[large] = rng.normal(electrons[large], np.sqrt(electrons[large]))

    # 5. Saturation clips in the charge domain, before readout - a full well
    #    cannot hold more charge no matter how the ADC is configured.
    noisy_e = np.clip(noisy_e, 0.0, well)

    # 6. Readout: electrons to DN, then add signal-independent read noise.
    dn = noisy_e / model.gain_e_per_dn
    if model.sigma_read_dn > 0:
        dn = dn + rng.normal(0.0, model.sigma_read_dn, size=dn.shape)

    # 7. Back to normalised units. The DN span is the unbinned one; the extra
    #    well capacity from binning is carried by well_multiplier.
    span = (model.white_level - model.black_level) * well_multiplier
    if floor is None:
        floor = -model.black_level / max(model.white_level - model.black_level, 1.0)
    return np.clip(dn / span, floor, 1.0).astype(np.float32)


def photon_transfer_curve(
    frames: np.ndarray,
    n_bins: int = 24,
    clip_quantile: float = 0.999,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure mean-vs-variance from a stack of repeated captures.

    Parameters
    ----------
    frames:
        Shape (N, H, W), N >= 2 captures of an identical scene. Temporal variance
        per pixel isolates the random noise and leaves fixed-pattern effects out,
        which is what makes this a clean estimator.

    Returns
    -------
    (mean, variance, count) per intensity bin.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError(f"need a stack of >= 2 frames, got shape {frames.shape}")

    mu = frames.mean(axis=0).ravel()
    var = frames.var(axis=0, ddof=1).ravel()

    hi = np.quantile(mu, clip_quantile)
    keep = mu <= hi  # drop saturated pixels: variance collapses there
    mu, var = mu[keep], var[keep]

    edges = np.linspace(mu.min(), mu.max(), n_bins + 1)
    idx = np.clip(np.digitize(mu, edges) - 1, 0, n_bins - 1)

    means, variances, counts = [], [], []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n < 32:  # too few pixels for a trustworthy variance
            continue
        means.append(float(mu[sel].mean()))
        variances.append(float(var[sel].mean()))
        counts.append(n)
    return np.array(means), np.array(variances), np.array(counts)


def fit_noise_params(mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    """Fit ``var = a*mu + b`` and report the sensor constants it implies.

    ``a`` is the inverse conversion gain (DN per electron) and ``b`` is the read
    noise variance. Both are in the normalised units the curve was measured in.
    """
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if mean.size < 3:
        raise ValueError("need at least 3 intensity bins to fit a photon transfer curve")

    a, b = np.polyfit(mean, variance, 1)
    predicted = a * mean + b
    ss_res = float(np.sum((variance - predicted) ** 2))
    ss_tot = float(np.sum((variance - variance.mean()) ** 2))
    return {
        "slope": float(a),
        "intercept": float(b),
        "read_noise_sigma": float(np.sqrt(max(b, 0.0))),
        "inverse_gain": float(a),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def estimate_read_noise(dark_frames: np.ndarray) -> float:
    """Read noise sigma from dark frames, in whatever units they carry."""
    dark_frames = np.asarray(dark_frames, dtype=np.float64)
    if dark_frames.ndim == 3 and dark_frames.shape[0] >= 2:
        return float(np.sqrt(dark_frames.var(axis=0, ddof=1).mean()))
    return float(dark_frames.std())
