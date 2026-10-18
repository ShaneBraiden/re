"""NAG - noise-aware guarded enhancement.

Two published methods form the base, and each fails in a way this testbench
measures:

* **Zero-DCE** (Guo et al., CVPR 2020) brightens with learned per-pixel curves
  and amplifies sensor noise in dark, flat regions.
* **Real-ESRGAN** (Wang et al., ICCVW 2021) restores resolution with a
  generator that also invents detail - substitution, in the paper's terms.

NAG makes one change to the pair: it computes a per-pixel noise map, carries it
through every stage, and lets each stage use it.

1. :mod:`~sensorsim.enhance.noise` - an exact noise std map from the capture
   metadata (or a blind estimate for uploads), carried through white balance,
   the tone curve and the learned curves in closed form.
2. :mod:`~sensorsim.enhance.ncdce` - NC-DCE: Zero-DCE with the noise map as a
   fourth input and a loss that bounds noise amplification in flat regions.
3. :mod:`~sensorsim.enhance.vpdenoise` - a guided filter whose ``eps`` is the
   carried noise variance, so it smooths only where noise was amplified.
4. :mod:`~sensorsim.enhance.guard` - G-ESRGAN: the generator's added detail
   scaled by a Wiener gate built from the noise map, then low-pass
   back-projection onto the measurement.

The restoration-bank arms that use these live in
:mod:`sensorsim.restore.learned.nag`. The upload flow, paper update and
reports are in :mod:`~sensorsim.enhance.pipeline`,
:mod:`~sensorsim.enhance.latex` and :mod:`~sensorsim.enhance.report`.
"""
