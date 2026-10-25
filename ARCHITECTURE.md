# SensorSim — Project Architecture

**Quantifying the Sensor–Algorithm Substitution Rate in Computational Photography**

A controlled testbench that degrades high-quality RAW captures to emulate weaker camera hardware, applies a bank of restoration pipelines, and measures how much of the lost information each pipeline recovers — across fidelity, perceptual, and machine-vision axes.

---

## 1. Research Framing

**Question.** At a fixed scene, how much of a hardware deficit can algorithms recover, and where does that recovery saturate?

**Hypotheses.**

| ID | Hypothesis | Test |
|----|-----------|------|
| H1 | Algorithmic recovery of a hardware deficit follows a saturating curve with an identifiable knee | Sweep degradation severity, fit recovery curve, locate knee |
| H2 | Beyond the knee, perceptual scores continue to improve while fidelity scores plateau or decline | Track fidelity and no-reference metrics on the same outputs |
| H3 | Heavy AI processing that improves human-perceived quality reduces machine-vision task accuracy | Run detection / OCR / keypoint matching on each output tier |

**Contribution.** A reproducible measurement of the substitution rate between sensor capability and computational processing, plus the saturation point beyond which gains are perceptual rather than informational.

---

## 2. System Overview

```
                    ┌──────────────────────────┐
                    │      Clean RAW Corpus     │
                    │  (SID / Zurich / FiveK /  │
                    │     own device captures)  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────┐
        │  STAGE 1 — DEGRADER                        │
        │  Forward sensor model                      │
        │  photon scaling → shot noise → read noise  │
        │  → PSF blur → pixel binning → quantization │
        │  Emits: degraded RAW + degradation record  │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │  STAGE 2 — RESTORER                        │
        │  Pluggable pipeline bank                   │
        │  classical ISP │ Zero-DCE │ NAFNet │       │
        │  Restormer │ Real-ESRGAN │ learned ISP     │
        │  Emits: reconstructed RGB per pipeline     │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │  STAGE 3 — EVALUATOR                       │
        │  Fidelity   │ Perceptual │ Machine-task    │
        │  PSNR SSIM  │ LPIPS NIQE │ mAP OCR SIFT    │
        │  Emits: tidy metrics table (one row per    │
        │  image × degradation × pipeline × metric)  │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │  STAGE 4 — ANALYZER                        │
        │  Recovery curves, knee detection,          │
        │  divergence analysis, significance tests   │
        │  Emits: figures + result tables for paper  │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Dashboard      │
                    │  (demo surface)  │
                    └──────────────────┘
```

Stages communicate only through files and a manifest. No stage imports another. This is what makes runs resumable and results auditable.

---

## 3. Stage Specifications

### Stage 1 — Degrader

Implements a forward sensor model applied to linear RAW, **before** any demosaicing.

| Parameter | Symbol | Range | Emulates |
|-----------|--------|-------|----------|
| Photon scale | `alpha` | 1.0 → 0.02 | Smaller aperture / sensor area / shorter exposure |
| Read noise sigma | `sigma_r` | 0 → 12 DN | Cheaper readout electronics |
| PSF radius | `r_psf` | 0 → 3.0 px | Diffraction and inferior optics |
| Binning factor | `b` | 1, 2, 3 | Smaller effective pixel pitch |
| Bit depth | `d` | 14 → 8 | Cheaper ADC |

Signal-dependent noise is Poisson–Gaussian: scale to photon count, sample Poisson, add Gaussian read noise, requantize. Applying Gaussian noise uniformly is the single most common mistake here and it invalidates the entire study — noise must scale with signal.

**Validation gate.** Before any experiments run, degraded output must be validated against real captures from a low-end sensor: compare noise variance vs. intensity curves and MTF slope. If synthetic and measured curves disagree beyond tolerance, the model is recalibrated. Document this in the paper's method section — it is the difference between a study and a simulation.

**Interface.**
```
degrade(raw_linear, params) -> (degraded_raw, DegradationRecord)
```

### Stage 2 — Restorer

Every pipeline satisfies one interface:
```
class Pipeline:
    name: str
    def restore(self, degraded_raw, meta) -> rgb_uint8
```

Registered via a config entry, not code changes. Bank:

- `classical_isp` — black level, demosaic, white balance, gamma, mild denoise. The control condition.
- `zero_dce` — low-light curve estimation
- `nafnet` / `restormer` — learned denoising and restoration
- `real_esrgan` — generative detail synthesis (the hallucination-prone arm, deliberately included)
- `learned_isp` — end-to-end RAW→RGB trained on Zurich

Outputs are cached by content hash of (input, pipeline, weights version) so sweeps are re-runnable without recompute.

### Stage 3 — Evaluator

Three metric families, computed on identical outputs.

**Fidelity (reference-based).** PSNR, SSIM — is the output *true* to the scene.

**Perceptual.** LPIPS (reference), NIQE and CLIP-IQA (no-reference) — does the output *look* good.

**Machine-task.** The unbiased judge:
- Object detection mAP (YOLO) on annotated scenes
- OCR character accuracy (Tesseract) on scenes containing text
- Keypoint match count and inlier ratio (SIFT / LoFTR) against the clean reference

Machine-task metrics are the paper's strongest evidence for H3, because they cannot be gamed by aesthetic tuning.

Output is a single tidy table:
```
image_id, degradation_params_hash, pipeline, metric_family, metric_name, value
```
Long format, not wide. Every downstream plot is a groupby on this.

### Stage 4 — Analyzer

- Recovery ratio per condition: `R = (M_restored - M_degraded) / (M_clean - M_degraded)`
- Fit saturating curve across the degradation axis; locate knee via maximum-curvature or piecewise-linear breakpoint
- Divergence plot: fidelity vs. perceptual on shared x-axis
- Paired significance tests across pipelines, with multiple-comparison correction
- Qualitative crop grid at fixed coordinates — text, faces, distant repeating detail

### Dashboard

Thin read-only layer over cached results. Severity slider, pipeline selector, synchronized image comparison, live metric readout. Its purpose is demonstration and figure generation, not computation. Flask + React, or Streamlit if time is short.

One opt-in exception: `sensorsim dashboard --upload` registers `dashboard/upload.py`, which adds a single POST route (`/api/enhance`) for running NAG on an uploaded image. Without the flag the app has no write routes, and `tests/test_dashboard.py` still enforces that.

### NAG — noise-aware guarded enhancement (`sensorsim/enhance/`)

A method built on the testbench rather than only measured by it. It extends Zero-DCE and Real-ESRGAN with one shared ingredient: a per-pixel noise map. On the testbench the map is exact, from the capture record and the Poisson–Gaussian model; on uploads it is a blind estimate. It is carried in closed form through white balance, demosaicing, the tone curve and the learned curves.

| Stage | What it does with the noise map |
|---|---|
| NC-DCE (`ncdce.py`, trained by `sensorsim/train/nc_dce.py`) | 4th input channel; loss bounds `|curve slope| * sigma` in flat regions |
| VP-denoise (`vpdenoise.py`) | guided filter with per-pixel `eps = (1.5 sigma)^2` |
| G-ESRGAN (`guard.py`) | Wiener gate on generated detail, `gamma = (E_y - 20 sigma^2) / E_y`; low-pass back-projection through the recorded PSF and binning |

The restoration-bank arms `nc_dce`, `g_esrgan` and `nag` are in `sensorsim/restore/learned/nag.py`. The study is `configs/experiments/enhance_study.yaml`, a separate run, so the pilot's numbers are untouched. `sensorsim paper-nag` writes the paper section between `%% AUTOGEN:nag-*` markers, with every number as a macro.

---

## 4. Repository Layout

```
sensorsim/
├── configs/
│   ├── degradation/          # sweep definitions
│   ├── pipelines/            # model + weights specs
│   └── experiments/          # full run manifests
├── sensorsim/
│   ├── degrader/
│   │   ├── noise.py          # Poisson-Gaussian model
│   │   ├── optics.py         # PSF, diffraction
│   │   ├── sampling.py       # binning, quantization
│   │   └── validate.py       # calibration against real sensors
│   ├── restore/
│   │   ├── base.py           # Pipeline interface
│   │   ├── classical.py
│   │   └── learned/
│   ├── evaluate/
│   │   ├── fidelity.py
│   │   ├── perceptual.py
│   │   └── machine_task.py
│   ├── analyze/
│   │   ├── curves.py         # recovery fitting
│   │   ├── knee.py           # saturation detection
│   │   └── figures.py        # paper-ready plots
│   └── io/
│       ├── raw.py            # rawpy loaders
│       └── manifest.py       # run tracking
├── dashboard/                # Flask API; serves the built frontend
├── frontend/                 # React + TypeScript dashboard (Vite), built into dashboard/web
├── experiments/              # run outputs, gitignored except manifests
├── paper/
│   ├── main.tex
│   ├── figures/              # symlinked from analyze output
│   └── refs.bib
├── notebooks/                # exploration only, nothing load-bearing
└── tests/
```

Rule: no figure in `paper/figures/` is ever hand-made. Every one is regenerated by `analyze/figures.py` from the metrics table. When a reviewer asks what changed, you rerun.

---

## 5. Data

| Source | Role |
|--------|------|
| SID (See-in-the-Dark) | Low-light RAW pairs, primary noise study |
| Zurich RAW-to-RGB | Learned ISP training and evaluation |
| MIT-Adobe FiveK | Diverse scene content, tone reference |
| HDR+ burst dataset | Burst-merge comparison arm |
| Own device RAW+JPEG pairs | Degrader calibration and device-specific section |

Own captures should be tripod-mounted, fixed scene, bracketed — including a text chart, a resolution chart, and a natural scene. These do double duty: calibration input and qualitative figure material.

---

## 6. Execution Sequence

| Phase | Work | Gate before moving on |
|-------|------|----------------------|
| 1 | Degrader + validation | Synthetic noise curves match measured within tolerance |
| 2 | Classical ISP baseline + fidelity metrics | End-to-end run produces a metrics table |
| 3 | Learned pipeline bank | All pipelines run under one interface, results cached |
| 4 | Perceptual + machine-task metrics | Full tidy table across the sweep |
| 5 | Analysis, knee detection, figures | Recovery curve reproduces from scratch |
| 6 | Dashboard | Demo runs offline from cached results |
| 7 | Paper writing | — |

Keep a dated results log from phase 1. The paper's experiments section is easier to write from a log than from memory, and negative results recorded early often become the discussion section.

---

## 7. Reproducibility Requirements

- Every run writes a manifest: config hash, git commit, weights checksums, seeds, hardware
- All randomness seeded and recorded
- Model weights pinned by checksum, never "latest"
- `make reproduce EXPERIMENT=<id>` regenerates every number and figure in the paper
- Metrics table committed as a release artifact alongside the paper

---

## 8. Stated Limitations

Name these in the paper before a reviewer does:

- Synthetic degradation approximates but does not equal a real sensor; lens aberrations, colour filter array response, and in-sensor processing are not modelled
- Sensor model is calibrated against a small number of devices
- No human perceptual study; perceptual claims rest on proxy metrics
- Machine-task metrics reflect the specific detectors used and may not generalise across architectures
- The knee is measured on this degradation parameterisation, not on the full space of real hardware tradeoffs

---

## 9. Key References

- Blau & Michaeli, *The Perception-Distortion Tradeoff*, CVPR 2018 — theoretical backbone for H2
- Chen et al., *Learning to See in the Dark*, CVPR 2018 — SID dataset and low-light RAW baseline
- Hasinoff et al., *Burst Photography for High Dynamic Range and Low-Light Imaging*, SIGGRAPH Asia 2016 — HDR+ pipeline
- Ignatov et al., *Replacing Mobile Camera ISP with a Single Deep Learning Model* — Zurich RAW-to-RGB
- Foi et al., *Practical Poissonian-Gaussian Noise Modeling* — noise model grounding
- Chen et al., *Simple Baselines for Image Restoration* (NAFNet), ECCV 2022
