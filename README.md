# SensorSim

**Quantifying the sensor–algorithm substitution rate in computational photography.**

A controlled testbench that degrades high-quality RAW captures to emulate weaker
camera hardware, applies a bank of restoration pipelines, and measures how much
of the lost information each pipeline recovers — across fidelity, perceptual and
machine-vision axes.

The design this implements is in [ARCHITECTURE.md](ARCHITECTURE.md). This file
is how to run it.

---

## Quick start

```bash
pip install -r requirements.txt

python -m sensorsim.cli run configs/experiments/pilot.yaml       # ~30 s
python -m sensorsim.cli analyze experiments/pilot/metrics.csv
python -m sensorsim.cli dashboard experiments/pilot              # http://127.0.0.1:5000
```

The pilot builds its own synthetic corpus on first run, so nothing needs
downloading to see the whole system work end to end.

With `make`:

```
make run analyze dashboard
make reproduce EXPERIMENT=pilot
```

---

## What the stages do

```
degrade  ->  restore  ->  evaluate  ->  analyze  ->  dashboard
```

Stages communicate only through files and a manifest; no stage imports another.
That is what makes runs resumable and results auditable.

| Stage | Module | Emits |
|-------|--------|-------|
| 1 Degrade | `sensorsim/degrader/` | degraded RAW + a `DegradationRecord` |
| 2 Restore | `sensorsim/restore/` | reconstructed RGB per pipeline, cached by content hash |
| 3 Evaluate | `sensorsim/evaluate/` | one tidy long table, `metrics.csv` |
| 4 Analyze | `sensorsim/analyze/` | recovery curves, knees, significance tests, figures |
| — Dashboard | `dashboard/` | a read-only view over the cached results |

### Commands

| Command | What it does |
|---------|--------------|
| `sensorsim corpus` | build the synthetic corpus |
| `sensorsim validate <config>` | run the calibration gate on its own |
| `sensorsim run <config>` | degrade, restore, evaluate; resumable |
| `sensorsim analyze <metrics.csv>` | curves, knees, significance, figures |
| `sensorsim reproduce <config>` | regenerate everything from scratch and diff it |
| `sensorsim pipelines` | list registered arms and what can actually run |
| `sensorsim dashboard <experiment>` | serve the dashboard (`--upload` adds the one write route, for enhancing uploads) |
| `sensorsim weights fetch\|verify` | download / check the pinned Real-ESRGAN and Zero-DCE checkpoints |
| `sensorsim train nc-dce` | train NC-DCE and its two ablations |
| `sensorsim enhance <image>` | run NAG on one image, add its figure to the paper, write reports to `temp/` |
| `sensorsim paper-nag` | regenerate the NAG tables, macros, paper section and method PDF |

Run these as `python -m sensorsim.cli <command>`, or as `sensorsim <command>`
after `pip install -e .`.

---

## The dashboard

```bash
python -m sensorsim.cli dashboard experiments/pilot            # http://127.0.0.1:5000
python -m sensorsim.cli dashboard experiments/pilot --upload   # also enables image upload
```

| Page | What it shows |
|------|---------------|
| Overview | the run at a glance: headline numbers, each arm's mean across the sweep, the worst-hardware leaderboard, what each severity emulates, the calibration gate, gaps and provenance |
| Explorer | one scene under one hardware condition: a synchronised wipe/zoom comparison of any two outputs, the ranking at that condition, the scene's sweep chart and every metric |
| Analysis | what `analyze` and `paper-nag` concluded: recovery ratios, knees, significance and planned tests, arm means, fidelity vs perception, training curves, the noise-model check and every figure |
| Enhance | upload an image and run NAG on it (with `--upload`), and browse every past run in `experiments/enhance/` |

Every run next to the one you name (any folder in `experiments/` with a
`dashboard.json`) is in the run switcher, and every view is a URL —
`#/explore?exp=enhance_study&image=nat_00&sev=4` reopens exactly that comparison.
Analysis tables are read from `paper/figures/<run>/`, or from `paper/figures/`
for the run whose arms match it. The front end is plain ES modules with no build
step and no CDN, so it runs offline; every route except the opt-in upload is a
GET over files earlier stages wrote.

---

## NAG: noise-aware guarded enhancement

NAG is a method developed on this testbench. It extends **Zero-DCE** (Guo et
al., CVPR 2020) and **Real-ESRGAN** (Wang et al., ICCVW 2021) with a per-pixel
noise map. On the testbench the map is exact, computed from the capture record.
On uploads it is estimated blindly. In both cases it is carried in closed form
through every stage:

1. **NC-DCE**: Zero-DCE's curve network with the noise map as a fourth input.
   It is trained zero-reference with a new loss that stops the curves
   amplifying noise in flat regions (`sensorsim/enhance/ncdce.py`,
   `sensorsim/train/nc_dce.py`).
2. **VP-denoise**: a guided filter whose strength at each pixel is the noise
   the curves left there (`sensorsim/enhance/vpdenoise.py`).
3. **G-ESRGAN**: Real-ESRGAN's added detail is scaled by a Wiener gate, so
   detail survives only where the measurement holds more than noise. A
   low-pass back-projection then keeps the output consistent with what was
   measured (`sensorsim/enhance/guard.py`).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or the CPU wheel
pip install python-docx

python -m sensorsim.cli weights fetch                     # pinned, hash-checked
python -m sensorsim.cli train nc-dce                      # ~13 min per variant on a laptop GPU
python -m sensorsim.cli run configs/experiments/enhance_study.yaml
python -m sensorsim.cli analyze experiments/enhance_study/metrics.csv --figures paper/figures/enhance_study
python -m sensorsim.cli paper-nag                         # tables, macros, paper blocks, temp/NAG_Method_Report.pdf
python -m sensorsim.cli enhance image.png --title "..."   # one image -> figure in the paper + reports in temp/
python -m sensorsim.cli dashboard experiments/pilot --upload   # the same from the browser, on the Enhance page
```

The paper is edited only between `%% AUTOGEN:<key>` markers, and only after a
timestamped backup. A rerun replaces those blocks rather than adding to them.
Every number in the generated prose is a macro from
`paper/overleaf/generated/proposed_numbers.tex`, and every sentence that states
a direction is built from the number it describes.

The Word version is rendered from the LaTeX, not written separately. Build the
PDF first (on Overleaf, or locally with any engine that keeps `main.aux` and
`main.bbl`), then convert, so section, figure and citation numbers match:

```bash
tectonic --keep-intermediates --outdir paper/build paper/overleaf/main.tex
python paper/make_docx.py                                 # -> paper/SensorSim_IEEE_Paper.docx
```

---

## Four things this implementation is strict about

**Noise scales with signal.** Photon counts are Poisson, so variance equals the
mean; read noise is a separate signal-independent Gaussian. A single uniform
Gaussian over the frame is the usual shortcut and it makes dark regions too
clean and bright regions too noisy — backwards for a low-light study.
`tests/test_noise.py` pins the photon transfer relationship
`var = mu/g + sigma_read^2` and fails if it drifts.

**A missing checkpoint is a gap, not a substitution.** The learned arms
(`nafnet`, `restormer`, `real_esrgan`, `zero_dce`, `learned_isp`) report
themselves unavailable without torch and pinned weights, and the runner writes
explicit `status: unavailable` rows into the metrics table. A row labelled
`nafnet` that was actually produced by a bilateral filter is a wrong number
wearing the right name, and it would survive review.

**The clean condition is measured, not assumed.** Every sweep contains the
identity condition at severity 0, because the recovery ratio is defined against
it. The Bayer demosaic ceiling differs per scene — a resolution chart caps ~10 dB
below a natural scene — so a ratio measured against an assumed ceiling would read
that difference as recovery.

**"No knee" stays available as a finding.** `analyze/knee.py` locates the knee
three independent ways and reports the spread between them. A knee the three
methods disagree about is reported as *not identifiable* rather than averaged
into a number.

---

## Two deliberate deviations from ARCHITECTURE.md

**Operation order.** The section 2 diagram lists `photon scaling → shot noise →
read noise → PSF blur → pixel binning → quantization`. The default here is the
physical light path — blur and binning happen in the optics and the pixel,
*before* photons are counted:

```
PSF blur -> binning -> photon scaling -> shot noise -> read noise -> quantisation
```

Convolving after the noise is added low-pass filters the noise itself, so the
degraded frame comes out cleaner than the hardware it emulates and every
recovery ratio measured against it is inflated. The diagram's order is available
as `order: listed` in a degradation config and every `DegradationRecord` carries
the order it used, so a run made either way is self-describing.
(`tests/test_degrader.py::test_ordering_matters_and_is_recorded` measures the
difference.)

**Dashboard stack.** Section 3 says "Flask + React, or Streamlit if time is
short". This is Flask plus a zero-build vanilla-JS front end. React would need a
CDN script tag or an npm build; the phase 6 gate is "demo runs offline from
cached results", and a CDN dependency fails that gate the first time a
conference room has no wifi.

---

## Corpus

The synthetic corpus generates three scene kinds, each chosen so a different
metric family has something to say:

| Kind | Content | Feeds |
|------|---------|-------|
| `resolution` | slanted edges, Siemens star, frequency sweep, step wedge | slanted-edge MTF, the calibration gate |
| `text` | text from headline to 8pt, dense paragraph | OCR accuracy, visible hallucination |
| `natural` | gradients, coloured objects, distant repeating detail, fine texture | fidelity, perceptual, keypoint matching |

Synthetic scenes are a scaffold, not a finding — they exist so every stage,
metric and figure can be exercised before the real datasets land. Numbers in the
paper come from the section 5 corpus (SID, Zurich, FiveK, HDR+, own captures).

To use a real corpus, point an experiment config at it:

```yaml
corpus:
  kind: directory
  raw_dir: data/sid/clean          # linear .npy frames, or camera RAW with rawpy
  reference_dir: data/sid/reference
```

---

## The calibration gate

Section 3 makes calibration a precondition, and the runner enforces it. Three
modes, set under `validation:` in an experiment config:

| Mode | What it checks | Safe to publish from |
|------|----------------|---------------------|
| `hardware` | synthetic vs measured photon transfer curve and MTF50, against stated tolerances | yes |
| `self` | the noise generator reproduces its own constants | no — catches code bugs only |
| `skip` | nothing | no — recorded as a warning in the manifest |

`self` is the pilot's default so the demo runs anywhere. It is not the section 3
gate and the manifest says so in as many words. For `hardware`, put stacks of at
least eight tripod captures of one static scene in the calibration directory:

```
data/calibration/
  measured_stack.npy     (N, H, W) real captures, normalised to [0, 1]
  synthetic_stack.npy    (N, H, W) the model's attempt at the same scene
  measured_edge.npy      a slanted-edge patch, for the MTF check
  synthetic_edge.npy
```

---

## Adding a restoration arm

A pipeline is added by a config entry, not a code change. To add a new *kind*,
implement the one interface and register it:

```python
from sensorsim.restore.base import Pipeline, register

@register("my_pipeline")
class MyPipeline(Pipeline):
    def restore(self, degraded_raw, meta):
        ...  # return uint8 RGB at meta.target_shape
```

Then name it in a bank config:

```yaml
pipelines:
  - {name: my_pipeline, kind: my_pipeline, params: {strength: 0.5}}
```

Outputs are cached by a content hash of the input frame, the pipeline identity,
its parameters and its weights checksum, so a changed checkpoint invalidates its
own cache entries.

To activate a learned arm, set `weights` to a pinned checkpoint and `arch` to a
`module:factory` that returns the matching `nn.Module`. Until then it reports
unavailable — by design.

---

## Reproducibility

Every run writes `manifest.json`: config hash, git commit (flagged if the tree
was dirty), package versions, weights checksums, seeds and hardware.
Randomness is seeded per `(image, condition, seed)`, so the same condition on
the same image always draws the same noise and different conditions never share
a realisation by accident.

```bash
python -m sensorsim.cli reproduce pilot
```

clears the cache, reruns from scratch, and compares every cell against the
previous table — exiting non-zero if any value moved.

---

## Tests

```bash
python -m pytest -q
```

186 tests, about 15 seconds. The suite is weighted toward the places where a
silent error would corrupt results rather than crash a run: the noise model, the
Bayer pattern mapping, the recovery-ratio denominator, and the rule that an
unavailable pipeline never produces a number.

---

## Layout

```
configs/          sweep, pipeline bank and experiment definitions
sensorsim/
  degrader/       forward sensor model + the calibration gate
  restore/        pipeline interface, registry, cache; classical/ and learned/
  evaluate/       fidelity, perceptual, machine-task metrics
  analyze/        curves, knee detection, significance, figures
  io/             RAW representation, run manifests
  data/           synthetic corpus
  config.py       config loading and sweep construction
  experiment.py   the run orchestrator - the only place the stages meet
dashboard/        Flask backend + zero-build front end
experiments/      run outputs (gitignored except manifests and metrics tables)
paper/            LaTeX source; figures/ is regenerated, never hand-made
tests/
```

---

## Optional dependencies

Everything above runs on the core install. These unlock specific arms:

| Extra | Unlocks |
|-------|---------|
| `pip install -e ".[raw]"` | camera RAW decode (`.dng`, `.arw`, `.cr2`, …) |
| `pip install -e ".[learned]"` | the learned pipeline bank, LPIPS |
| `pip install -e ".[ocr]"` | OCR character accuracy (also needs the tesseract binary) |
| `pip install -e ".[detect]"` | detection mAP |

`sensorsim pipelines` lists what can currently run and why the rest cannot.
