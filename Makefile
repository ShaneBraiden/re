# SensorSim
#
# Every target delegates to the `sensorsim` CLI, so there is one implementation
# and `make` stays a convenience rather than a second source of truth. On a
# machine without make, run the python commands directly - they are printed in
# the README and shown in `sensorsim --help`.

PYTHON      ?= python
EXPERIMENT  ?= pilot
CONFIG      ?= configs/experiments/$(EXPERIMENT).yaml
METRICS      = experiments/$(EXPERIMENT)/metrics.csv
METRIC      ?= psnr
BASELINE    ?= classical_isp
PORT        ?= 5000

.PHONY: help install install-dev corpus validate run analyze figures dashboard test lint reproduce clean clean-all \
        weights train study paper-nag enhance dashboard-upload frontend frontend-dev

help:
	@echo "SensorSim targets"
	@echo "  make install                 core dependencies"
	@echo "  make install-dev             core + test tooling, editable install"
	@echo "  make corpus                  build the synthetic corpus"
	@echo "  make validate                run the calibration gate alone"
	@echo "  make run                     degrade -> restore -> evaluate"
	@echo "  make analyze                 curves, knees, significance, figures"
	@echo "  make dashboard               serve the read-only dashboard"
	@echo "  make frontend                build the React dashboard (needs Node 20+)"
	@echo "  make frontend-dev            live-reloading frontend on :5173 (run make dashboard too)"
	@echo "  make test                    the test suite"
	@echo "  make reproduce EXPERIMENT=id regenerate every number and figure"
	@echo ""
	@echo "NAG (noise-aware guarded enhancement)"
	@echo "  make weights                 fetch + verify the pinned checkpoints"
	@echo "  make train                   train NC-DCE and its ablations"
	@echo "  make study                   run + analyze the NAG study"
	@echo "  make paper-nag               tables, macros, paper section, method PDF"
	@echo "  make enhance IMAGE=path      enhance one image, update paper, write reports"
	@echo "  make dashboard-upload        dashboard with the upload page enabled"
	@echo ""
	@echo "  EXPERIMENT=$(EXPERIMENT)  METRIC=$(METRIC)  BASELINE=$(BASELINE)"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

corpus:
	$(PYTHON) -m sensorsim.cli corpus --out data/synthetic --per-kind 2 --size 512

validate:
	$(PYTHON) -m sensorsim.cli validate $(CONFIG)

run:
	$(PYTHON) -m sensorsim.cli run $(CONFIG)

analyze figures:
	$(PYTHON) -m sensorsim.cli analyze $(METRICS) --metric $(METRIC) --baseline $(BASELINE)

dashboard:
	$(PYTHON) -m sensorsim.cli dashboard experiments/$(EXPERIMENT) --port $(PORT)

dashboard-upload:
	$(PYTHON) -m sensorsim.cli dashboard experiments/$(EXPERIMENT) --port $(PORT) --upload

# The React frontend builds into dashboard/web, which the dashboard serves at /.
frontend:
	cd frontend && npm ci && npm run build

frontend-dev:
	cd frontend && npm run dev

weights:
	$(PYTHON) -m sensorsim.cli weights fetch
	$(PYTHON) -m sensorsim.cli weights verify

train:
	$(PYTHON) -m sensorsim.cli train nc-dce

study:
	$(PYTHON) -m sensorsim.cli run configs/experiments/enhance_study.yaml
	$(PYTHON) -m sensorsim.cli analyze experiments/enhance_study/metrics.csv --figures paper/figures/enhance_study

paper-nag:
	$(PYTHON) -m sensorsim.cli paper-nag

IMAGE ?= image.png
enhance:
	$(PYTHON) -m sensorsim.cli enhance $(IMAGE)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall -q sensorsim dashboard tests

# Section 7: regenerates every number and figure in the paper. Clears the
# restoration cache first - a cache hit would only prove the cache still works -
# and fails loudly if any value differs from the previous run.
reproduce:
	$(PYTHON) -m sensorsim.cli reproduce $(CONFIG) --metric $(METRIC) --baseline $(BASELINE)

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('experiments').glob('*/cache')]"

clean-all: clean
	$(PYTHON) -c "import shutil; shutil.rmtree('experiments/$(EXPERIMENT)', ignore_errors=True)"
	$(PYTHON) -c "import shutil; shutil.rmtree('paper/figures', ignore_errors=True)"
