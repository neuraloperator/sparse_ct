# Convenience targets for the CTO sparse-view CT project.
# Run `make help` to list targets. Requires the python env from `make setup`.

PYTHON ?= python
export PYTHONPATH := $(CURDIR)

TORCH_RADON_DIR ?= torch-radon
TORCH_RADON_REPO ?= https://github.com/matteo-ronchetti/torch-radon.git

.DEFAULT_GOAL := help
.PHONY: help setup torch-radon weights data-aapm data-kits train-aapm train-kits test-aapm test-kits

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install Python deps (then run `make torch-radon` separately)
	$(PYTHON) -m pip install -r requirements.txt
	pip install --no-build-isolation torch-harmonics

torch-radon:  ## Clone, patch, and build torch-radon (needs a CUDA build env: nvcc + toolkit)
	@test -d $(TORCH_RADON_DIR) || git clone $(TORCH_RADON_REPO) $(TORCH_RADON_DIR)
	cd $(TORCH_RADON_DIR) && patch -p1 --forward < $(CURDIR)/torch-radon_fix/torch-radon_fix.patch || true
	cd $(TORCH_RADON_DIR) && $(PYTHON) setup.py install

weights:  ## Download pretrained weights from Hugging Face into ./weights
	$(PYTHON) download_weights.py

data-aapm:  ## Preprocess AAPM (pass AAPM_RAW_DIR=/path/to/full_1mm) -> data/aapm
	$(if $(AAPM_RAW_DIR),AAPM_RAW_DIR="$(AAPM_RAW_DIR)" )$(PYTHON) src/data/preprocess_aapm.py

data-kits:  ## Preprocess C4KC-KiTS (pass KITS_RAW_DIR=/path/to/C4KC-KiTS) -> data/kits
	$(if $(KITS_RAW_DIR),KITS_RAW_DIR="$(KITS_RAW_DIR)" )$(PYTHON) src/data/preprocess_kits.py

train-aapm:  ## Train on AAPM
	$(PYTHON) scripts/train.py -c configs/aapm.yaml

train-kits:  ## Train on C4KC-KiTS
	$(PYTHON) scripts/train.py -c configs/kits.yaml

test-aapm:  ## Test pretrained AAPM weights (no wandb); metrics -> results/
	$(PYTHON) scripts/test.py -c configs/aapm.yaml --fix init_exp_dir weights/aapm --no-wandb

test-kits:  ## Test pretrained KiTS weights (no wandb); metrics -> results/
	$(PYTHON) scripts/test.py -c configs/kits.yaml --fix init_exp_dir weights/kits --no-wandb
