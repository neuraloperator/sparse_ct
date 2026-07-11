# Convenience targets for the CTO sparse-view CT project.
# Run `make help` to list targets. Requires the python env from `make setup`.

PYTHON ?= python
export PYTHONPATH := $(CURDIR)

.DEFAULT_GOAL := help
.PHONY: help setup weights data-aapm data-kits train-aapm train-kits test-aapm test-kits

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install Python deps (torch-radon still needs the manual step in README)
	$(PYTHON) -m pip install -r requirements.txt
	pip install --no-build-isolation torch-harmonics

weights:  ## Download pretrained weights from Hugging Face into ./weights
	$(PYTHON) download_weights.py

data-aapm:  ## Preprocess AAPM (set AAPM_RAW_DIR to your raw download) -> data/aapm
	$(PYTHON) src/data/preprocess_aapm.py

data-kits:  ## Preprocess C4KC-KiTS (set KITS_RAW_DIR to your raw download) -> data/kits
	$(PYTHON) src/data/preprocess_kits.py

train-aapm:  ## Train on AAPM
	$(PYTHON) scripts/train.py -c configs/aapm.yaml

train-kits:  ## Train on C4KC-KiTS
	$(PYTHON) scripts/train.py -c configs/kits.yaml

test-aapm:  ## Test pretrained AAPM weights (no wandb); metrics -> results/
	$(PYTHON) scripts/test.py -c configs/aapm.yaml --fix init_exp_dir weights/aapm --no-wandb

test-kits:  ## Test pretrained KiTS weights (no wandb); metrics -> results/
	$(PYTHON) scripts/test.py -c configs/kits.yaml --fix init_exp_dir weights/kits --no-wandb
