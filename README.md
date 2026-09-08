# Resolution-Agnostic Neural Operators for Multi-Rate Sparse-View CT<br><sub>ECCV 2026</sub>

![Teaser image](./docs/teaser.png)

**Resolution-Agnostic Neural Operators for Multi-Rate Sparse-View CT**<br>
ECCV 2026<br>
Aujasvit Datta*, Jiayun Wang*, Asad Aali, Anima Anandkumar
<br>arXiv: https://arxiv.org/abs/2512.12236<br>


![CTO architecture](./docs/architecture.png)

All common tasks are wrapped as `make` targets. Run `make help` to list them. Every target is a wrapper around a `python ...` command. 

## 1. Configuration

Before running anything, review `configs/base.yaml` and update it for your setup. Common changes:
- `use_wandb`: set to `false` to disable Weights & Biases logging for both train and test.
- `logger.project` and `logger.entity`: your Weights & Biases project and account. Leave `entity` as `null` to use your logged-in default, or set `WANDB_ENTITY`.
- `trainer.num_devices`: the number of GPUs to use.
- `trainer.max_epochs` and `trainer.lr`: training hyperparameters.

Dataset-specific settings (model, sampler, batch size, data paths) are in `configs/aapm.yaml` and `configs/kits.yaml`.

## 2. Install

```
conda create --name cto_env python=3.11
conda activate cto_env
make setup          # install python dependencies
make torch-radon    # build torch-radon (needs a CUDA build environment: nvcc + toolkit)
```
Tested with Python 3.11 on Linux. `make torch-radon` clones torch-radon, applies the required patch from `torch-radon_fix/`, and builds it.

## 3. Data

The Low-dose CT **AAPM** dataset is substantially smaller than the **C4KC-KiTS** dataset, hence easier to work with.

- **AAPM**: download from https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144226105715, then unzip `FD_1mm.zip` to `full_1mm/`.
- **C4KC-KiTS**: download from https://www.cancerimagingarchive.net/collection/c4kc-kits/.

To preprocess the data, run:
```
make data-aapm AAPM_RAW_DIR=/path/to/full_1mm
make data-kits KITS_RAW_DIR=/path/to/C4KC-KiTS
```
Preprocessed data is written to `data/aapm/` and `data/kits/`, matching the the paths given in the configs.

## 4. Test the pretrained models

Download our pretrained weights and test them (needs the corresponding preprocessed dataset from step 2):

```
make weights        # download pretrained weights from Hugging Face -> ./weights
make test-aapm      # test the AAPM model
make test-kits      # test the C4KC-KiTS model
```
Metrics are printed and saved to `results/` as CSV + JSON.

## 5. Train a model

To train your own model, use the following commands:
```
make train-aapm
make train-kits
```
Trained models are saved under `models/`. Training logs to Weights & Biases by default; set `WANDB_ENTITY=<your-username>` to choose the entity, or set `use_wandb: false` in `configs/base.yaml` to turn logging off.

To test a model you trained yourself instead of the pretrained weights, run `python scripts/test.py -c configs/aapm.yaml` without `--fix init_exp_dir`, so it reads the checkpoint from the config's `exp_dir` under `models/`.
