# Resolution-Independent Neural Operators for Multi-Rate Sparse-View CT

## Quickstart

To reproduce our numbers on the pretrained models you don't need the datasets — download the weights and run test:

```bash
make setup          # install python deps (see the torch-radon note below)
make weights        # download pretrained weights from Hugging Face -> ./weights
make test-aapm      # test the AAPM model; metrics printed + saved to results/
make test-kits      # test the C4KC-KiTS model
```

`make help` lists every target (`setup`, `weights`, `data-aapm`/`data-kits`, `train-aapm`/`train-kits`, `test-aapm`/`test-kits`). Each is a thin wrapper over a `python ...` command, so you can run them directly too (shown in the sections below).

Testing writes CSV + JSON to `results/` and needs **no Weights & Biases account** — the `test-*` targets pass `--no-wandb`. To enable W&B logging, drop `--no-wandb` and set `export WANDB_ENTITY=<your-username>` (or edit `configs/base.yaml`).

## Installation
- Install required python packages using the following command
```
conda create --name cto_env python=3.11
conda activate cto_env
python -m pip install -r requirements.txt
pip install --no-build-isolation torch-harmonics
```
The code was tested using a conda environment running Python 3.11 on a Linux computer.
### torch-radon
- Download **torch-radon** from https://github.com/matteo-ronchetti/torch-radon
```
git clone https://github.com/matteo-ronchetti/torch-radon.git
```
- Due to some out-dated Pytorch function in torch-radon, you need to modify code by running
```
cd torch-radon
patch -p1 < path/to/cto_cvpr/torch-radon_fix/torch-radon_fix.patch
```
- Install torch-radon by running
```
python setup.py install
```

<!-- ### neuraloperator
- Install the **neuraloperator** library by following the instructions given at
  https://github.com/neuraloperator/neuraloperator -->

## Datasets

The Low-dose CT **AAPM** dataset is substantially smaller than the **C4KC-KiTS Dataset**, hence easier to work with.

### AAPM Dataset
- Download original **AAPM** dataset from https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144226105715
- Navigate to `Dataset_dir_to_downloaded_AAMP16/Patient_Data/Training_Image_Data/1mm B30`
- Unzip the `FD_1mm.zip` to `full_1mm/`
- Point the preprocessor at your raw download: either set `export AAPM_RAW_DIR=/path/to/full_1mm/` or edit `Line 5` of `src/data/preprocess_aapm.py`
- Run the preprocessing (or `make data-aapm`). By default the processed data is saved at `data/aapm/` (matching `configs/aapm.yaml`)
```
python src/data/preprocess_aapm.py
```
### C4KC-KiTS Dataset
- Download the **C4KC-KiTS** kidney CT dataset:  
  https://www.cancerimagingarchive.net/collection/c4kc-kits/
- Point the preprocessor at your raw download: either set `export KITS_RAW_DIR=/path/to/C4KC-KiTS` or edit `RAW_DICOM_DIR` at the top of `src/data/preprocess_kits.py` (and optionally `INTER_ORGANIZED_DIR`, the patient-wise intermediate dir)
- Run the preprocessing (or `make data-kits`). By default the processed data is saved at `data/kits/` (matching `configs/kits.yaml`)
```
python src/data/preprocess_kits.py
```
<!-- ## Preprocessing

Run the preprocessing scripts after downloading the datasets:

```
python src/data/aapm.py
python src/data/kits.py
``` -->

## Running Experiments

Before training/testing, some commands that may need to be run before running the final script include the following. These are system dependent, and you may have to modify them depending on your system configuration/ whether you are using a job scheduler like SLURM.

```
source ~/.bashrc
cd PATH/TO/CTO_CVPR
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3 
export WANDB_API_KEY=your_api_key
```
Also, set the number of GPUs being used in `configs/base.yaml`.

To train on the AAPM dataset, use the train.py script with the config file `configs/aapm.yaml`. To use wandb logging, update the logger section in the yaml file `configs/base.yaml`. Delete that header to not use any logging. By default, the trained model will get saved under `models/` directory.

```
python scripts/train.py -c configs/aapm.yaml
```

Similarly, to test the trained model, run the command below. Results are printed and saved (CSV + JSON) in the `results/` directory. Add `--no-wandb` to skip Weights & Biases entirely (metrics are still saved locally).
```
python scripts/test.py -c configs/aapm.yaml
```

To test the **pretrained** weights instead of a locally trained model, download them and point `init_exp_dir` at the checkpoint folder (this is what `make test-aapm` runs):
```
python download_weights.py
python scripts/test.py -c configs/aapm.yaml --fix init_exp_dir weights/aapm --no-wandb
```

To train and test on the C4KC-KiTS dataset, follow the same steps detailed above with the config file `configs/kits.yaml`.