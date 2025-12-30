# Resolution-Independent Neural Operators for Multi-Rate Sparse-View CT

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
- Update `Line 5` at the top of `cto_cvpr/src/data/preprocess_aapm.py` with the updated dataset path
- Run the preprocessing with the following command. By default, the processed data will be saved at `cto_cvpr/data/aapm16`
```
python src/data/preprocess_aapm.py
```
### C4KC-KiTS Dataset
- Download the **C4KC-KiTS** kidney CT dataset:  
  https://www.cancerimagingarchive.net/collection/c4kc-kits/
- Update `Lines 21-22` at the top of `cto_cvpr/src/data/preprocess_aapm.py` with the original dataset path `RAW_DICOM_DIR` and the desired intermediate dataset path `INTER_ORGANIZED_DIR` where data will be stored patient wise. 
- Run the preprocessing with the following command. By default, the processed data will be saved at `cto_cvpr/data/kits`
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

Similarly, to test the trained model, you can run the following command. Results are printed and saved in the `results/` directory.
```
python scripts/test.py -c configs/aapm.yaml
```

To train and test on the C4KC-KiTS dataset, follow the same steps detailed above with the config file `configs/kits.yaml`.