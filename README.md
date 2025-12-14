# Resolution-Independent Neural Operators for Multi-Rate Sparse-View CT

## Installation

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

### neuraloperator
- Install the **neuraloperator** library by following the instructions given at
  https://github.com/neuraloperator/neuraloperator

## Datasets

### AAPM Dataset
- Download original **AAPM** dataset from https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144226105715
- Navigate to `Dataset_dir_to_downloaded_AAMP16/Patient_Data/Training_Image_Data/1mm B30`
- Unzip the `FD_1mm.zip` to `full_1mm/`
- Update `Line 200` in `cto_cvpr/src/data/aapm.py` with the updated dataset path
- Run the preprocessing with the following command. By default, the processed data will be saved at `cto_cvpr/data/aapm16`
```
python src/data/aapm.py
```
### C4KC-KiTS Dataset
- Download the **C4KC-KiTS** kidney CT dataset:  
  https://www.cancerimagingarchive.net/collection/c4kc-kits/
- Run the preprocessing with the following command:
```
python src/data/kits.py
```
<!-- ## Preprocessing

Run the preprocessing scripts after downloading the datasets:

```
python src/data/aapm.py
python src/data/kits.py
``` -->

## Running Experiments

To train on the AAPM dataset, use the train.py script with the config file `configs/aapm.yaml`. To use wandb logging, update the logger section in the yaml file. Delete that header to not use any logging. By default, the trained model will get saved under `models/` directory.

```
python scripts/train.py -c configs/aapm.yaml
```

Similarly, to test the trained model, you can run the following command. If logging is enabled, it will also create a wandb run with a table containing all metrics across different sampling rates.
```
python scripts/test.py -c configs/aapm.yaml
```

To train and test on the C4KC-KiTS dataset, follow the same steps detailed above with the config file `configs/kits.yaml`.