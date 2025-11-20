# CT Reconstruction Codebase

## Installation

- Install **torch_radon**  
  https://github.com/matteo-ronchetti/torch-radon

- Install the **neuraloperator** library  
  https://github.com/neuraloperator/neuraloperator

## Datasets

- Download the **AAPM Low-Dose CT** dataset:  
  https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/

- Download the **C4KC-KiTS** kidney CT dataset:  
  https://www.cancerimagingarchive.net/collection/c4kc-kits/

## Preprocessing

Run the preprocessing scripts after downloading the datasets:

```
python src/data/aapm.py
python src/data/kits.py
```

## Running Experiments

Use the main script with a configuration file:

```
python scripts/main.py -c path_to_config_file
```
Examples of config files can be found in the `configs/` folder