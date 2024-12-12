# CheXFound

This repository contains source code to train and evaluate the vision-centric foundation model **CheXFound**. 
We pretrained CheXFound on up to 1 million chest X-ray images from publicly available sources. 

## Environment
The training and evaluation code requires PyTorch 2.0 and xFormers 0.0.18 as well as a number of other 3rd party packages. 
Note that the code has only been tested with the specified versions and also expects a Linux environment. 
To setup all the required dependencies for training and evaluation, please follow the instructions below:

Clone the repository and then create and activate a dinov2 conda environment using the provided environment definition:
```commandline
conda env create -f conda-extras.yaml
conda activate dinov2-extras
```
## Dataset
### CXR-1M
The root directory of the dataset should hold the following contents:
- `<ROOT>/train/mimic/mimic_000001.jpg`
- `<ROOT>/train/[...]`
- `<ROOT>/train/chexpert/chexpert_000001.jpg`
- `<ROOT>/train/[...]`
- `<ROOT>/train/padchest/padchest_000001.jpg`
- `<ROOT>/train/[...]`
- `<ROOT>/train/brax/brax_000001.jpg`
- `<ROOT>/train/[...]`
- `<ROOT>/val`
- `<ROOT>/test`
- `<ROOT>/labels.txt`

The provided dataset implementation expects a few additional metadata files to be present under the extra directory:
- `<EXTRA>/class-ids-TRAIN.npy`
- `<EXTRA>/class-ids-VAL.npy`
- `<EXTRA>/class-names-TRAIN.npy`
- `<EXTRA>/class-names-VAL.npy`
- `<EXTRA>/entries-TEST.npy`
- `<EXTRA>/entries-TRAIN.npy`
- `<EXTRA>/entries-VAL.npy`

These metadata files can be generated (once) with the following lines of Python code:

```python
from chexfound.data.datasets import ImageNet

root_dir = "<ROOT>"
extra_dir = "<EXTRA>"

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=root_dir, extra=extra_dir)
    dataset.dump_extra()
```

## Foundation model pretraining

## Evaluation

## Model interpretation

![glori_attns](/notebooks/glori_attns.png)