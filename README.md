# CheXFound

This repository contains source code to train and evaluate the vision-centric foundation model **CheXFound**. 
We pretrained CheXFound on up to 1 million chest X-ray images from publicly available sources. 

## Guide to CheXFound with GLoRI

A jupyter notebook file [chexfound_example.ipynb](./notebooks/chexfound_example.ipynb)  is created to illustrate model inference and interpretation.
Model checkpoints and configuration files required to kick off the jupyter notebook can be accessed from [Google Drive](https://dial-rpi.slack.com/archives/D05NHE6C013/p1734381991204209).
A GLoRI is trained on top of the frozen foundation model CheXFound.
Examples show chest X-rays with cardiomegaly. Predictive confidence is computed.
![predictive_confidence](/notebooks/predictive_confidence.png)

Attention maps for several chest X-rays are provided to illustrate interpretation results.
![glori_attns](/notebooks/glori_attns.png)

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
- `<ROOT>/train/mimic/mimic_000000.jpg`
- `<ROOT>/train/mimic/mimic_000001.jpg`
- `<ROOT>/train/[...]`
- `<ROOT>/train/chexpert/chexpert_000000.jpg`
- `<ROOT>/train/chexpert/chexpert_000001.jpg`
- `<ROOT>/train/[...]`
- `<ROOT>/train/padchest/padchest_000000.jpg`
- `<ROOT>/train/padchest/padchest_000001.jpg`
- `<ROOT>/train/[...]`
- `<ROOT>/train/brax/brax_000000.jpg`
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
CheXFound is pretrained using the ViT-L architecture at a range of chest X-ray resolutions.
Pretraining CheXFound at image resolution 512x512 can be done with the following command lines:
```commandline
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup torchrun --nproc_per_node=8 chexfound/train/train.py \
--config-file chexfound/configs/train/vitl16_ibot333_highres512.yaml \
--output-dir /outputs/chexfound/ibot333_highres512 \
train.dataset_path=CXRDatabase:split=TRAIN:root="/path/to/<ROOT>":extra="/path/to/<EXTRA>" \
&> /outputs/chexfound/ibot333_highres512.log &
```

## Evaluation
This codebase implements linear probe and global and local representations integration (GLoRI).
Evaluating linear probe performance can be done with the following command lines:
```commandline
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup torchrun --nproc_per_node=4 chexfound/eval/classification/linear_glori.py \
--batch-size 64 \
--val-epochs 100 \
--epochs 100 \
--image-size 512 \
--save-checkpoint-frequency 20 \
--eval-period-epochs 20 \
--val-metric-type binary_auc \
--config-file /outputs/chexfound/ibot333_highres512/config.yaml \
--pretrained-weights /outputs/chexfound/ibot333_highres512/eval/training_249999/teacher_checkpoint.pth \
--output-dir /outputs/chexfound/shenzhen/ibot333_512_eval_shenzhen \
--train-dataset Shenzhen:split=TRAIN:root=/eval/shenzhen \
--val-dataset Shenzhen:split=VAL:root=/eval/shenzhen \
--test-dataset Shenzhen:split=TEST:root=/eval/shenzhen \
 &> /outputs/chexfound/shenzhen/ibot333_512_eval_shenzhen.log &
```

Evaluating GLoRI performance can use the command lines below:
```commandline
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup torchrun --nproc_per_node=4 chexfound/eval/classification/linear_glori.py \
--batch-size 64 \
--val-epochs 100 \
--epochs 100 \
--image-size 512 \
--glori \
--cat-cls \
--save-checkpoint-frequency 20 \
--eval-period-epochs 20 \
--val-metric-type binary_auc \
--config-file /outputs/chexfound/ibot333_highres512/config.yaml \
--pretrained-weights /outputs/chexfound/ibot333_highres512/eval/training_249999/teacher_checkpoint.pth \
--output-dir /outputs/chexfound/shenzhen/ibot333_512_eval_shenzhen_glori \
--train-dataset Shenzhen:split=TRAIN:root=/eval/shenzhen \
--val-dataset Shenzhen:split=VAL:root=/eval/shenzhen \
--test-dataset Shenzhen:split=TEST:root=/eval/shenzhen \
 &> /outputs/chexfound/shenzhen/ibot333_512_eval_shenzhen_glori.log &
```
