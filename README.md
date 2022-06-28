# Rectal Cancer LNM Classification
## Introduction
We design an efficient classification strategy for paper, **CTE-Net: Automatic COVID-19 Lung Infection Segmentation Boosted by Low-level Features and Fine-grained Textures**. 
Specifically, we design a channel tuning strategy and fine-grain texture enhancing unit to improve COVID-19 infection segmentation.

## Usage
### Installation
1. Requirements

- numpy>=1.21.5
- pandas>=1.1.5
- torch>=1.7.0
- torchvision>=0.11.1
- monai>=0.8.1
- pillow
- yaml
- json

2. Install dependencies.
```shell
pip install -r requirements.txt
```

### Dataset
The COVID-19 Challenge is a public dataset. See [here](https://covid-segmentation.grand-challenge.org).

### Training and Evaluation
The path of dataset need to be set in **./CTE_Net/CTE-Net.yaml** before training.
```
python train.py --config_path ./CTE_Net/CTE-Net.yaml
```
### Evaluation
```
bash evaluation.sh
```
The value of `training_dir` need to be set before evaluation, so the code knows which model needs to be evaluated.

