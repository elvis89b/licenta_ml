# UBFNet for Polyp Segmentation

This repository contains the source code and experimental implementations developed for polyp segmentation on the PolypDB dataset.

The final models are available on the [`ubfnet`](https://github.com/elvis89b/licenta_ml/tree/ubfnet) branch:

- `UBFNet` — modality-wise evaluation
- `UBFNet_EFPM` — center-wise evaluation

The `main` branch contains the complete experimental development history.

This work extends the original [FocusNet](https://github.com/JunZengz/FocusNet) implementation.

## Clone the repository

```bash
git clone https://github.com/elvis89b/licenta_ml.git
cd licenta_ml
git switch ubfnet
```

## Create the Conda environment

```bash
conda create -n UBFNet python=3.8.16 -y
conda activate UBFNet
```

## Install dependencies

```bash
pip install torch torchvision
pip install timm numpy opencv-python scikit-learn tqdm wandb albumentations gdown
```

For GPU training, install a PyTorch version compatible with the CUDA version available on the system.

## Download the pretrained backbone

The models use a pretrained PVTv2-B4 backbone.

Download the pretrained weights from Google Drive:

https://drive.google.com/file/d/18n1UdWEL31XN20hJDBqP0M5ccU4InnQT/view

Place the downloaded file at:

```text
pretrained_pth/pvt_v2_b4.pth
```

The expected structure is:

```text
pretrained_pth/
└── pvt_v2_b4.pth
```

The file can also be downloaded from the terminal:

```bash
mkdir -p pretrained_pth

gdown 18n1UdWEL31XN20hJDBqP0M5ccU4InnQT \
  -O pretrained_pth/pvt_v2_b4.pth
```

## Dataset

The experiments were performed on the PolypDB dataset using two evaluation protocols.

### Modality-wise

- WLI
- BLI
- FICE
- LCI
- NBI

### Center-wise

- Simula
- BKAI
- Karolinska

The dataset is not included in this repository.

Configure the image and mask paths inside:

```text
train_modality_wise.py
test_modality_wise.py
train_center_wise.py
test_center_wise.py
```

## Training

### Modality-wise training — UBFNet

```bash
python train_modality_wise.py
```

### Center-wise training — UBFNet_EFPM

```bash
python train_center_wise.py
```

EFPM is applied only during center-wise training and is not used during inference.

## Evaluation

### Modality-wise evaluation

```bash
python test_modality_wise.py
```

### Center-wise evaluation

```bash
python test_center_wise.py
```

## Repository structure

```text
model/
├── UBFNet.py
├── UBFNet_EFPM.py
└── __init__.py

train_modality_wise.py
test_modality_wise.py
train_center_wise.py
test_center_wise.py
```

Pretrained weights, datasets, generated results, W&B files, and model checkpoints are not included in this repository.

