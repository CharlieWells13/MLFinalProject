# Deep Learning: Bounding Box Regression

This module trains a **ResNet-18-based regressor** to predict 4 bounding box values per image:  
`(x_center, y_center, width, height)` normalized to `[0, 1]`.

## ResNet-18 Overview and Project Modification

ResNet-18 is built from:
- A **stem** (`7x7 conv -> batch norm -> ReLU -> max pool`)
- Four residual stages (`layer1` to `layer4`) with **skip connections**
- Global average pooling to produce a compact feature vector

In this project, we adapted ResNet-18 for regression instead of classification:
- Removed classification output behavior
- Added a regression head to output **4 continuous values**
- Optional output squashing with sigmoid for normalized box targets
- Implemented in:
  - `models\model_pytorch.py` (from-scratch PyTorch ResNet-18)

## Data Processing Pipeline -> Training

The pipelines read dataset splits and annotations, then convert raw examples into model-ready tensors:

- Read image IDs from:
  - `data\annotations\trainval.txt`
  - `data\annotations\test.txt`
- Parse XML annotations in `data\annotations\xmls\*.xml`
- Convert `(xmin, ymin, xmax, ymax)` to normalized `(x_center, y_center, width, height)`
- Load RGB images from `data\images\*.jpg`
- Resize to configured `image_size` (default `224`)

Pipeline implementation:
- PyTorch: `train\data_pipeline.py` -> `DataLoader`

## Config-Driven Hyperparameter Training

Training is fully config-driven through YAML files:
- PyTorch config: `train\config.yaml`

Configurable settings include:
- Core training params (`epochs`, `batch_size`, `lr`, `weight_decay`, `image_size`)
- Model behavior (`freeze_backbone`)
- Checkpoint run settings (`checkpoint_root`, `checkpoint_prefix`, `checkpoint_run_name`)
- Optimizer selection + extra optimizer params
- Loss selection + extra loss params

Training script:
- PyTorch: `train\train.py`

Example runs:

```powershell
python deep_learning\train\train.py
```
