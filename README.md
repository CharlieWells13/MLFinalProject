# ML Final Project: Object Localization Pipeline

This repository contains our CSCI final project for **Option 2: End-to-End ML Pipeline for Object Localization**.  
The goal is to predict object bounding boxes from pet images.

## Team

- Charlie Wells
- Gihwan (Finn) Jung
- Aleksandre Khvadagadze

## Project Scope

We are building and comparing:
- a **deep learning** bounding-box regression pipeline
- a **traditional ML** baseline pipeline

The current implemented code in this repo is focused on the deep learning component, with docs organized so each teammate can fill their assigned section.

## Repository Structure



## Deep Learning Implementation

- ResNet-18 backbone adapted for bounding box regression
- Output target: 4 normalized values  
  `(x_center, y_center, width, height)`
- Config-driven training and hyperparameter control

Training configuration is handled through YAML files.

## Data Flow (Deep Learning)

1. Read split files (`trainval.txt`, `test.txt`)
2. Parse XML annotations from `data\annotations\xmls\*.xml`
3. Convert boxes from `(xmin, ymin, xmax, ymax)` to normalized `(x_center, y_center, width, height)`
4. Load/resize images from `data\images\*.jpg`
5. Train model and save best checkpoint by validation loss

## How to Run

From repo root:

```powershell
python deep_learning\train\<framework>\train.py
```

Optional custom config:

```powershell
python deep_learning\train\<framework>\train.py --config deep_learning\train\<framework>\config.yaml
```

## Main Dependencies

- Python 3.10+
- A deep learning framework (PyTorch or TensorFlow/Keras)
- PyYAML
- Pillow
- NumPy

## Documentation

- `deep_learning\README.md`: deep learning architecture and training overview
- `docs\deliverable_1.md`: Deliverable I outline
- `docs\deliverable_2.md`: Deliverable II write-up (sectioned by teammate contribution)
