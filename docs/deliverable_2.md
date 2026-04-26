# Homework 03: Final Project Warm-up (Deliverable II)
## Option 2: Object Localization

## Scope of This Document

This document is organized by task, with separate subsections for each teammate's contribution.

---
## Task 1: Detail your data preparation strategy for constructing the chosen model.

Stategy:
- images missing annotations (and therefore bounding boxes) will be skipped/cut, as our project is entirely based off predicting bounding boxes. This still leaves us with plenty of data at 100 images per classification.
- images will all be scaled to 224x224x3. From my research, this a pretty standard size for doing ML, and we can always change it later if need be

Steps:
1. Write a script to download the images and annotations from the internet. Since there are a lot of images, this will take a bit. Once the code is ran, the data is stored in [raw_data/](../raw_data/). The images are .jpg files, and the bounding boxes are saved in xml files in the annotations folder.
2. Write a script to go thru every image and:
    - if it has an annotation (and therefore a bounding box), convert it to rgb, and resize it to 224x224x3. Then scale the bounding box as well based off how the image was scaled. Then, append the image and bounding box data to their respective .npy arrays
    - if there is no matching annotation for the image, skip it. 
3. Download the .npy arrays to the disc, once the notebook is run, you can find them at [preprocessed_data/](../preprocessed_data/)


## Task 2

[preprocess_dataset.ipynb](../data_preprocessing/preprocess_dataset.ipynb).



## Task 3: Choose Two Models and Explain Why

### Deep Learning Contribution

For deep learning, I implemented a **ResNet-18-based bounding box regressor** with two selectable backbones:
- pretrained ResNet-18 backbone (`models\model_pretrained.py`)
- from-scratch ResNet-18 backbone (`models\model_scratch.py`)

Model behavior:
- predicts 4 normalized values: `(x_center, y_center, width, height)`
- uses a regression head on top of the backbone feature vector
- applies sigmoid at output time to keep predictions in `[0, 1]`

Why this model fits:
- ResNet-18 is efficient for 224x224 input images
- residual blocks improve optimization stability
- output format directly matches localization regression targets

### Preprocessing Contribution

_To be completed by preprocessing teammate._

### Traditional ML Contribution

**Random Forest Regressor** with PCA preprocessing:
- PCA reduces each image from 224×224×3 = 150,528 features down to 100 principal components
- Random Forest predicts all 4 bounding box coordinates `(x, y, width, height)` simultaneously

Why this model fits:
- natively supports multi-output regression — one model, four outputs
- no feature scaling required
- PCA keeps memory and training time manageable

---

## Task 4: Hyperparameter Selection Strategy

### Deep Learning Contribution

I implemented config-driven hyperparameter control in `deep_learning\train\config.yaml` so experiments can be changed without code edits.

Configurable items include:
- `epochs`, `batch_size`, `lr`, `weight_decay`, `image_size`
- backbone controls: `use_pretrained_backbone`, `freeze_backbone`
- optimizer `name` and optimizer `params`
- loss `name` and loss `params`
- checkpoint run settings (`checkpoint_root`, `checkpoint_prefix`, `checkpoint_run_name`)
- early stopping (`early_stopping_patience`)

This supports reproducible experiments and controlled comparisons.

### Preprocessing Contribution

_To be completed by preprocessing teammate._

### Traditional ML Contribution

Hyperparameters searched using `RandomizedSearchCV` (20 combinations, 3-fold CV, scored by mean IoU):
- `pca__n_components`: [100, 200, 300]
- `rf__n_estimators`: [100, 200, 300]
- `rf__max_depth`: [None, 10, 20, 30]
- `rf__min_samples_leaf`: [1, 2, 4]

Best values found: `n_components=100`, `n_estimators=200`, `max_depth=None`, `min_samples_leaf=2`

---

## Task 5: Model Performance Evaluation

### Deep Learning Contribution

Training uses regression-focused loss functions for box prediction. Implemented options include:
- `smooth_l1`, `huber`, `mse`, `l1`
- `smooth_l1_iou` (combined coordinate + IoU-aware objective)

Current evaluation flow:
- validation loss during training
- best-checkpoint selection by validation loss
- prediction export to XML (one XML per image) for downstream IoU analysis on held-out data

These metrics are appropriate because outputs are continuous box coordinates.

### Preprocessing Contribution

_To be completed by preprocessing teammate._

### Traditional ML Contribution

Metrics used:
- **MSE** — average squared pixel error per coordinate
- **IoU** — bounding box overlap (0 = no overlap, 1 = perfect match)

Both are reported on train and test sets.

---

## Task 6: Underfitting and Overfitting

### Deep Learning Contribution

How I monitor:
- compare training and validation loss curves
- low train loss + much higher val loss indicates overfitting
- high train and val losses indicate underfitting

Mitigation options already supported:
- tune learning rate, batch size, and epochs via config
- switch optimizer/loss and adjust their parameters
- freeze/unfreeze backbone (`freeze_backbone`)
- use regularization settings (dropout in model head, weight decay)
- enable early stopping (`early_stopping_patience`)

### Preprocessing Contribution

_To be completed by preprocessing teammate._

### Traditional ML Contribution

How I monitor:
- compare train and test MSE and IoU
- large gap (low train, high test) indicates overfitting
- both poor indicates underfitting

Results: train IoU 0.588 vs test IoU 0.373 — overfitting from `max_depth=None` allowing fully grown trees

Mitigation options:
- limit `max_depth`
- increase `min_samples_leaf`

---

## Task 8: Prepare ML Algorithms

### Deep Learning Contribution

Implemented a complete deep learning pipeline for ResNet-18 localization regression, including training and prediction export.

What training code does:
- builds the model
- loads `preprocessed_data\images.npy` and `preprocessed_data\bboxes.npy`
- uses fixed split indices from `preprocessed_data\train_indices.npy` and `preprocessed_data\val_indices.npy`
- selects optimizer/loss from config
- trains with validation each epoch
- saves best checkpoint using validation loss

Task 8.1 status:
- deep learning training and checkpointing code for localization regression is completed

Task 8.2 status:
- training/validation loss tracking is implemented
- prediction pipeline (`deep_learning\predict\predict.py`) is implemented and exports XML predictions for the test split (`preprocessed_data\test_indices.npy`) into per-run folders under `deep_learning\predict\runs`

### Preprocessing Contribution

_To be completed by preprocessing teammate._

### Traditional ML Contribution

Code is in [machine_learning/random_forest.ipynb](../machine_learning/random_forest.ipynb).

What the notebook does:
- loads `preprocessed_data\images.npy` and `preprocessed_data\bboxes.npy`
- uses fixed split indices from `preprocessed_data\train_indices.npy` and `preprocessed_data\test_indices.npy`
- applies PCA + Random Forest in a sklearn `Pipeline` to prevent data leakage during cross-validation
- runs `RandomizedSearchCV` (20 combinations, 3-fold CV, scored by mean IoU)
- reports MSE and IoU on train and test sets with a sample visualization
- saves trained pipeline to `machine_learning\models\random_forest_pipeline.joblib`
- exports predictions to `machine_learning\models\predictions.xml`

Best hyperparameters found: `n_components=100`, `n_estimators=200`, `max_depth=None`, `min_samples_leaf=2`

| Split | MSE | IoU |
|-------|-----|-----|
| Train | 258.76 | 0.588 |
| Test | 1130.66 | 0.373 |
