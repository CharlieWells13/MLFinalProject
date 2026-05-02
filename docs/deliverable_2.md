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
4. Split the data into train/validate/test via indicies so that data only needs to be downloaded once. Data is split by classification as well, as since there are 100 data points of each of the classifications, there will be 70 train, 10 validate, and 20 test data points per classification.


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


### Traditional ML Contribution

**ExtraTreesRegressor** with HOG feature extraction and PCA:
- HOG (Histogram of Oriented Gradients) extracts edge and shape features from each image — far more informative for localization than raw pixels
- PCA reduces HOG features to a compact representation before the model
- ExtraTreesRegressor predicts all 4 bounding box coordinates `(x, y, width, height)` simultaneously

Why this model fits:
- HOG captures spatial structure (edges, gradients) relevant to object position
- ExtraTrees uses randomized splits — better generalization and less overfitting than standard Random Forest
- natively supports multi-output regression — one model, four outputs

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


### Traditional ML Contribution

Hyperparameters searched using `GridSearchCV` (8 combinations × 5-fold CV = 40 total fits, scored by mean IoU):
- `pca__n_components`: [100, 200]
- `et__n_estimators`: [200]
- `et__min_samples_leaf`: [1, 4]
- `et__max_depth`: [None, 20]

Best values found: `n_components=200`, `n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`

---

## Task 5: Model Performance Evaluation

### Preprocessing Contribution

The predictions will be exported to XML from all models and flow into a single (yet to be built) evalution pipeline. The pipeline will be able to perform IoU evaluation via using the command line in some form similar to 
~ evaluate_prediction.py prediction_data.py
It will then save the evaluation to a log, so that progress can be easily tracked.

Read more about the specifics of the two models below.

### Deep Learning Contribution

Training uses regression-focused loss functions for box prediction. Implemented options include:
- `smooth_l1`, `huber`, `mse`, `l1`
- `smooth_l1_iou` (combined coordinate + IoU-aware objective)

Current evaluation flow:
- validation loss during training
- best-checkpoint selection by validation loss
- prediction export to XML (one XML per image) for downstream IoU analysis on held-out data

These metrics are appropriate because outputs are continuous box coordinates.

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

The pipeline will log the IoU score on the test data and the train data, allowing for easy comparison for over and underfitting.

### Traditional ML Contribution

How I monitor:
- compare train and test MSE and IoU
- large gap (low train, high test) indicates overfitting
- both poor indicates underfitting

Results: train IoU 1.000 vs test IoU 0.392 — ExtraTrees with `max_depth=None` grows trees until each leaf has one sample, perfectly fitting the training set. Cross-validation confirmed that constraining the model (`max_depth=20`, `min_samples_leaf=4`) lowered validation IoU, so the unconstrained model is retained. Test IoU is the reliable performance indicator.

Mitigation options tested:
- `max_depth=20` — reduced train fit but also reduced CV IoU
- `min_samples_leaf=4` — same outcome; CV preferred the unconstrained model

---

## Task 7: End to End Flowchart

![End to End Flowchart](ML_Final_Flowchart.jpg)

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


### Traditional ML Contribution

Code is in [machine_learning/random_forest.ipynb](../machine_learning/random_forest.ipynb).

What the notebook does:
- loads `preprocessed_data\images.npy` and `preprocessed_data\bboxes.npy`
- uses fixed split indices from `preprocessed_data\train_indices.npy` and `preprocessed_data\test_indices.npy`
- extracts HOG features from each image (orientations=8, pixels_per_cell=16×16, cells_per_block=2×2)
- applies PCA + ExtraTreesRegressor in a sklearn `Pipeline` to prevent data leakage during cross-validation
- runs `GridSearchCV` (8 combinations × 5-fold CV = 40 fits, scored by mean IoU)
- reports MSE and IoU on train and test sets with a sample visualization
- saves trained pipeline to `machine_learning\models\random_forest_pipeline.joblib`
- exports predictions to `machine_learning\models\predictions.xml`

Best hyperparameters found: `n_components=200`, `n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`

| Split | MSE | IoU |
|-------|-----|-----|
| Train | 0.02 | 1.000 |
| Test | 1031.74 | 0.392 |
