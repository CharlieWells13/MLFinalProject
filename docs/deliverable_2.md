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

For deep learning, I selected a **ResNet-18-based bounding box regression model**.

Model idea:
- use ResNet-18 as a feature extractor
- replace classification output with a regression head
- predict 4 normalized box values: `(x_center, y_center, width, height)`

Why this model fits:
- ResNet-18 is lightweight enough for this dataset/image size
- Residual connections improve optimization stability
- Regression head directly matches localization output format

### Preprocessing Contribution

_To be completed by preprocessing teammate._

### Traditional ML Contribution

Random Forest Regressor with PCA preprocessing. Raw images are 224×224×3 = 150,528 features per sample, so PCA reduces this to 300 components first. The Random Forest then predicts all 4 bounding box coordinates (x, y, width, height) simultaneously as a multi-output regressor.

---

## Task 4: Hyperparameter Selection Strategy

### Deep Learning Contribution

I implemented config-driven hyperparameter control so experiments can be changed without code edits.

Configurable items include:
- `epochs`, `batch_size`, `lr`, `weight_decay`, `image_size`, `freeze_backbone`
- optimizer `name` and optimizer `params`
- loss `name` and loss `params`
- checkpoint output path

This supports reproducible experiments and controlled comparisons.

### Preprocessing Contribution

_To be completed by preprocessing teammate._

### Traditional ML Contribution

Three hyperparameters drive performance: `n_components` (PCA), `n_estimators`, and `max_depth`. More PCA components preserve more image variance but slow training. More trees reduce prediction variance at the cost of compute. Deeper trees fit training data more closely but risk overfitting. We use `n_components=300`, `n_estimators=300`, and the default `max_depth=None`. Cross-validation for hyperparameter tuning is planned for Deliverable III.

---

## Task 5: Model Performance Evaluation

### Deep Learning Contribution

Training uses regression-focused loss functions (e.g., Smooth L1/Huber/MSE-family losses) for box coordinate prediction.

Evaluation plan for localization:
- validation loss during training
- IoU-based comparison between predicted and ground-truth boxes
- optional threshold report (for example, IoU >= 0.5)

These metrics are appropriate because outputs are continuous box coordinates.

### Preprocessing Contribution

_To be completed by preprocessing teammate._

### Traditional ML Contribution

MSE measures average squared pixel error per coordinate. IoU measures bounding box overlap (0 = no overlap, 1 = perfect). Both are reported on train and test sets.

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

### Preprocessing Contribution

_To be completed by preprocessing teammate._

### Traditional ML Contribution

Train and test MSE and IoU are compared. A large gap (low train loss, high test loss) indicates overfitting. Both being poor indicates underfitting. The results show this clearly — train IoU of 0.588 vs test IoU of 0.373 — consistent with `max_depth=None` allowing fully grown trees that memorize training data. Test IoU is the reliable performance indicator. Mitigation options include limiting `max_depth` or increasing `min_samples_leaf`.

---

## Task 8: Prepare ML Algorithms

### Deep Learning Contribution

Implemented a complete deep learning training pipeline for ResNet-18 bounding box regression.

What training code does:
- builds the model
- loads prepared image/annotation data through a training data pipeline
- selects optimizer/loss from config
- trains with validation each epoch
- saves best checkpoint using validation loss

Task 8.1 status:
- deep learning sample code for localization regression is completed

Task 8.2 status:
- training/validation loss tracking is implemented
- IoU-based reporting is planned as the next evaluation extension

### Preprocessing Contribution

_To be completed by preprocessing teammate._

### Traditional ML Contribution

Code is in [machine_learning/random_forest.ipynb](../machine_learning/random_forest.ipynb). The notebook loads the preprocessed data, applies PCA, splits into 80/20 train/test, runs RandomizedSearchCV (20 combinations, 3-fold CV, scored by mean IoU), and reports MSE and IoU on both sets with a visualization of predictions on sample test images.

Best hyperparameters found: `n_components=100`, `n_estimators=200`, `max_depth=None`, `min_samples_leaf=2`

| Split | MSE | IoU |
|-------|-----|-----|
| Train | 258.76 | 0.588 |
| Test | 1130.66 | 0.373 |
