# Homework 03: Final Project Warm-up (Deliverable II)
## Preprocess the Dataset and Prepare ML Pipeline

## Introduction

From **Final Project (Deliverable I)**, your group has selected **one** of the following three projects as your course project:

1. **ML application in healthcare**  
   **Option 1:** Build End-to-End ML Pipeline for **Warfarin Dosing Prediction**

2. **ML application in computer vision**  
   **Option 2:** Develop End-to-End ML Pipeline for **Object Localization**

3. **ML application in bioinformatics**  
   **Option 3:** Build End-to-End Machine Learning for **RNA Motif Analysis Using Classification and Clustering**

This assignment is **Deliverable II**. Each group is required to complete the **data preparation for the ML algorithm** and **design the pipeline** for the final project.

---

## Textbook Access for Coding Support

Most of the sample codes for the analyses in this assignment can be found in the textbook.

### How to access the textbook through the SLU library
1. Go to the **SLU Library** website.
2. Search in **Pius XII Memorial Library** for:  
   **Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems**  
   Version: **2022-11-08**
3. Click **SLU full text** and log in using your SLU credentials.

**Instructor advice:** Keep the textbook open while working on the assignment, since it will be useful as a coding reference.

---

# Part I: Preprocess Your Final Project Data

## Task 1 (5 points)
Describe your **data preparation strategy** for constructing the chosen model.

Include the specific preprocessing steps you plan to use for your dataset, such as:
- handling **missing values**  
  (refer to `CSCI5750_MissingData_note.pdf`)
- managing **categorical variables**  
  (for **Projects 1 and 3**)
- working with **image data**  
  (for **Project 2**)

---

## Task 2 (10 points)
Write Python code to preprocess the data according to the requirements for your selected project.

### A. Warfarin Dosing Project (Project 1)

#### Requirement 1
Successfully import the raw **tabular data file** into Python.  
(You may also use Excel to preprocess data locally.)

#### Requirement 2
Properly:
- convert **categorical text data** into numerical representations
- handle **missing values** using **at least one technique** from `CSCI5750_MissingData_note.pdf`

#### Requirement 3
Preprocess the data and save them into NumPy variables:
- **X** = feature dataset
- **Y** = label variable

---

### B. Object Detection Project (Project 2)

#### Requirement 1
Correctly load:
- image files
- the annotation table containing **bounding box** information

#### Requirement 2
Accurately visualize the images along with the **ground-truth bounding boxes** around the objects of interest.

#### Requirement 3
Properly preprocess all images and bounding box information and save them into NumPy variables:
- **X** = feature dataset
- **Y** = label variable

**Reference:** Use the project materials/textbook for image loading and preprocessing.

---

### C. RNA Motif Project (Project 3)

#### Requirement 1
Correctly import the **motif feature dataset** (provided dataset 1) into Python.

#### Requirement 2
Properly:
- merge the motif datasets into a **single full dataset**
- convert categorical text data into numerical labels for the **`Label`** column

#### Requirement 3
Preprocess the data and save them into NumPy variables:
- **X** = feature dataset
- **Y** = label variable

---

### Requirement 4 (applies to any project)
Conduct **comprehensive exploratory data analysis (EDA)** and visualization on the original dataset relevant to your project.

Using Python packages such as:
- `pandas`
- `matplotlib`

would be highly beneficial.

---

## Task 3 (10 points)
Choose **two machine learning models** that are well-suited for your selected project.

Explain:
- why each model fits your project
- how you plan to use each model for prediction

You may choose from methods practiced in previous assignments.

---

## Task 4 (5 points)
Explain how you would choose the **optimal hyperparameters** for your selected ML models.

Discuss:
- your approach for hyperparameter selection
- tradeoffs between different hyperparameter choices
- how they affect model complexity
- how they may lead to **underfitting** or **overfitting**

---

## Task 5 (5 points)
After building the model, explain how you would **evaluate its performance**.

Briefly describe:
- which performance metrics you will use
- why those metrics are appropriate for your selected project

---

## Task 6 (5 points)
Explain how you would determine whether a model is:
- **underfitted**
- **overfitted**

Use techniques discussed in class, such as:
- bias/variance analysis
- other relevant techniques

Also outline your strategy for mitigating:
- overfitting
- underfitting

---

## Task 7 (5 points)
Refine the **graphical end-to-end flowchart** you created in Deliverable I.

Make sure the new flowchart clearly shows the **new analysis added in Deliverable II** at each step.

---

## Task 8 (10 points)
Prepare ML algorithms for the final project.

Possible prediction settings include:
- binary classification
- multi-class classification
- multi-label regression

You need to prepare **sample code** for each machine learning algorithm to:
- fit on the training data
- make predictions on the test data

Before starting Deliverable III, review the corresponding textbook sections.

---

### Task 8.1
Review textbook pages **317–327** on training **multi-layer neural network models**.

Prepare sample code for your final project data.

**Note:** You may also review what you did in **Lab06: Intro to Image Processing & Tips in Deep Learning**.

---

### Task 8.2
Define multiple evaluation metrics and calculate performance on the **training** and **test** datasets.

Use:
- accuracy
- precision
- recall
- F1-score
- ROC score

#### For Projects 1 and 3
You may use the following `sklearn.metrics` functions:
- `accuracy_score`
- `precision_score`
- `recall_score`
- `f1_score`
- `roc_curve` / `roc_auc_score`

Make sure the order of:
- actual labels
- predicted labels

is set correctly.

#### Example
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

acc =  # sample code for accuracy
prec = # sample code for precision
recall = # sample code for recall
roc =  # sample code for ROC score
f1 =   # sample code for F1-score