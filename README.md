# ML Final Project: Object Localization Pipeline

End-to-end object localization project (CSCI 4750) on Oxford-IIIT Pet images.  
The task is to predict one bounding box per image and compare deep learning vs traditional ML.

## Team

- Charlie Wells
- Gihwan (Finn) Jung
- Aleksandre Khvadagadze

## Final Results (IoU)

| Model | Split | Mean IoU | Median IoU |
|---|---|---:|---:|
| ResNet-18 regressor | Train | 0.793 | 0.821 |
| ResNet-18 regressor | Test | 0.750 | 0.777 |
| PCA + Random Forest | Train | 0.321 | 0.240 |
| PCA + Random Forest | Test | 0.354 | 0.357 |

Detailed evaluation artifacts are under `eval/results/`.

## Repository Layout

- `data_preprocessing/`: dataset preprocessing notebooks/scripts
- `preprocessed_data/`: generated arrays and split indices (created by preprocessing)
- `deep_learning/`: ResNet models, training pipeline, prediction export
- `machine_learning/`: PCA + Random Forest baseline
- `eval/`: unified IoU evaluator, XML predictions, result reports/plots
- `docs/`: flowchart, architecture diagram, deliverables, final report

## Pipeline Overview

1. Preprocess images + XML annotations into fixed tensors/arrays.
2. Train deep learning model (`ResNet-18`) and traditional baseline (`PCA + Random Forest`).
3. Export predictions for each split to a shared XML format.
4. Evaluate with a unified IoU pipeline (stats + visual diagnostics).

Flowchart: `docs/ML_Final_Flowchart.jpg`  
ResNet architecture: `docs/resnet_18_architecture.png`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Instructions

Run all commands from repo root.

1. Preprocess dataset (notebook-driven):

```powershell
jupyter notebook data_preprocessing\preprocess_dataset.ipynb
```

2. Train deep learning model with one config:

```powershell
python deep_learning\train\train.py --config deep_learning\train\configs\E20_B16_LR0001.yaml
```

3. Run deep learning grid search (all configs):

```powershell
powershell -ExecutionPolicy Bypass -File deep_learning\train\run_grid_search.ps1
```

4. Run Random Forest baseline:

```powershell
python machine_learning\random_forest.py
```

5. Evaluate predictions:

```powershell
python eval\evaluate_model.py --help
```

## Key Artifacts

- Best deep model checkpoint: `deep_learning/checkpoints/best/best.pt`
- Deep model predictions: `eval/predictions/resnet_18/`
- Random Forest predictions: `eval/predictions/random_forest/`
- Evaluation reports: `eval/results/resnet_18_test/report.json`, `eval/results/random_forest_test/report.json`
- Final report: `docs/final_report.md`

## Documentation

- `deep_learning/README.md`
- `docs/deliverable_1.md`
- `docs/deliverable_2.md`
- `docs/final_report.md`
