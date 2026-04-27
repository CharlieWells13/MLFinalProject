# Model Evaluation Reports
_Updated: 2026-04-27 17:50:13_

---

## 2026-04-27 17:50:13 — resnet18_pretrained

| Split | Mean IoU | Median IoU | % preds ≥ IoU 0.50 | % preds ≥ IoU 0.75 | n |
|-------|----------|------------|--------------------|--------------------|-|
| **Train** | 0.809 | 0.844 | 96.7% | 79.1% | 2580 |
| **Test** | 0.809 | 0.840 | 98.0% | 77.2% | 738 |

**Hyperparams:** backbone=pretrained  ·  image_size=224  ·  checkpoint=/Users/charlie/classes/ML/MLFinalProject/deep_learning/checkpoints/best/best_model.pt  ·  epochs=15  ·  lr=0.001  ·  weight_decay=0.0001  ·  batch_size=32  ·  loss=smooth_l1_iou  ·  optimizer=adamw

![IoU Distribution](20260427_175013_resnet18_pretrained/iou_distribution.png)

| Train sample | Test sample |
|---|---|
| ![Train](20260427_175013_resnet18_pretrained/sample_bbox_train.png) | ![Test](20260427_175013_resnet18_pretrained/sample_bbox_test.png) |

---

## 2026-04-27 17:50:12 — random_forest

| Split | Mean IoU | Median IoU | % preds ≥ IoU 0.50 | % preds ≥ IoU 0.75 | n |
|-------|----------|------------|--------------------|--------------------|-|
| **Train** | 0.585 | 0.620 | 71.2% | 18.2% | 2580 |
| **Test** | 0.376 | 0.379 | 28.2% | 0.9% | 738 |

**Hyperparams:** n_estimators=200  ·  max_depth=None  ·  min_samples_leaf=2  ·  pca_n_components=100

![IoU Distribution](20260427_175012_random_forest/iou_distribution.png)

| Train sample | Test sample |
|---|---|
| ![Train](20260427_175012_random_forest/sample_bbox_train.png) | ![Test](20260427_175012_random_forest/sample_bbox_test.png) |

---
