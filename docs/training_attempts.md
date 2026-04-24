# Training Attempts Log

Performance is summarized from each run's `metrics.yaml`, prioritizing localization quality (`success_rate_iou_0p50` and `mean_iou`) rather than validation loss.

| Run Name (Checkpoint-Aligned) | Checkpoint | Performance Summary | Notes |
| --- | --- | --- | --- |
| `model_pytorch_best_legacy_epoch020_valloss0p006720` | `model_pytorch_best_legacy_epoch020_valloss0p006720/model.pt` | `success@0.50: 55.24%`, `mean_iou: 0.4991` (`success@0.75: 11.02%`, `success@0.90: 0.68%`) | Eval folder: `None`. |
| `model_pytorch_best` | `model_pytorch_best.pt` | `success@0.50: 55.24%`, `mean_iou: 0.4991` (`success@0.75: 11.02%`, `success@0.90: 0.68%`) | Eval folder: `20260423_133400`; metrics match the legacy eval output. |
| `model_pytorch_best_epoch019_valloss0p504873` | `model_pytorch_best_epoch019_valloss0p504873/model.pt` | `success@0.50: 55.92%`, `mean_iou: 0.5008` (`success@0.75: 9.66%`, `success@0.90: 0.41%`) | Eval folder: `20260423_143020`. |
| `model_best_20260423_150329` | `model_best_20260423_150329/best_model.pt` | `success@0.50: 81.36%`, `mean_iou: 0.6499` (`success@0.75: 35.92%`, `success@0.90: 2.99%`) | Eval folder: `20260423_161341`; strongest run by both primary metrics. |

## Ranking by Primary Metrics

1. `model_best_20260423_150329` (`mean_iou: 0.6499`, `success@0.50: 81.36%`)
2. `model_pytorch_best_epoch019_valloss0p504873` (`mean_iou: 0.5008`, `success@0.50: 55.92%`)
3. `model_pytorch_best_legacy_epoch020_valloss0p006720` and `model_pytorch_best` (tie: `mean_iou: 0.4991`, `success@0.50: 55.24%`)

## Next Step

Use `model_best_20260423_150329` as the current baseline and compare future training changes against its `success@0.50` and `mean_iou`.
