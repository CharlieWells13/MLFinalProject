# Training Attempts Log

| Checkpoint Folder | Setup | Performance Summary | Notes |
| --- | --- | --- | --- |
| `smooth_l1_e20_scratch_legacy` | Smooth L1, 20 epochs (legacy) | `val_loss: 0.00672` at epoch 20 | Legacy backfilled checkpoint metadata. You observed IoU only around `~0.5`, so low loss did not translate to strong IoU. |
| `smooth_l1_iou_e20_scratch` | Smooth L1 + IoU, ~20 epochs | `train_loss: 0.51047`, `val_loss: 0.50487` at epoch 19 | Better aligned objective than pure Smooth L1, but still moderate validation loss. |
| `smooth_l1_iou_e100_scratch_bestval_0p4053` | Smooth L1 + IoU, configured 100 epochs | Best `val_loss: 0.40527` at epoch 33; final at epoch 38: `train_loss: 0.29755`, `val_loss: 0.43089` | Run stopped early (patience-based), so effectively not full 100 epochs. |
| `smooth_l1_iou_e100_scratch_bestval_0p3377` | Smooth L1 + IoU, 100 epochs | Best `val_loss: 0.33775` at epoch 94; final epoch 100: `train_loss: 0.15602`, `val_loss: 0.35593` | Best-performing run so far by validation loss. |

## Additional note on the 80-epoch check

From the 100-epoch run (`smooth_l1_iou_e100_scratch_bestval_0p3377`), epoch 80 had `val_loss: 0.35643`, which is close to late-epoch values and supports your note that increasing to 80 epochs did not make a major difference.

## Next step

Try pretrained weights and compare against the current best baseline (`smooth_l1_iou_e100_scratch_bestval_0p3377`, best `val_loss: 0.33775`).
