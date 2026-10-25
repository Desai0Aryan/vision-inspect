# Model comparison

Split: `test` (150 images). Confidence thresholds were tuned for max F1 on `val`, never on test.

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Image-level acc | Latency (ms, CPU) | FPS | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| classical | 0.673 | 0.479 | 0.813 | 0.735 | 0.772 | 0.967 | 21.8 | 45.8 | 0.10 |
| yolo_pretrained | 0.991 | 0.830 | 0.989 | 0.989 | 0.989 | 1.000 | 104.7 | 9.6 | 0.45 |

## Per-class AP@0.5

| Model | scratch | crack | stain | missing_part |
|---|---|---|---|---|
| classical | 0.687 | 0.356 | 0.708 | 0.941 |
| yolo_pretrained | 0.965 | 0.998 | 1.000 | 1.000 |