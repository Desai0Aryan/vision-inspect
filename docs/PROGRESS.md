# Build progress

```
[██████████░░░░░░░░░░░░░░░░░░░░]  2 / 6   (33%)
```

| # | Part | Status |
|---|------|--------|
| 1 | Scaffold, synthetic dataset, metrics, classical baseline | ✅ done |
| 2 | Train pretrained YOLO, compare against baseline (P / R / mAP) | ✅ done |
| 3 | ONNX Runtime, INT8 quantization, batching, latency benchmarks | ⏳ next |
| 4 | FastAPI backend: upload, live stream, tracking, report | ⏳ |
| 5 | React dashboard | ⏳ |
| 6 | Docker, CI, docs, polish | ⏳ |

## Part 1 deliverables
- Reproducible dataset generator (4 classes) and YOLO-format writer
- Conveyor demo video with per-frame ground truth
- Metrics: IoU, AP, mAP@0.5, mAP@0.5:0.95, P / R / F1, image-level confusion
- Classical baseline: test mAP@0.5 = 0.673, ~51 FPS on CPU
- 7 unit tests passing

## Part 2 deliverables
- `scripts/train_yolo.py`, `scripts/compare_models.py`, `visioninspect/models/yolo.py`
- Trained weights `models/yolo_pretrained.pt`
- Test mAP@0.5: classical 0.673 vs YOLOv8n 0.991; CPU latency 22 ms vs 105 ms
