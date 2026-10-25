# Build progress

```
[███████████████░░░░░░░░░░░░░░░]  3 / 6   (50%)
```

| # | Part | Status |
|---|------|--------|
| 1 | Scaffold, synthetic dataset, metrics, classical baseline | ✅ done |
| 2 | Train pretrained YOLO, compare against baseline (P / R / mAP) | ✅ done |
| 3 | ONNX Runtime, INT8 quantization, batching, latency benchmarks | ✅ done |
| 4 | FastAPI backend: upload, live stream, tracking, report | ⏳ next |
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

## Part 3 deliverables
- `visioninspect/inference/onnx_detector.py` (letterbox, batched inference, class-aware NMS)
- `scripts/export_onnx.py` (ONNX export + INT8 static quantization), `scripts/benchmark.py`
- Models: `models/yolo_fp32.onnx`, `models/yolo_int8.onnx`
- INT8: 2.7x faster than PyTorch at bs=1, 3.4x smaller, mAP@0.5 0.988 vs 0.991
- 14 unit tests passing
