# Build progress

```
[██████████████████████████████]  6 / 6   (100%)
```

| # | Part | Status |
|---|------|--------|
| 1 | Scaffold, synthetic dataset, metrics, classical baseline | ✅ done |
| 2 | Train pretrained YOLO, compare against baseline (P / R / mAP) | ✅ done |
| 3 | ONNX Runtime, INT8 quantization, batching, latency benchmarks | ✅ done |
| 4 | FastAPI backend: upload, live stream, tracking, report | ✅ done |
| 5 | React dashboard | ✅ done |
| 6 | Docker, CI, docs, polish | ✅ done |

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

## Part 4 deliverables
- `api/main.py` (REST + 2 WebSockets + MJPEG), `visioninspect/pipeline.py`, `visioninspect/live.py`, `visioninspect/registry.py`
- `visioninspect/tracking/tracker.py` (defect tracker + part tracker), `visioninspect/reporting/report.py` (JSON / CSV / PDF)
- `scripts/eval_video.py` (system-level scoring against per-frame ground truth)
- Close-camera clip: 6/6 parts, 100% OK/NOK verdicts, 7/7 unique defects, frame mAP@0.5 0.973; 1.23x real time with frame skipping
- Known weakness documented: far camera (small parts) drops frame mAP to 0.51
- 21 unit/integration tests passing

## Part 5 deliverables
- `frontend/` React 19 + Vite dashboard: Inspect a clip, Live, Single part, Models (built output in `frontend/dist`)
- Belt timeline, KPI ledger, defect gallery, latency chart, report downloads, live webcam over WebSocket
- API additions: sample parts with ground truth, frame stats, static dashboard serving
- `scripts/ui_smoke.py` (headless browser run, zero console errors), screenshots in `docs/screenshots/`
- 24 Python tests passing

## Part 6 deliverables
- `Dockerfile` (ui / runtime / full stages), `docker-compose.yml`, `docker/entrypoint.sh`, `.dockerignore`
- `requirements-runtime.txt` (serving, no PyTorch) and `requirements-train.txt`
- `.github/workflows/ci.yml` (tests on 3.11 + 3.12, dashboard build, Docker build + health smoke test)
- `docs/architecture.png` / `.mmd`, final README, Makefile targets `docker`, `docker-full`, `up`, `down`
- Registry hides the PyTorch backend when Ultralytics is absent
- Verified without Docker: 24 tests pass in a torch-free venv; simulated container layout served and processed the demo job
- Not verified: the actual `docker build` (no daemon in the sandbox)
