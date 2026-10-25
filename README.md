# VisionInspect: Real-Time Computer Vision Inspection System

Detect, localize and classify manufacturing defects in images and video, with a
fair comparison between a classical baseline and a fine-tuned deep detector,
an optimized (ONNX / INT8) inference path, a FastAPI service and a React dashboard.

> **Build status:** Part 1 of 6 complete. See [`docs/PROGRESS.md`](docs/PROGRESS.md).

```
[██████░░░░░░░░░░░░░░░░░░░░░░░░] 1/6 parts  ·  Foundations + baseline ✅
```

## What is in Part 1

| Piece | Where | Why it matters |
|---|---|---|
| Procedural assembly-line dataset (4 defect classes, exact boxes, YOLO format) | `visioninspect/data/synth.py` | Fully reproducible benchmark, no downloads, no leakage between splits |
| Conveyor-belt demo video with per-frame ground truth | `visioninspect/data/video.py` | Drives the live and upload demos, and lets us score video like stills |
| Detection metrics written from scratch (precision, recall, F1, AP@0.5, mAP@0.5:0.95, image-level OK/NOK) | `visioninspect/metrics.py` | Same evaluator for every model, so comparisons are apples to apples |
| Classical OpenCV baseline (colour segmentation, template check, morphology) | `visioninspect/baselines/classical.py` | The bar the deep model has to beat |
| Unit tests | `tests/` | Metrics and generator are verified |

### Defect classes

`scratch` (thin bright line) · `crack` (dark jagged line with branches) · `stain` (soft discoloured blob) · `missing_part` (a screw, capacitor or chip that should be there)

## Quick start

```bash
pip install -r requirements.txt
make data          # 600 train / 150 val / 150 test images -> data/inspect
make video         # conveyor demo clip -> data/demo/conveyor.mp4
make baseline      # evaluate the classical baseline on the test split
make test
```

## Baseline results (test split, 150 images)

| Metric | Classical baseline |
|---|---|
| mAP@0.5 | 0.673 |
| mAP@0.5:0.95 | 0.479 |
| Precision / Recall / F1 (conf ≥ 0.5) | 0.813 / 0.735 / 0.772 |
| Image-level accuracy (defective vs OK) | 0.967 |
| Latency (CPU, mean) | ~19.5 ms (~51 FPS) |

Per-class AP@0.5: scratch 0.687 · crack 0.356 · stain 0.708 · missing_part 0.941.
The baseline is strong on rigid, template-like cases (`missing_part`) and weak on
cracks, which look a lot like scratches and stains to a rule-based system. This is
exactly where a learned detector should win in Part 2.

## Roadmap

1. ✅ Foundations: dataset, metrics, baseline, tests
2. ⏳ Train and compare a pretrained YOLO against the baseline
3. ⏳ Inference optimization: ONNX Runtime, INT8 quantization, batching, latency benchmarks
4. ⏳ FastAPI backend: video upload, live stream, tracking, downloadable report
5. ⏳ React dashboard
6. ⏳ Docker, CI, docs, polish

## Honest notes

* The dataset is synthetic. It is great for a reproducible pipeline and for
  stress-testing the system, but absolute numbers will differ on real factory data.
  The code path (YOLO-format data, metrics, serving) is identical, so swapping in a
  real dataset such as MVTec AD or NEU-DET only needs a new `data.yaml`.
* Plates are axis-aligned with small positional jitter. The baseline relies on that,
  and the deep model does not.
