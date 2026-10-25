# VisionInspect: Real-Time Computer Vision Inspection System

Detect, localize and classify manufacturing defects in images and video, with a
fair comparison between a classical baseline and a fine-tuned deep detector,
an optimized (ONNX / INT8) inference path, a FastAPI service and a React dashboard.

> **Build status:** Part 3 of 6 complete. See [`docs/PROGRESS.md`](docs/PROGRESS.md).

```
[███████████████░░░░░░░░░░░░░░░] 3/6 parts  ·  Foundations ✅  ·  YOLO vs baseline ✅  ·  Optimized inference ✅
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

## Part 2: pretrained YOLO vs classical baseline

COCO-pretrained **YOLOv8n** fine-tuned for 12 epochs at 512 px on CPU (about 45 minutes on 2 cores).
Both models go through the *same* evaluator on the held-out test split (150 images).
Each model's confidence threshold is tuned for F1 on `val`, never on test.

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Image-level acc | CPU latency |
|---|---|---|---|---|---|---|---|
| Classical OpenCV | 0.673 | 0.479 | 0.813 | 0.735 | 0.772 | 0.967 | 22 ms (46 FPS) |
| **YOLOv8n (pretrained)** | **0.991** | **0.830** | **0.989** | **0.989** | **0.989** | **1.000** | 105 ms (9.6 FPS) |

Per-class AP@0.5 (classical → YOLO): scratch 0.69 → 0.97, crack 0.36 → 1.00, stain 0.71 → 1.00, missing_part 0.94 → 1.00.

**The trade-off:** YOLO is much more accurate but about 5x slower on this 2-core CPU in plain PyTorch.
Part 3 closes that gap. (The 105 ms here is a single un-batched pass; the controlled benchmark below measures PyTorch at about 57-69 ms.)

Artifacts: `outputs/compare/` (metrics table, PR curves, per-class AP, class-confusion matrices, qualitative grid)
and `outputs/yolo_training/` (loss and mAP curves). Reproduce with:

```bash
python scripts/train_yolo.py --weights models/yolov8n.pt --epochs 12 --imgsz 512 --name yolo_pretrained
python scripts/compare_models.py --split test --yolo models/yolo_pretrained.pt --imgsz 512
```

Caveats: the data is synthetic and fairly clean, so near-perfect scores are expected and will drop on real
factory images. The pretraining ablation (`--scratch`) is implemented but was not run here, to keep CPU time down.

## Part 3: inference optimization (ONNX Runtime, INT8, batching)

The trained YOLO is exported to ONNX (dynamic batch), then quantized to INT8 with ONNX Runtime static
quantization (QDQ, per-channel weights, MinMax calibration on 64 training images). All numbers are end to end
(letterbox + model + NMS) on the 150-image test split, 2 CPU threads.

| Backend | Size | mAP@0.5 | mAP@0.5:0.95 | ms/img bs=1 | ms/img bs=4 | Best throughput | Speed-up (bs=1) |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 6.5 MB | 0.991 | 0.830 | 68.6 | 57.2 | 17.5 img/s | 1.0x |
| ONNX Runtime FP32 | 12.9 MB | 0.991 | 0.830 | 38.5 | 35.8 | 28.0 img/s | 1.8x |
| **ONNX Runtime INT8** | **3.8 MB** | 0.988 | 0.807 | **25.7** | **21.8** | **45.9 img/s** | **2.7x** |

* ONNX FP32 reproduces PyTorch accuracy exactly, which validates the custom letterbox / decode / NMS code.
* INT8 is **2.6x faster** than PyTorch at batch 4 and **3.4x smaller**, for -0.003 mAP@0.5 and -0.023 mAP@0.5:0.95.
* Batching helps a little (about 10-15%) on a 2-core CPU; it would help far more on a GPU.
* Timings on a shared 2-core VM vary by roughly +/-15% run to run. The ranking and the ratios are stable.

**Lesson learned (kept as a documented ablation):** quantizing *every* node (`--naive`) gives **mAP = 0.000**.
The YOLOv8 head's final 1x1 class predictor outputs logits with a huge dynamic range, and 8-bit activations flush the
class scores to zero. Keeping the 118 nodes of the final box/class predictors and the decode in float, while quantizing
the rest, recovers 99.7% of the accuracy and is the default recipe in `scripts/export_onnx.py`.

```bash
python scripts/export_onnx.py --weights models/yolo_pretrained.pt --imgsz 512 --calib 64
python scripts/benchmark.py --split test      # -> outputs/benchmark/{benchmark.md,benchmark.json,latency_throughput.png}
```

## Roadmap

1. ✅ Foundations: dataset, metrics, baseline, tests
2. ✅ Train and compare a pretrained YOLO against the baseline
3. ✅ Inference optimization: ONNX Runtime, INT8 quantization, batching, latency benchmarks
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
