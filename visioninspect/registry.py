"""Lazy, thread-safe registry of the available detectors.

Backends: ``int8`` (ONNX Runtime INT8, default), ``fp32`` (ONNX Runtime), ``pytorch`` and ``classical``.
A backend is only listed if its weights are present, so the API degrades gracefully.
"""
from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("VI_MODELS_DIR", ROOT / "models"))
IMGSZ = int(os.environ.get("VI_IMGSZ", 512))

_SPEC = {
    "int8": ("ONNX Runtime INT8", MODELS / "yolo_int8.onnx"),
    "fp32": ("ONNX Runtime FP32", MODELS / "yolo_fp32.onnx"),
    "pytorch": ("PyTorch FP32", MODELS / "yolo_pretrained.pt"),
    "classical": ("Classical OpenCV baseline", None),
}
_cache: dict = {}
_lock = threading.Lock()


def available() -> list[dict]:
    out = []
    has_torch = importlib.util.find_spec("ultralytics") is not None
    for key, (label, path) in _SPEC.items():
        if key == "pytorch" and not has_torch:
            continue  # slim runtime image ships without PyTorch
        if path is None or Path(path).exists():
            out.append({"id": key, "label": label, "size_mb": round(Path(path).stat().st_size / 1e6, 1) if path else 0.0})
    return out


def default_model() -> str:
    ids = [m["id"] for m in available()]
    for k in ("int8", "fp32", "pytorch", "classical"):
        if k in ids:
            return k
    raise RuntimeError("no model available")


def get_detector(key: str | None = None, conf: float = 0.45):
    key = key or default_model()
    if key not in {m["id"] for m in available()}:
        raise KeyError(key)
    with _lock:
        if key not in _cache:
            if key == "classical":
                from visioninspect.baselines.classical import ClassicalDetector
                det = ClassicalDetector()
                det.predict_batch = lambda imgs, _d=det: [_d.predict(i) for i in imgs]
            elif key == "pytorch":
                from visioninspect.models.yolo import YoloDetector
                det = YoloDetector(str(_SPEC[key][1]), imgsz=IMGSZ, conf=0.05)
            else:
                from visioninspect.inference.onnx_detector import OnnxDetector
                det = OnnxDetector(str(_SPEC[key][1]), imgsz=IMGSZ, conf=0.05, name=key)
            _cache[key] = det
    return _cache[key]
