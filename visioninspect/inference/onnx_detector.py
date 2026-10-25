"""ONNX Runtime detector with letterboxing, batched inference and class-aware NMS.

Works with any YOLOv8-style export whose output is [B, 4+nc, N] (cx, cy, w, h, class scores).
Same ``predict`` / ``predict_batch`` API as the PyTorch wrapper, so it drops into the
evaluator, the benchmark and (Part 4) the FastAPI service unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from visioninspect import CLASSES


def letterbox(img: np.ndarray, size: int, color: int = 114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


def preprocess(imgs: list[np.ndarray], size: int):
    """BGR uint8 list -> float32 NCHW RGB in [0,1] plus per-image (ratio, padx, pady)."""
    batch, meta = [], []
    for im in imgs:
        lb, r, px, py = letterbox(im, size)
        batch.append(lb[..., ::-1])
        meta.append((r, px, py))
    x = np.ascontiguousarray(np.stack(batch).transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
    return x, meta


def postprocess(out: np.ndarray, meta, conf: float, iou: float, nc: int) -> list[np.ndarray]:
    """out: [B, 4+nc, N]. Returns list of [n,6] arrays (xyxy in original image px, conf, cls)."""
    results = []
    for b in range(out.shape[0]):
        p = out[b].T                           # [N, 4+nc]
        scores = p[:, 4:4 + nc]
        cls = scores.argmax(1)
        confs = scores[np.arange(len(p)), cls]
        keep = confs >= conf
        if not keep.any():
            results.append(np.zeros((0, 6), np.float32))
            continue
        p, cls, confs = p[keep], cls[keep], confs[keep]
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        r, px, py = meta[b]
        x1, y1 = (cx - w / 2 - px) / r, (cy - h / 2 - py) / r
        x2, y2 = (cx + w / 2 - px) / r, (cy + h / 2 - py) / r
        # class-aware NMS via per-class coordinate offset
        off = cls[:, None] * 10000.0
        boxes_xywh = np.stack([x1 + off[:, 0], y1 + off[:, 0], x2 - x1, y2 - y1], 1)
        idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), confs.tolist(), conf, iou)
        idx = np.array(idx).reshape(-1).astype(int)
        results.append(np.stack([x1[idx], y1[idx], x2[idx], y2[idx], confs[idx], cls[idx]], 1).astype(np.float32))
    return results


class OnnxDetector:
    def __init__(self, path: str, imgsz: int = 512, conf: float = 0.05, iou: float = 0.6, threads: int | None = None,
                 name: str | None = None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz, self.conf, self.iou, self.nc = imgsz, conf, iou, len(CLASSES)
        self.name = name or f"onnx({path.split('/')[-1]})"

    def predict_batch(self, imgs: list[np.ndarray]) -> list[np.ndarray]:
        x, meta = preprocess(imgs, self.imgsz)
        out = self.sess.run(None, {self.inp: x})[0]
        return postprocess(out, meta, self.conf, self.iou, self.nc)

    def predict(self, img: np.ndarray) -> np.ndarray:
        return self.predict_batch([img])[0]

    def warmup(self, n: int = 3, bs: int = 1):
        dummy = [np.zeros((self.imgsz, self.imgsz, 3), np.uint8)] * bs
        for _ in range(n):
            self.predict_batch(dummy)
