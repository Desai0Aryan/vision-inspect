"""Thin wrapper that gives a trained YOLO the same ``predict(img) -> [N,6]`` API as the baseline."""
from __future__ import annotations

import time

import numpy as np


class YoloDetector:
    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.05, iou: float = 0.6, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device
        self.name = f"yolo({weights.split('/')[-1]})"

    def predict(self, img: np.ndarray) -> np.ndarray:
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 6), np.float32)
        return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1).astype(np.float32)

    def predict_batch(self, imgs: list[np.ndarray]) -> list[np.ndarray]:
        rs = self.model.predict(imgs, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        out = []
        for r in rs:
            b = r.boxes
            out.append(np.zeros((0, 6), np.float32) if b is None or len(b) == 0 else
                       np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1).astype(np.float32))
        return out

    def warmup(self, n: int = 3):
        dummy = np.zeros((640, 640, 3), np.uint8)
        for _ in range(n):
            self.predict(dummy)
