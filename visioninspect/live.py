"""Frame-at-a-time inspector for live use (webcam over WebSocket, MJPEG demo stream)."""
from __future__ import annotations

import time

import cv2
import numpy as np

from visioninspect import CLASSES
from visioninspect.pipeline import InspectConfig, draw_detections, draw_hud
from visioninspect.tracking.tracker import DefectTracker, PartTracker


class LiveInspector:
    def __init__(self, detector, model_name: str, cfg: InspectConfig | None = None, frame_w: int = 960):
        self.det, self.model, self.cfg = detector, model_name, cfg or InspectConfig()
        self.dtrack = DefectTracker(min_hits=max(2, self.cfg.min_hits - 1))
        self.ptrack = PartTracker(frame_w)
        self.n = 0
        self.ema_fps = 0.0
        self.ema_lat = 0.0
        self._t_prev = None

    def step(self, frame: np.ndarray, draw: bool = True):
        t0 = time.perf_counter()
        raw = self.det.predict_batch([frame])[0]
        lat = (time.perf_counter() - t0) * 1000
        dets = raw[raw[:, 4] >= self.cfg.conf] if len(raw) else raw
        confirmed = self.dtrack.update(dets, self.n)
        self.ptrack.update(frame, self.n)
        self.ptrack.assign_defects(confirmed)
        parts = self.ptrack.all_parts()
        n_rej = sum(1 for p in parts if p.rejected)
        now = time.perf_counter()
        if self._t_prev:
            f = 1 / max(now - self._t_prev, 1e-6)
            self.ema_fps = f if self.ema_fps == 0 else 0.9 * self.ema_fps + 0.1 * f
        self._t_prev = now
        self.ema_lat = lat if self.ema_lat == 0 else 0.9 * self.ema_lat + 0.1 * lat
        out = frame
        if draw:
            out = frame.copy()
            tb = np.array([[*t.box, t.mean_conf, t.cls] for t in confirmed], np.float32).reshape(-1, 6)
            draw_detections(out, tb, [t.id for t in confirmed])
            draw_hud(out, self.ema_fps, self.ema_lat, len(parts), n_rej, len(self.dtrack.confirmed()), self.model)
        info = {"frame": self.n, "latency_ms": round(lat, 2), "fps": round(self.ema_fps, 1), "parts": len(parts), "rejected": n_rej,
                "defects": len(self.dtrack.confirmed()),
                "detections": [{"class": CLASSES[int(d[5])], "confidence": round(float(d[4]), 3),
                                "box": [round(float(v), 1) for v in d[:4]]} for d in dets]}
        self.n += 1
        return out, info
