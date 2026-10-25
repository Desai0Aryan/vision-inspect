"""Classical (non-learned) OpenCV baseline detector.

This is the bar the deep model has to clear. It uses hand-engineered
domain knowledge:

1. Segment the plate by colour (HSV) -> plate bbox.
2. Look for each expected component at its template location; weak contrast
   against the plate => ``missing_part``.
3. Mask out components, then find residual anomalies in CIELAB:
     - thick, soft deviations           -> ``stain``
     - thin dark curvilinear structures -> ``crack``
     - thin bright curvilinear lines    -> ``scratch``

Output format matches the deep models: float array [N, 6] (xyxy, conf, cls).
"""
from __future__ import annotations

import cv2
import numpy as np

from visioninspect import CLASSES
from visioninspect.data.synth import COMPONENTS, COMPONENT_RADIUS, component_box

CID = {c: i for i, c in enumerate(CLASSES)}


class ClassicalDetector:
    name = "classical-opencv"

    def __init__(self, missing_thr: float = 16.0, thin_dark: float = 42.0, thin_bright: float = 40.0,
                 stain_thr: float = 22.0, min_len: float = 24.0):
        self.missing_thr = missing_thr
        self.thin_dark = thin_dark
        self.thin_bright = thin_bright
        self.stain_thr = stain_thr
        self.min_len = min_len

    # -- plate localisation --------------------------------------------------
    def _plate_bbox(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 70, 25), (85, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 20000:
            return None
        return cv2.boundingRect(c)  # x, y, w, h

    # -- main entry ----------------------------------------------------------
    def predict(self, img: np.ndarray) -> np.ndarray:
        bb = self._plate_bbox(img)
        if bb is None:
            return np.zeros((0, 6), np.float32)
        px, py, pw, ph = bb
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        roi = lab[py:py + ph, px:px + pw]
        med = np.median(roi.reshape(-1, 3), axis=0)
        dist = np.linalg.norm(roi - med, axis=2)
        dist_s = cv2.GaussianBlur(dist, (0, 0), 1.0)
        out: list[list[float]] = []

        # known-component zones (so they are not mistaken for defects)
        comp_mask = np.zeros((ph, pw), np.uint8)
        for i, (kind, nx, ny) in enumerate(COMPONENTS):
            x1, y1, x2, y2 = component_box(i, pw, ph, pad=10)
            cv2.rectangle(comp_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

        # --- missing parts: component should create strong local contrast
        for i, (kind, nx, ny) in enumerate(COMPONENTS):
            cx, cy = int(nx * pw), int(ny * ph)
            r = int(COMPONENT_RADIUS[kind] * 0.8)
            zone = np.zeros((ph, pw), np.uint8)
            cv2.circle(zone, (cx, cy), r, 255, -1)
            score = float(dist[zone > 0].mean())
            if score < self.missing_thr:
                conf = float(np.clip(0.55 + (self.missing_thr - score) / self.missing_thr * 0.45, 0, 0.99))
                x1, y1, x2, y2 = component_box(i, pw, ph)
                out.append([px + x1, py + y1, px + x2, py + y2, conf, CID["missing_part"]])

        # --- stains: large, soft, low-gradient regions
        free = cv2.bitwise_not(comp_mask)
        inner = np.zeros((ph, pw), np.uint8)
        cv2.rectangle(inner, (14, 14), (pw - 15, ph - 15), 255, -1)
        free = cv2.bitwise_and(free, inner)
        stain_mask = ((dist_s > self.stain_thr * 0.8) & (free > 0)).astype(np.uint8) * 255
        # opening with a large element removes thin lines (cracks/scratches), keeps blobs
        stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        stain_mask = cv2.dilate(stain_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        cnts, _ = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stain_boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 220:
                continue
            x, y, w, h = cv2.boundingRect(c)
            m = np.zeros((ph, pw), np.uint8)
            cv2.drawContours(m, [c], -1, 255, -1)
            strength = float(dist_s[m > 0].mean())
            conf = float(np.clip(0.4 + strength / 60, 0, 0.95))
            stain_boxes.append((x, y, x + w, y + h))
            out.append([px + x, py + y, px + x + w, py + y + h, conf, CID["stain"]])

        # --- thin structures: use channel L only
        L = cv2.GaussianBlur(roi[..., 0], (0, 0), 0.8)
        Lmed = cv2.medianBlur(roi[..., 0].astype(np.uint8), 21).astype(np.float32)
        diff = L - Lmed
        stain_zone = np.zeros((ph, pw), np.uint8)
        for x1, y1, x2, y2 in stain_boxes:
            cv2.rectangle(stain_zone, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 255, -1)
        free2 = cv2.bitwise_and(free, cv2.bitwise_not(stain_zone))
        for sign, thr, cname in ((-1, self.thin_dark, "crack"), (1, self.thin_bright, "scratch")):
            m = ((sign * diff > thr) & (free2 > 0)).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if max(w, h) < self.min_len:
                    continue
                (_, _), (rw, rh), _ = cv2.minAreaRect(c)
                long_s, short_s = max(rw, rh), max(min(rw, rh), 1)
                mm = np.zeros((ph, pw), np.uint8)
                cv2.drawContours(mm, [c], -1, 255, -1)
                strength = float(np.abs(diff[mm > 0]).mean())
                conf = float(np.clip(0.35 + strength / 90 + min(long_s / 300, 0.2), 0, 0.95))
                out.append([px + x, py + y, px + x + w, py + y + h, conf, CID[cname]])
        if not out:
            return np.zeros((0, 6), np.float32)
        return self._nms(np.array(out, np.float32))

    @staticmethod
    def _nms(dets: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
        from visioninspect.metrics import iou_matrix
        order = np.argsort(-dets[:, 4])
        dets = dets[order]
        keep = []
        for i in range(len(dets)):
            if all(dets[i, 5] != dets[j, 5] or iou_matrix(dets[i:i + 1, :4], dets[j:j + 1, :4])[0, 0] < iou_thr for j in keep):
                keep.append(i)
        return dets[keep]
