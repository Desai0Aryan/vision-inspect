"""Lightweight multi-object tracking (SORT-style, no extra dependencies).

Two trackers are used by the video pipeline:

* ``DefectTracker`` follows individual defects frame to frame so that one scratch seen in
  60 frames is counted *once*, not 60 times. It matches on IoU with a constant-velocity
  prediction, keeps a running class vote, and only "confirms" a track after ``min_hits``.
* ``PartTracker`` follows whole parts on the belt (plate segmentation by colour) so we can report
  parts inspected, parts rejected and yield. A part is rejected if any confirmed defect lies inside it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from visioninspect import CLASSES
from visioninspect.metrics import iou_matrix


@dataclass
class Track:
    id: int
    box: np.ndarray                      # xyxy
    cls_votes: Counter = field(default_factory=Counter)
    conf_sum: float = 0.0
    conf_max: float = 0.0
    hits: int = 1
    age: int = 0                         # frames since creation
    missed: int = 0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    first_frame: int = 0
    last_frame: int = 0
    best_frame: int = 0
    best_box: np.ndarray | None = None
    last_conf: float = 0.0

    @property
    def cls(self) -> int:
        return self.cls_votes.most_common(1)[0][0]

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / max(self.hits, 1)

    def predicted(self) -> np.ndarray:
        shift = np.array([self.vel[0], self.vel[1], self.vel[0], self.vel[1]], np.float32) * (self.missed + 1)
        return self.box + shift


def _greedy_match(cost: np.ndarray, thr: float):
    """Greedy assignment on a similarity matrix (higher is better). Returns list of (row, col)."""
    pairs = []
    if cost.size == 0:
        return pairs
    c = cost.copy()
    while True:
        i, j = np.unravel_index(np.argmax(c), c.shape)
        if c[i, j] < thr:
            break
        pairs.append((int(i), int(j)))
        c[i, :] = -1
        c[:, j] = -1
    return pairs


class DefectTracker:
    def __init__(self, iou_thr: float = 0.15, max_missed: int = 8, min_hits: int = 4):
        self.iou_thr, self.max_missed, self.min_hits = iou_thr, max_missed, min_hits
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._next = 1

    def update(self, dets: np.ndarray, frame_idx: int) -> list[Track]:
        """dets [N,6] (xyxy, conf, cls). Returns the currently *active, confirmed* tracks."""
        dets = dets.reshape(-1, 6)
        pred = np.array([t.predicted() for t in self.tracks], np.float32).reshape(-1, 4)
        sim = iou_matrix(pred, dets[:, :4]) if len(self.tracks) and len(dets) else np.zeros((len(self.tracks), len(dets)))
        pairs = _greedy_match(sim, self.iou_thr)
        matched_t = {i for i, _ in pairs}
        matched_d = {j for _, j in pairs}
        for i, j in pairs:
            t, d = self.tracks[i], dets[j]
            new_box = d[:4].astype(np.float32)
            c_old, c_new = (t.box[:2] + t.box[2:]) / 2, (new_box[:2] + new_box[2:]) / 2
            t.vel = 0.6 * t.vel + 0.4 * (c_new - c_old) / (t.missed + 1)
            t.box = new_box
            t.hits += 1
            t.missed = 0
            t.last_frame = frame_idx
            t.cls_votes[int(d[5])] += 1
            t.conf_sum += float(d[4])
            t.last_conf = float(d[4])
            if d[4] >= t.conf_max:
                t.conf_max, t.best_frame, t.best_box = float(d[4]), frame_idx, new_box.copy()
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.missed += 1
        for j, d in enumerate(dets):
            if j in matched_d:
                continue
            t = Track(self._next, d[:4].astype(np.float32), Counter({int(d[5]): 1}), float(d[4]), float(d[4]),
                      first_frame=frame_idx, last_frame=frame_idx, best_frame=frame_idx, best_box=d[:4].astype(np.float32).copy(), last_conf=float(d[4]))
            self._next += 1
            self.tracks.append(t)
        alive = []
        for t in self.tracks:
            t.age += 1
            if t.missed > self.max_missed:
                self.finished.append(t)
            else:
                alive.append(t)
        self.tracks = alive
        return [t for t in self.tracks if t.hits >= self.min_hits and t.missed == 0]

    def confirmed(self) -> list[Track]:
        return [t for t in self.finished + self.tracks if t.hits >= self.min_hits]


# ----------------------------------------------------------------------------- parts
@dataclass
class Part:
    id: int
    box: np.ndarray
    first_frame: int
    last_frame: int
    max_area: float
    missed: int = 0
    defect_ids: set = field(default_factory=set)
    classes: Counter = field(default_factory=Counter)
    seen_whole: bool = False

    @property
    def rejected(self) -> bool:
        return len(self.defect_ids) > 0


def segment_parts(frame: np.ndarray, min_area: float = 9000) -> list[np.ndarray]:
    """Colour-segment green plates -> list of xyxy boxes."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 70, 25), (85, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append(np.array([x, y, x + w, y + h], np.float32))
    return out


class PartTracker:
    def __init__(self, frame_w: int, iou_thr: float = 0.1, max_missed: int = 6, min_frames: int = 5):
        self.w = frame_w
        self.iou_thr, self.max_missed, self.min_frames = iou_thr, max_missed, min_frames
        self.parts: list[Part] = []
        self.done: list[Part] = []
        self._next = 1

    def update(self, frame: np.ndarray, frame_idx: int) -> list[Part]:
        boxes = segment_parts(frame)
        pred = np.array([p.box for p in self.parts], np.float32).reshape(-1, 4)
        det = np.array(boxes, np.float32).reshape(-1, 4)
        sim = iou_matrix(pred, det) if len(pred) and len(det) else np.zeros((len(pred), len(det)))
        pairs = _greedy_match(sim, self.iou_thr)
        mt, md = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, j in pairs:
            p = self.parts[i]
            p.box, p.missed, p.last_frame = det[j], 0, frame_idx
            area = float((det[j][2] - det[j][0]) * (det[j][3] - det[j][1]))
            p.max_area = max(p.max_area, area)
            if det[j][0] > 2 and det[j][2] < self.w - 2:
                p.seen_whole = True
        for i, p in enumerate(self.parts):
            if i not in mt:
                p.missed += 1
        for j, b in enumerate(det):
            if j in md:
                continue
            area = float((b[2] - b[0]) * (b[3] - b[1]))
            self.parts.append(Part(self._next, b, frame_idx, frame_idx, area, seen_whole=bool(b[0] > 2 and b[2] < self.w - 2)))
            self._next += 1
        alive = []
        for p in self.parts:
            if p.missed > self.max_missed:
                self.done.append(p)
            else:
                alive.append(p)
        self.parts = alive
        return self.parts

    def assign_defects(self, tracks: list[Track]):
        for t in tracks:
            cx, cy = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
            for p in self.parts:
                if p.box[0] <= cx <= p.box[2] and p.box[1] <= cy <= p.box[3]:
                    p.defect_ids.add(t.id)
                    break

    def all_parts(self) -> list[Part]:
        return [p for p in self.done + self.parts if (p.last_frame - p.first_frame + 1) >= self.min_frames]
