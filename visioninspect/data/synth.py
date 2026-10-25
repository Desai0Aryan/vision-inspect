"""Procedural generator for an assembly-line inspection dataset.

A "part" is a green plate carrying 7 components (4 screws, 1 chip, 2 capacitors).
Four defect types are rendered procedurally, with exact ground-truth boxes:

    scratch      thin bright line
    crack        dark jagged line with branches
    stain        soft brownish blob
    missing_part a component that should be present but is not

Because the generator is deterministic given a seed, the whole benchmark is
reproducible without downloading anything, and it drives both the still-image
dataset and the conveyor-belt demo video.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from visioninspect import CLASSES

CLS_ID = {c: i for i, c in enumerate(CLASSES)}

# Component template in plate-normalised coordinates: (kind, cx, cy)
COMPONENTS: list[tuple[str, float, float]] = [
    ("screw", 0.10, 0.16),
    ("screw", 0.90, 0.16),
    ("screw", 0.10, 0.84),
    ("screw", 0.90, 0.84),
    ("chip", 0.50, 0.50),
    ("cap", 0.27, 0.50),
    ("cap", 0.73, 0.50),
]
COMPONENT_RADIUS = {"screw": 15, "cap": 11, "chip": 34}  # px at scale 1.0
PLATE_BGR = np.array([62, 112, 48], dtype=np.float32)


@dataclass
class Defect:
    cls: str
    points: np.ndarray | None = None      # polyline in plate-local px (scratch/crack)
    branches: list = field(default_factory=list)
    center: tuple[float, float] | None = None  # stain
    radii: tuple[float, float] | None = None
    angle: float = 0.0
    component: int | None = None          # missing_part index


@dataclass
class PartSpec:
    w: int
    h: int
    defects: list[Defect]
    seed: int

    @property
    def missing(self) -> set[int]:
        return {d.component for d in self.defects if d.cls == "missing_part"}


def component_box(idx: int, w: int, h: int, pad: int = 6) -> tuple[float, float, float, float]:
    kind, nx, ny = COMPONENTS[idx]
    r = COMPONENT_RADIUS[kind]
    cx, cy = nx * w, ny * h
    rx = r if kind != "chip" else r
    ry = r if kind != "chip" else r * 0.8
    return cx - rx - pad, cy - ry - pad, cx + rx + pad, cy + ry + pad


def _jagged(rng, p0, length, angle, step=9, jitter=0.55):
    pts = [np.array(p0, dtype=np.float32)]
    a = angle
    for _ in range(max(2, int(length / step))):
        a += rng.normal(0, jitter)
        pts.append(pts[-1] + step * np.array([np.cos(a), np.sin(a)], dtype=np.float32))
    return np.array(pts)


def sample_spec(rng: np.random.Generator, n_defects: int | None = None, normal_prob: float = 0.15) -> PartSpec:
    w = int(rng.integers(380, 421))
    h = int(rng.integers(300, 341))
    if n_defects is None:
        n_defects = 0 if rng.random() < normal_prob else int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
    defects: list[Defect] = []
    used_missing: set[int] = set()
    for _ in range(n_defects):
        cls = CLASSES[int(rng.integers(0, len(CLASSES)))]
        if cls == "missing_part":
            free = [i for i in range(len(COMPONENTS)) if i not in used_missing]
            if not free:
                continue
            idx = int(rng.choice(free))
            used_missing.add(idx)
            defects.append(Defect(cls, component=idx))
        elif cls in ("scratch", "crack"):
            p0 = (rng.uniform(0.15 * w, 0.85 * w), rng.uniform(0.15 * h, 0.85 * h))
            length = rng.uniform(55, 130) if cls == "scratch" else rng.uniform(60, 140)
            ang = rng.uniform(0, 2 * np.pi)
            pts = _jagged(rng, p0, length, ang, step=9, jitter=0.12 if cls == "scratch" else 0.6)
            branches = []
            if cls == "crack":
                for _b in range(int(rng.integers(0, 3))):
                    k = int(rng.integers(1, len(pts) - 1))
                    branches.append(_jagged(rng, pts[k], rng.uniform(20, 45), ang + rng.choice([-1, 1]) * 0.9, 7, 0.5))
            defects.append(Defect(cls, points=pts, branches=branches))
        else:  # stain
            c = (rng.uniform(0.15 * w, 0.85 * w), rng.uniform(0.15 * h, 0.85 * h))
            rx, ry = rng.uniform(14, 32), rng.uniform(12, 26)
            defects.append(Defect(cls, center=c, radii=(rx, ry), angle=rng.uniform(0, 180)))
    return PartSpec(w, h, defects, int(rng.integers(0, 2**31 - 1)))


def _draw_component(img, kind, cx, cy, rng):
    cx, cy = int(cx), int(cy)
    if kind == "screw":
        cv2.circle(img, (cx, cy), 16, (35, 60, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 14, (200, 205, 208), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 14, (150, 155, 160), 2, cv2.LINE_AA)
        a = rng.uniform(0, np.pi)
        dx, dy = int(9 * np.cos(a)), int(9 * np.sin(a))
        cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), (70, 70, 75), 3, cv2.LINE_AA)
    elif kind == "cap":
        cv2.circle(img, (cx, cy), 12, (25, 45, 25), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 10, (30, 140, 235), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, (20, 100, 190), -1, cv2.LINE_AA)
    else:  # chip
        cv2.rectangle(img, (cx - 36, cy - 28), (cx + 36, cy + 28), (28, 28, 30), -1)
        for k in range(-3, 4):
            cv2.rectangle(img, (cx - 42, cy + k * 7 - 2), (cx - 36, cy + k * 7 + 2), (190, 200, 205), -1)
            cv2.rectangle(img, (cx + 36, cy + k * 7 - 2), (cx + 42, cy + k * 7 + 2), (190, 200, 205), -1)
        cv2.circle(img, (cx - 24, cy - 17), 3, (120, 120, 125), -1, cv2.LINE_AA)


def _draw_missing_pad(img, kind, cx, cy):
    cx, cy = int(cx), int(cy)
    r = COMPONENT_RADIUS[kind]
    if kind == "chip":
        cv2.rectangle(img, (cx - 36, cy - 28), (cx + 36, cy + 28), (52, 98, 40), 1)
    else:
        cv2.circle(img, (cx, cy), r - 3, (52, 98, 40), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, (35, 70, 28), -1, cv2.LINE_AA)


def render_plate(spec: PartSpec) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Render the plate patch (h,w,3) and its defect boxes in plate-local px."""
    rng = np.random.default_rng(spec.seed)
    w, h = spec.w, spec.h
    plate = np.empty((h, w, 3), np.float32)
    plate[:] = PLATE_BGR * rng.uniform(0.92, 1.08)
    # copper-ish trace pattern + fine noise
    for _ in range(14):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1 = x0 + rng.integers(-120, 120)
        cv2.line(plate, (int(x0), int(y0)), (int(x1), int(y0)), (74, 128, 58), 2, cv2.LINE_AA)
        cv2.line(plate, (int(x1), int(y0)), (int(x1), int(y0 + rng.integers(-80, 80))), (74, 128, 58), 2, cv2.LINE_AA)
    plate += rng.normal(0, 2.0, plate.shape).astype(np.float32)
    img = np.clip(plate, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (35, 70, 28), 4)

    boxes: list[tuple[int, float, float, float, float]] = []
    missing = spec.missing
    for i, (kind, nx, ny) in enumerate(COMPONENTS):
        cx, cy = nx * w, ny * h
        if i in missing:
            _draw_missing_pad(img, kind, cx, cy)
        else:
            _draw_component(img, kind, cx, cy, rng)

    overlay = img.copy()
    for d in spec.defects:
        if d.cls == "missing_part":
            boxes.append((CLS_ID[d.cls], *component_box(d.component, w, h)))
        elif d.cls == "scratch":
            cv2.polylines(img, [d.points.astype(np.int32)], False, (205, 220, 215), 2, cv2.LINE_AA)
            x1, y1 = d.points.min(0) - 4
            x2, y2 = d.points.max(0) + 4
            boxes.append((CLS_ID[d.cls], x1, y1, x2, y2))
        elif d.cls == "crack":
            allpts = [d.points] + list(d.branches)
            for p in allpts:
                cv2.polylines(img, [p.astype(np.int32)], False, (18, 20, 18), 3, cv2.LINE_AA)
            cat = np.concatenate(allpts)
            x1, y1 = cat.min(0) - 5
            x2, y2 = cat.max(0) + 5
            boxes.append((CLS_ID[d.cls], x1, y1, x2, y2))
        else:
            cx, cy = d.center
            rx, ry = d.radii
            cv2.ellipse(overlay, (int(cx), int(cy)), (int(rx), int(ry)), d.angle, 0, 360, (28, 62, 108), -1, cv2.LINE_AA)
            ext = max(rx, ry)
            boxes.append((CLS_ID[d.cls], cx - ext * 0.95, cy - ext * 0.95, cx + ext * 0.95, cy + ext * 0.95))
    if any(d.cls == "stain" for d in spec.defects):
        blurred = cv2.GaussianBlur(overlay, (0, 0), 3.2)
        mask = (np.abs(blurred.astype(np.int16) - img.astype(np.int16)).sum(2) > 6)[..., None]
        img = np.where(mask, cv2.addWeighted(img, 0.35, blurred, 0.65, 0), img)
    # clip boxes to plate
    out = []
    for c, x1, y1, x2, y2 in boxes:
        out.append((c, max(0.0, x1), max(0.0, y1), min(w - 1.0, x2), min(h - 1.0, y2)))
    return img, out


def make_background(rng: np.random.Generator, W: int, H: int) -> np.ndarray:
    base = rng.uniform(88, 108)
    bg = np.full((H, W, 3), base, np.float32)
    bg += np.linspace(-10, 10, W, dtype=np.float32)[None, :, None] * rng.uniform(-1, 1)
    bg += rng.normal(0, 3.0, (H, W, 1)).astype(np.float32)
    streaks = rng.normal(0, 4.0, (1, W, 1)).astype(np.float32)
    bg += cv2.GaussianBlur(np.repeat(streaks, H, 0)[..., 0], (0, 0), 1.2)[..., None]
    bg[..., 0] += 4  # slightly cool tint
    return np.clip(bg, 0, 255).astype(np.uint8)


def finish(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Global camera effects: exposure, vignette, blur, sensor noise."""
    f = img.astype(np.float32) * rng.uniform(0.85, 1.12)
    H, W = f.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.18 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    f *= vig[..., None]
    if rng.random() < 0.5:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 0.9))
    f += rng.normal(0, rng.uniform(1.5, 4.0), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def render_sample(rng: np.random.Generator, size: int = 640, spec: PartSpec | None = None):
    """Render one still image. Returns (image BGR uint8, labels float32 [N,5] xyxy+cls)."""
    spec = spec or sample_spec(rng)
    plate, boxes = render_plate(spec)
    bg = make_background(rng, size, size)
    ox = int(rng.integers(20, max(21, size - spec.w - 20)))
    oy = int(rng.integers(20, max(21, size - spec.h - 20)))
    bg[oy:oy + spec.h, ox:ox + spec.w] = plate
    img = finish(bg, rng)
    labels = np.array([[x1 + ox, y1 + oy, x2 + ox, y2 + oy, c] for c, x1, y1, x2, y2 in boxes], np.float32).reshape(-1, 5)
    return img, labels
