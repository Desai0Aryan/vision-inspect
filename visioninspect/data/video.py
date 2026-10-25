"""Conveyor-belt demo video: parts glide left->right, some defective.

Writes an MP4 plus a ground-truth JSON (per-frame boxes) so the live pipeline
can be evaluated on video exactly like on stills.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from visioninspect import CLASSES
from visioninspect.data.synth import make_background, render_plate, sample_spec


def make_demo_video(out_path: str | Path, seconds: int = 14, fps: int = 30, W: int = 960, H: int = 540,
                    seed: int = 7, n_parts: int = 6, scale: float = 0.62) -> dict:
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = seconds * fps
    spacing = 340  # px between part origins, wider than a part so none overlap
    belt_speed = (n_parts * spacing + W + 300) / total  # px / frame, so every part crosses the scene
    parts = []
    for i in range(n_parts):
        # guarantee a mix: part 0 and 3 clean, others defective
        spec = sample_spec(rng, n_defects=0 if i in (0, 3) else int(rng.choice([1, 2])))
        plate, boxes = render_plate(spec)
        plate = cv2.resize(plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        boxes = [(c, x1 * scale, y1 * scale, x2 * scale, y2 * scale) for c, x1, y1, x2, y2 in boxes]
        parts.append({"img": plate, "boxes": boxes, "x0": -plate.shape[1] - i * spacing,
                      "y": int(rng.integers(90, H - plate.shape[0] - 70))})
    base_bg = make_background(rng, W + 64, H)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gt = {"fps": fps, "width": W, "height": H, "classes": CLASSES, "frames": []}
    for f in range(total):
        shift = int(f * belt_speed * 0.0) % 64
        frame = base_bg[:, shift:shift + W].copy()
        # moving belt ticks
        for k in range(-1, W // 80 + 2):
            x = int((k * 80 + f * belt_speed) % (W + 80)) - 40
            cv2.line(frame, (x, 0), (x, 22), (70, 70, 74), 3)
            cv2.line(frame, (x, H - 22), (x, H), (70, 70, 74), 3)
        frame_boxes = []
        for pi, p in enumerate(parts):
            x = int(p["x0"] + f * belt_speed)
            h, w = p["img"].shape[:2]
            if x + w <= 0 or x >= W:
                continue
            x1, x2 = max(x, 0), min(x + w, W)
            frame[p["y"]:p["y"] + h, x1:x2] = p["img"][:, x1 - x:x2 - x]
            for c, bx1, by1, bx2, by2 in p["boxes"]:
                gx1, gx2 = max(bx1 + x, 0), min(bx2 + x, W)
                if (gx2 - gx1) >= 0.6 * (bx2 - bx1):
                    frame_boxes.append({"cls": CLASSES[int(c)], "box": [float(gx1), float(by1 + p["y"]), float(gx2), float(by2 + p["y"])], "part": pi})
        # camera effects
        frame = cv2.GaussianBlur(frame, (0, 0), 0.6)
        frame = np.clip(frame.astype(np.float32) + rng.normal(0, 2.2, frame.shape), 0, 255).astype(np.uint8)
        vw.write(frame)
        gt["frames"].append(frame_boxes)
    vw.release()
    out_path.with_suffix(".gt.json").write_text(json.dumps(gt))
    return {"path": str(out_path), "frames": total, "fps": fps, "parts": n_parts}
