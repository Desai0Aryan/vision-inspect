"""Write the synthetic benchmark to disk in YOLO format (images/, labels/, data.yaml)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from visioninspect import CLASSES
from visioninspect.data.synth import render_sample


def build_dataset(out_dir: str | Path, n_train=600, n_val=150, n_test=150, size=640, seed=42, progress=None) -> dict:
    out = Path(out_dir)
    splits = {"train": n_train, "val": n_val, "test": n_test}
    stats = {"splits": {}, "instances_per_class": {c: 0 for c in CLASSES}}
    for s, (split, n) in enumerate(splits.items()):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed * 1000 + s)  # independent stream per split -> no leakage
        for i in range(n):
            img, labels = render_sample(rng, size)
            name = f"{split}_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for x1, y1, x2, y2, c in labels:
                cx, cy = (x1 + x2) / 2 / size, (y1 + y2) / 2 / size
                bw, bh = (x2 - x1) / size, (y2 - y1) / size
                lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stats["instances_per_class"][CLASSES[int(c)]] += 1
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            if progress:
                progress(split, i + 1, n)
        stats["splits"][split] = n
    data_yaml = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
                 "names": {i: c for i, c in enumerate(CLASSES)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_split(data_dir: str | Path, split: str):
    """Yield (path, image BGR, gt [N,5] xyxy px + cls) for a split."""
    d = Path(data_dir)
    for p in sorted((d / "images" / split).glob("*.jpg")):
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        lab = d / "labels" / split / f"{p.stem}.txt"
        rows = []
        if lab.exists() and lab.read_text().strip():
            for ln in lab.read_text().strip().splitlines():
                c, cx, cy, bw, bh = map(float, ln.split())
                rows.append([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H, c])
        yield p, img, np.array(rows, np.float32).reshape(-1, 5)
