"""Evaluate the classical baseline on a split and write a JSON report + preview grid.

    python scripts/eval_baseline.py --data data/inspect --split test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visioninspect import CLASSES, CLASS_COLORS_BGR  # noqa: E402
from visioninspect.baselines.classical import ClassicalDetector  # noqa: E402
from visioninspect.data.dataset import load_split  # noqa: E402
from visioninspect.metrics import evaluate  # noqa: E402


def draw(img, dets, gts=None):
    out = img.copy()
    if gts is not None:
        for x1, y1, x2, y2, c in gts:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1, cv2.LINE_AA)
    for x1, y1, x2, y2, conf, c in dets:
        name = CLASSES[int(c)]
        col = CLASS_COLORS_BGR[name]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2, cv2.LINE_AA)
        cv2.putText(out, f"{name} {conf:.2f}", (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inspect")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/baseline")
    ap.add_argument("--conf", type=float, default=0.5)
    a = ap.parse_args()

    det = ClassicalDetector()
    preds, gts, lat, shown = [], [], [], []
    for path, img, gt in load_split(a.data, a.split):
        t = time.perf_counter()
        p = det.predict(img)
        lat.append((time.perf_counter() - t) * 1000)
        preds.append(p)
        gts.append(gt)
        if len(shown) < 8 and len(gt):
            shown.append(draw(img, p, gt))

    rep = evaluate(preds, gts, CLASSES, conf_thr=a.conf)
    rep["model"] = det.name
    rep["latency_ms"] = {"mean": float(np.mean(lat)), "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
    rep["fps"] = 1000.0 / rep["latency_ms"]["mean"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics_{a.split}.json").write_text(json.dumps(rep, indent=2))
    if shown:
        small = [cv2.resize(s, (480, 480)) for s in shown]
        rows = [np.hstack(small[i:i + 4]) for i in range(0, len(small) - len(small) % 4, 4)]
        if rows:
            cv2.imwrite(str(out / "preview.jpg"), np.vstack(rows))

    print(f"model={det.name} split={a.split} n={len(gts)}")
    print(f"  mAP@0.5={rep['mAP50']:.3f}  mAP@0.5:0.95={rep['mAP50_95']:.3f}  P={rep['precision']:.3f}  R={rep['recall']:.3f}  F1={rep['f1']:.3f}")
    for k, v in rep["per_class"].items():
        print(f"  {k:13s} AP50={v['AP50']:.3f}  P={v['precision']:.2f} R={v['recall']:.2f}  (n={v['n_gt']})")
    print(f"  image-level: {rep['image_level']}")
    print(f"  latency mean={rep['latency_ms']['mean']:.1f} ms  p95={rep['latency_ms']['p95']:.1f} ms  ({rep['fps']:.0f} FPS)")


if __name__ == "__main__":
    main()
