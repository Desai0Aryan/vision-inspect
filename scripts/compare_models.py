"""Evaluate several detectors with the SAME evaluator and produce a comparison report.

    python scripts/compare_models.py --split test --yolo models/yolo_pretrained.pt --yolo-scratch models/yolo_scratch.pt

Outputs (outputs/compare/): results.json, comparison.md, bar chart, PR curves,
per-class AP chart, confusion-style error analysis, and a qualitative grid.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visioninspect import CLASSES, CLASS_COLORS_BGR  # noqa: E402
from visioninspect.baselines.classical import ClassicalDetector  # noqa: E402
from visioninspect.data.dataset import load_split  # noqa: E402
from visioninspect.metrics import average_precision, evaluate, iou_matrix  # noqa: E402

PALETTE = {"classical": "#8a8f98", "yolo_pretrained": "#2f81f7", "yolo_scratch": "#e3873b"}


def run_model(det, items, warm=3):
    for _, img, _ in items[:warm]:
        det.predict(img)
    preds, lat = [], []
    for _, img, _ in items:
        t = time.perf_counter()
        preds.append(det.predict(img))
        lat.append((time.perf_counter() - t) * 1000)
    return preds, lat


def best_conf(preds, gts, grid=np.arange(0.1, 0.9, 0.05)):
    """Operating point that maximises F1 on the evaluated split's *val* data."""
    best, bt = -1, 0.25
    for t in grid:
        f = evaluate(preds, gts, CLASSES, conf_thr=float(t))["f1"]
        if f > best:
            best, bt = f, float(t)
    return bt


def confusion(preds, gts, thr, iou_thr=0.5):
    """Class confusion among localised defects (rows=truth, cols=pred, last col=missed)."""
    n = len(CLASSES)
    M = np.zeros((n, n + 1), int)
    for p, g in zip(preds, gts):
        p = p[p[:, 4] >= thr] if len(p) else p
        if len(g) == 0:
            continue
        ious = iou_matrix(g[:, :4], p[:, :4]) if len(p) else np.zeros((len(g), 0))
        for i, gt in enumerate(g):
            if ious.shape[1] and ious[i].max() >= iou_thr:
                j = int(np.argmax(ious[i]))
                M[int(gt[4]), int(p[j, 5])] += 1
            else:
                M[int(gt[4]), n] += 1
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inspect")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tune-split", default="val", help="split used to pick each model's confidence threshold")
    ap.add_argument("--yolo", default=None)
    ap.add_argument("--yolo-scratch", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="outputs/compare")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    test = list(load_split(a.data, a.split))
    tune = list(load_split(a.data, a.tune_split))
    gts = [g for _, _, g in test]
    tgts = [g for _, _, g in tune]

    models = {"classical": ClassicalDetector()}
    if a.yolo:
        from visioninspect.models.yolo import YoloDetector
        models["yolo_pretrained"] = YoloDetector(a.yolo, imgsz=a.imgsz)
    if a.yolo_scratch:
        from visioninspect.models.yolo import YoloDetector
        models["yolo_scratch"] = YoloDetector(a.yolo_scratch, imgsz=a.imgsz)

    results, all_preds, thrs = {}, {}, {}
    for key, det in models.items():
        print(f"[{key}] tuning threshold on {a.tune_split} ...", flush=True)
        tp, _ = run_model(det, tune)
        thr = best_conf(tp, tgts)
        print(f"[{key}] evaluating on {a.split} (thr={thr:.2f}) ...", flush=True)
        preds, lat = run_model(det, test)
        rep = evaluate(preds, gts, CLASSES, conf_thr=thr)
        rep["latency_ms"] = {"mean": float(np.mean(lat)), "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
        rep["fps"] = 1000 / rep["latency_ms"]["mean"]
        rep["confusion"] = confusion(preds, gts, thr).tolist()
        results[key], all_preds[key], thrs[key] = rep, preds, thr
        print(f"   mAP50={rep['mAP50']:.3f} mAP50-95={rep['mAP50_95']:.3f} P={rep['precision']:.3f} R={rep['recall']:.3f} "
              f"lat={rep['latency_ms']['mean']:.1f}ms", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    # ---------- charts
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    keys = list(results)
    cols = [PALETTE.get(k, "#555") for k in keys]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (m, title) in zip(axs, [("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95"), ("f1", "F1 @ tuned threshold")]):
        v = [results[k][m] for k in keys]
        b = ax.bar(keys, v, color=cols)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.bar_label(b, fmt="%.3f", padding=2)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out / "headline_metrics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        v = [results[k]["per_class"].get(c, {}).get("AP50", 0) for c in CLASSES]
        ax.bar(np.arange(len(CLASSES)) + i * w - 0.4 + w / 2, v, w, label=k, color=PALETTE.get(k))
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-class average precision")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "per_class_ap.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(CLASSES), figsize=(16, 3.6), sharey=True)
    for ci, c in enumerate(CLASSES):
        for k in keys:
            ap_, rec, prec, _ = average_precision(all_preds[k], gts, ci)
            if len(rec):
                axs[ci].plot(rec, prec, label=f"{k} ({ap_:.2f})", color=PALETTE.get(k), lw=2)
        axs[ci].set_title(c)
        axs[ci].set_xlabel("recall")
        axs[ci].set_xlim(0, 1)
        axs[ci].set_ylim(0, 1.02)
        axs[ci].legend(frameon=False, fontsize=8, loc="lower left")
    axs[0].set_ylabel("precision")
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.4))
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, keys):
        M = np.array(results[k]["confusion"])
        ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(len(CLASSES) + 1), CLASSES + ["missed"], rotation=30, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black")
        ax.set_title(f"{k}\n(truth rows, predicted cols)")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=150)
    plt.close(fig)

    # ---------- qualitative grid: same images, each model
    rng = np.random.default_rng(0)
    idxs = [i for i, g in enumerate(gts) if len(g) >= 2][:3]
    rows = []
    for i in idxs:
        cells = []
        for k in keys:
            img = test[i][1].copy()
            for x1, y1, x2, y2, c in gts[i]:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1, cv2.LINE_AA)
            for x1, y1, x2, y2, conf, c in all_preds[k][i]:
                if conf < thrs[k]:
                    continue
                col = CLASS_COLORS_BGR[CLASSES[int(c)]]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2, cv2.LINE_AA)
                cv2.putText(img, f"{CLASSES[int(c)]} {conf:.2f}", (int(x1), max(14, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cv2.putText(img, k, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(cv2.resize(img, (420, 420)))
        rows.append(np.hstack(cells))
    if rows:
        cv2.imwrite(str(out / "qualitative.jpg"), np.vstack(rows))

    # ---------- markdown report
    lines = ["# Model comparison", "", f"Split: `{a.split}` ({len(gts)} images). Confidence thresholds were tuned for max F1 on `{a.tune_split}`, never on test.", "",
             "| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Image-level acc | Latency (ms, CPU) | FPS | Threshold |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        r = results[k]
        lines.append(f"| {k} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                     f"{r['image_level']['accuracy']:.3f} | {r['latency_ms']['mean']:.1f} | {r['fps']:.1f} | {thrs[k]:.2f} |")
    lines += ["", "## Per-class AP@0.5", "", "| Model | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(f"{results[k]['per_class'].get(c, {}).get('AP50', float('nan')):.3f}" for c in CLASSES) + " |")
    (out / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
