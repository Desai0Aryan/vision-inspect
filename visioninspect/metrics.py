"""Detection metrics implemented from scratch (COCO-style 101-point AP).

Conventions
-----------
pred:  float array [N, 6] -> x1, y1, x2, y2, confidence, class
gt:    float array [M, 5] -> x1, y1, x2, y2, class
"""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a [N,4] and b [M,4] (xyxy)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP (COCO)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, x, side="left")
    return float(np.mean(mpre[np.clip(idx, 0, len(mpre) - 1)]))


def _match_class(preds, gts, cls, iou_thr):
    """Greedy match for one class. Returns (conf, tp flags, n_gt)."""
    confs, tps = [], []
    n_gt = 0
    for p, g in zip(preds, gts):
        pc = p[p[:, 5] == cls] if len(p) else p.reshape(0, 6)
        gc = g[g[:, 4] == cls] if len(g) else g.reshape(0, 5)
        n_gt += len(gc)
        if len(pc) == 0:
            continue
        order = np.argsort(-pc[:, 4])
        pc = pc[order]
        ious = iou_matrix(pc[:, :4], gc[:, :4])
        taken = np.zeros(len(gc), bool)
        for i in range(len(pc)):
            tp = False
            if len(gc):
                j = int(np.argmax(np.where(taken, -1, ious[i])))
                if ious[i, j] >= iou_thr and not taken[j]:
                    taken[j] = True
                    tp = True
            confs.append(pc[i, 4])
            tps.append(tp)
    return np.array(confs), np.array(tps, bool), n_gt


def average_precision(preds, gts, cls, iou_thr=0.5):
    conf, tp, n_gt = _match_class(preds, gts, cls, iou_thr)
    if n_gt == 0:
        return float("nan"), np.array([]), np.array([]), np.array([])
    if len(conf) == 0:
        return 0.0, np.array([]), np.array([]), np.array([])
    order = np.argsort(-conf)
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1)
    return _ap_from_pr(recall, precision), recall, precision, conf[order]


def evaluate(preds, gts, class_names, conf_thr: float = 0.25) -> dict:
    """Full detection report.

    Returns mAP@0.5, mAP@0.5:0.95, per-class AP, and precision / recall / F1
    at ``conf_thr`` (IoU 0.5), plus image-level defect/OK classification metrics.
    """
    ious = np.arange(0.5, 0.96, 0.05)
    nc = len(class_names)
    per_class = {}
    ap50, ap5095 = [], []
    for c in range(nc):
        aps = [average_precision(preds, gts, c, t)[0] for t in ious]
        if np.isnan(aps[0]):
            continue
        per_class[class_names[c]] = {"AP50": float(aps[0]), "AP50_95": float(np.mean(aps))}
        ap50.append(aps[0])
        ap5095.append(np.mean(aps))

    # P / R / F1 at the operating point
    tp_tot = fp_tot = fn_tot = 0
    for c in range(nc):
        filt = [p[p[:, 4] >= conf_thr] if len(p) else p for p in preds]
        conf, tp, n_gt = _match_class(filt, gts, c, 0.5)
        tp_tot += int(tp.sum())
        fp_tot += int((~tp).sum())
        fn_tot += n_gt - int(tp.sum())
        if class_names[c] in per_class:
            tpc, fpc = int(tp.sum()), int((~tp).sum())
            per_class[class_names[c]].update(
                precision=tpc / max(tpc + fpc, 1), recall=tpc / max(n_gt, 1), n_gt=n_gt)
    precision = tp_tot / max(tp_tot + fp_tot, 1)
    recall = tp_tot / max(tp_tot + fn_tot, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # image-level: does the model flag a defective part at all?
    y_true = np.array([len(g) > 0 for g in gts], dtype=bool)
    y_pred = np.array([bool(len(p) and (p[:, 4] >= conf_thr).any()) for p in preds], dtype=bool)
    itp = int((y_true & y_pred).sum())
    ifp = int((~y_true & y_pred).sum())
    ifn = int((y_true & ~y_pred).sum())
    itn = int((~y_true & ~y_pred).sum())

    return {
        "mAP50": float(np.mean(ap50)) if ap50 else 0.0,
        "mAP50_95": float(np.mean(ap5095)) if ap5095 else 0.0,
        "precision": precision, "recall": recall, "f1": f1, "conf_thr": conf_thr,
        "per_class": per_class,
        "image_level": {
            "accuracy": (itp + itn) / max(len(gts), 1),
            "precision": itp / max(itp + ifp, 1),
            "recall": itp / max(itp + ifn, 1),
            "confusion": {"tp": itp, "fp": ifp, "fn": ifn, "tn": itn},
        },
    }
