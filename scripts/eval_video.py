"""Score the full video pipeline (detector + trackers) against the demo clip's ground truth.

    python scripts/eval_video.py --video data/demo/conveyor.mp4 --model int8

Reports frame-level mAP / P / R, plus the *system-level* numbers that matter on a line:
parts counted, per-part OK/NOK verdict accuracy, and unique-defect count vs truth
(tests whether the tracker avoids counting one defect many times).
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visioninspect import CLASSES  # noqa: E402
from visioninspect.metrics import evaluate  # noqa: E402
from visioninspect.pipeline import InspectConfig, process_video  # noqa: E402
from visioninspect.registry import get_detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/demo/conveyor.mp4")
    ap.add_argument("--model", default="int8")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--stride", type=int, default=1, help="run the detector every N frames")
    ap.add_argument("--tag", default="", help="suffix for the result file")
    ap.add_argument("--out", default="outputs/video_eval")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads(Path(a.video).with_suffix(".gt.json").read_text())

    det = get_detector(a.model)
    det.warmup(2) if hasattr(det, "warmup") else None
    res = process_video(a.video, tempfile.mkdtemp(), det, a.model, InspectConfig(conf=a.conf, frame_stride=a.stride))
    s = res["summary"]

    # --- frame-level detection quality using confirmed-track events
    ev_by_frame = {}
    for e in res["events"]:
        ev_by_frame.setdefault(e["frame"], []).append([e["x1"], e["y1"], e["x2"], e["y2"], e["confidence"], CLASSES.index(e["class"])])
    preds, gts = [], []
    for i, fr in enumerate(gt["frames"]):
        preds.append(np.array(ev_by_frame.get(i, []), np.float32).reshape(-1, 6))
        gts.append(np.array([[*f["box"], CLASSES.index(f["cls"])] for f in fr], np.float32).reshape(-1, 5))
    rep = evaluate(preds, gts, CLASSES, conf_thr=0.0)

    # --- system-level
    parts_gt = {}
    for fr in gt["frames"]:
        for f in fr:
            parts_gt.setdefault(f["part"], set()).add((f["cls"], round(f["box"][0] - 0)))  # coarse identity
    n_gt_parts = 6
    gt_defective = set(parts_gt)
    # unique GT defects: (part, class, order) approximated by distinct class/part pairs counted from frame 1st sightings
    uniq = set()
    for fr in gt["frames"]:
        for f in fr:
            uniq.add((f["part"], f["cls"], tuple(round(v / 40) for v in (f["box"][1], f["box"][3]))))
    pred_nok = {p["id"] for p in s["parts_detail"] if p["verdict"] == "NOK"}
    pred_parts = len(s["parts_detail"])
    # parts are numbered in order of appearance == generator order
    gt_nok = {i + 1 for i in gt_defective}
    verdict_ok = sum(1 for p in s["parts_detail"] if (p["verdict"] == "NOK") == (p["id"] in gt_nok))
    result = {
        "model": a.model, "video": a.video, "frame_stride": a.stride,
        "frame_level": {"mAP50": rep["mAP50"], "precision": rep["precision"], "recall": rep["recall"], "f1": rep["f1"]},
        "system_level": {"parts_counted": pred_parts, "parts_truth": n_gt_parts,
                         "verdict_accuracy": verdict_ok / max(pred_parts, 1),
                         "nok_pred": sorted(pred_nok), "nok_truth": sorted(gt_nok),
                         "unique_defects_pred": s["defects"]["unique"], "unique_defect_instances_truth_approx": len(uniq)},
        "performance": s["performance"],
    }
    (out / f"video_eval_{a.model}{a.tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
