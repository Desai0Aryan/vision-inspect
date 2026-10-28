"""Video inspection pipeline: detect -> track -> annotate -> collect statistics.

``process_video`` is a generator-friendly core used by the uploaded-clip job runner and
by the live MJPEG endpoint. It never holds the whole video in memory.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from visioninspect import CLASSES, CLASS_COLORS_BGR
from visioninspect.tracking.tracker import DefectTracker, PartTracker

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    FFMPEG = "ffmpeg"


@dataclass
class InspectConfig:
    conf: float = 0.45
    iou: float = 0.6
    frame_stride: int = 1           # run the detector every N frames (tracker coasts in between)
    min_hits: int = 4
    max_snapshots: int = 12
    alert_defect_rate: float = 0.30  # alert if more than this share of parts is rejected
    draw_hud: bool = True


@dataclass
class FrameStat:
    idx: int
    latency_ms: float | None
    n_det: int


def _txt(img, s, org, col=(255, 255, 255), scale=0.5, thick=1, bg=True):
    (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if bg:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 3, y + b), (20, 20, 20), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, dets, track_ids=None, conf_thr: float = 0.0) -> np.ndarray:
    for k, d in enumerate(dets):
        x1, y1, x2, y2, conf, c = d
        if conf < conf_thr:
            continue
        name = CLASSES[int(c)]
        col = CLASS_COLORS_BGR[name]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2, cv2.LINE_AA)
        label = f"{name} {conf:.2f}" if track_ids is None else f"#{track_ids[k]} {name} {conf:.2f}"
        _txt(frame, label, (int(x1), max(14, int(y1) - 5)), col, 0.45)
    return frame


def draw_hud(frame, fps_proc, latency_ms, n_parts, n_rej, n_defects, model, alert=False):
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (W, 34), (24, 24, 28), -1)
    _txt(frame, f"{model.upper()}", (10, 23), (120, 200, 255), 0.55, 1, bg=False)
    _txt(frame, f"{fps_proc:5.1f} FPS", (110, 23), (255, 255, 255), 0.55, 1, bg=False)
    _txt(frame, f"{latency_ms:5.1f} ms", (215, 23), (255, 255, 255), 0.55, 1, bg=False)
    _txt(frame, f"parts {n_parts}", (320, 23), (255, 255, 255), 0.55, 1, bg=False)
    _txt(frame, f"rejected {n_rej}", (410, 23), (80, 80, 255) if n_rej else (255, 255, 255), 0.55, 1, bg=False)
    _txt(frame, f"defects {n_defects}", (530, 23), (255, 255, 255), 0.55, 1, bg=False)
    if alert:
        cv2.rectangle(frame, (W - 170, 4), (W - 6, 30), (40, 40, 220), -1)
        _txt(frame, "DEFECT RATE ALERT", (W - 164, 22), (255, 255, 255), 0.45, 1, bg=False)
    return frame


def transcode_h264(src: Path, dst: Path) -> bool:
    """Re-encode to browser-playable H.264 (cv2's mp4v is not)."""
    try:
        r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], capture_output=True, timeout=600)
        return r.returncode == 0 and dst.exists()
    except Exception:
        return False


def process_video(src: str | Path, out_dir: str | Path, detector, model_name: str, cfg: InspectConfig | None = None,
                  progress=None, cancel=None) -> dict:
    cfg = cfg or InspectConfig()
    out_dir = Path(out_dir)
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError("could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    raw = out_dir / "annotated_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    dtrack = DefectTracker(min_hits=cfg.min_hits)
    ptrack = PartTracker(W)
    frame_stats: list[dict] = []
    events: list[dict] = []          # one row per (frame, detection)
    snapshots: dict[int, dict] = {}  # track id -> best crop info
    best_frames: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
    lat_all: list[float] = []
    last_dets = np.zeros((0, 6), np.float32)
    t_start = time.perf_counter()
    idx = 0
    ema_fps, ema_lat = 0.0, 0.0
    stamps: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cancel and cancel():
            break
        t0 = time.perf_counter()
        ran = idx % cfg.frame_stride == 0
        if ran:
            ti = time.perf_counter()
            raw_d = detector.predict_batch([frame])[0]
            lat = (time.perf_counter() - ti) * 1000
            lat_all.append(lat)
            last_dets = raw_d[raw_d[:, 4] >= cfg.conf] if len(raw_d) else raw_d
            ema_lat = lat if ema_lat == 0 else 0.9 * ema_lat + 0.1 * lat
        dets = last_dets
        confirmed = dtrack.update(dets, idx) if ran else [t for t in dtrack.tracks if t.hits >= cfg.min_hits]
        ptrack.update(frame, idx)
        ptrack.assign_defects(confirmed)

        # per-detection event log + best snapshot per confirmed track
        for t in confirmed:
            if ran and t.last_frame == idx:
                events.append({"frame": idx, "time_s": round(idx / fps, 3), "track_id": t.id, "class": CLASSES[t.cls],
                               "confidence": round(t.last_conf, 4),
                               "x1": round(float(t.box[0]), 1), "y1": round(float(t.box[1]), 1),
                               "x2": round(float(t.box[2]), 1), "y2": round(float(t.box[3]), 1)})
            if t.best_frame == idx and t.best_box is not None:
                best_frames[t.id] = (t.conf_max, frame.copy(), t.best_box.copy())

        vis = frame.copy()
        tr_boxes = np.array([[*t.box, t.mean_conf, t.cls] for t in confirmed], np.float32).reshape(-1, 6)
        draw_detections(vis, tr_boxes, [t.id for t in confirmed])
        for p in ptrack.parts:
            col = (60, 60, 255) if p.rejected else (80, 200, 80)
            cv2.rectangle(vis, (int(p.box[0]), int(p.box[1])), (int(p.box[2]), int(p.box[3])), col, 1, cv2.LINE_AA)
            _txt(vis, f"PART {p.id} {'NOK' if p.rejected else 'OK'}", (int(p.box[0]), int(p.box[3]) + 16), col, 0.45)
        n_parts = len(ptrack.all_parts())
        n_rej = sum(1 for p in ptrack.all_parts() if p.rejected)
        alert = n_parts >= 3 and n_rej / n_parts > cfg.alert_defect_rate
        dt = time.perf_counter() - t0
        stamps.append(time.perf_counter())
        stamps = stamps[-30:]
        ema_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 4 else 0.0
        if cfg.draw_hud:
            draw_hud(vis, ema_fps, ema_lat, n_parts, n_rej, len(dtrack.confirmed()), model_name, alert)
        vw.write(vis)
        frame_stats.append({"frame": idx, "latency_ms": round(lat, 2) if ran else None, "n_det": int(len(dets))})
        if progress and (idx % 5 == 0):
            progress({"frame": idx, "total": total, "fps": round(ema_fps, 1), "latency_ms": round(ema_lat, 1),
                      "parts": n_parts, "rejected": n_rej, "defects": len(dtrack.confirmed()),
                      "preview": _jpeg_small(vis)})
        idx += 1
    cap.release()
    vw.release()
    wall = time.perf_counter() - t_start

    # finalize
    for t in dtrack.tracks:
        dtrack.finished.append(t)
    tracks = sorted(dtrack.confirmed(), key=lambda t: t.first_frame)
    parts = sorted(ptrack.all_parts(), key=lambda p: p.first_frame)
    snaps = []
    for t in sorted(tracks, key=lambda t: -t.conf_max)[:cfg.max_snapshots]:
        if t.id in best_frames:
            _, fr, bx = best_frames[t.id]
            x1, y1, x2, y2 = [int(v) for v in bx]
            pad = 40
            crop = fr[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)].copy()
            cv2.rectangle(crop, (pad if x1 >= pad else x1, pad if y1 >= pad else y1),
                          ((pad if x1 >= pad else x1) + (x2 - x1), (pad if y1 >= pad else y1) + (y2 - y1)), CLASS_COLORS_BGR[CLASSES[t.cls]], 2)
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            name = f"track_{t.id:03d}.jpg"
            cv2.imwrite(str(out_dir / "snapshots" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            snaps.append(name)

    h264 = out_dir / "annotated.mp4"
    ok = transcode_h264(raw, h264)
    if not ok:
        raw.replace(h264)
    else:
        raw.unlink(missing_ok=True)

    track_to_part = {}
    for p in parts:
        for tid in p.defect_ids:
            track_to_part[tid] = p.id
    cls_counts = {c: 0 for c in CLASSES}
    for t in tracks:
        cls_counts[CLASSES[t.cls]] += 1
    lat = np.array(lat_all) if lat_all else np.zeros(1)
    n_parts, n_rej = len(parts), sum(1 for p in parts if p.rejected)
    summary = {
        "model": model_name,
        "video": {"frames": idx, "fps": round(fps, 2), "width": W, "height": H, "duration_s": round(idx / fps, 2)},
        "config": cfg.__dict__,
        "parts": {"inspected": n_parts, "rejected": n_rej, "passed": n_parts - n_rej,
                  "yield_pct": round(100 * (n_parts - n_rej) / n_parts, 1) if n_parts else None,
                  "defect_rate_pct": round(100 * n_rej / n_parts, 1) if n_parts else None},
        "defects": {"unique": len(tracks), "by_class": cls_counts},
        "performance": {"frames_inferred": len(lat_all), "latency_ms_mean": round(float(lat.mean()), 2),
                        "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
                        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
                        "latency_ms_max": round(float(lat.max()), 2),
                        "inference_fps": round(1000 / float(lat.mean()), 1),
                        "end_to_end_fps": round(idx / wall, 1) if wall else None, "wall_s": round(wall, 2),
                        "realtime_factor": round((idx / fps) / wall, 2) if wall else None},
        "alert": {"triggered": bool(n_parts >= 3 and n_rej / max(n_parts, 1) > cfg.alert_defect_rate),
                  "threshold_pct": cfg.alert_defect_rate * 100},
        "tracks": [{"id": t.id, "class": CLASSES[t.cls], "mean_conf": round(t.mean_conf, 3), "max_conf": round(t.conf_max, 3),
                    "first_frame": t.first_frame, "last_frame": t.last_frame, "hits": t.hits,
                    "first_time_s": round(t.first_frame / fps, 2), "part_id": track_to_part.get(t.id),
                    "snapshot": f"track_{t.id:03d}.jpg" if f"track_{t.id:03d}.jpg" in snaps else None} for t in tracks],
        "parts_detail": [{"id": p.id, "verdict": "NOK" if p.rejected else "OK", "defects": len(p.defect_ids),
                          "first_frame": p.first_frame, "last_frame": p.last_frame, "seen_whole": p.seen_whole} for p in parts],
        "snapshots": snaps,
    }
    return {"summary": summary, "events": events, "frame_stats": frame_stats}


def _jpeg_small(img, w: int = 480) -> bytes:
    h = int(img.shape[0] * w / img.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, h)), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()
