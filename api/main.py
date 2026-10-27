"""VisionInspect API.

REST
  GET  /api/health                      liveness + model availability
  GET  /api/models                      available backends (int8 / fp32 / pytorch / classical)
  GET  /api/benchmarks                  results of Parts 2 and 3 (accuracy + latency)
  POST /api/predict/image               one image -> detections (+ annotated JPEG as base64)
  POST /api/jobs                        upload a clip, start background inspection  -> {job_id}
  GET  /api/jobs                        list jobs
  GET  /api/jobs/{id}                   status, progress, summary
  GET  /api/jobs/{id}/video             annotated H.264 video (Range supported)
  GET  /api/jobs/{id}/report.{json,csv,pdf}
  GET  /api/jobs/{id}/snapshots/{name}  best crop per defect
  DELETE /api/jobs/{id}
  GET  /api/stream/demo                 MJPEG: demo conveyor processed live
  GET  /metrics                         Prometheus-style counters
WebSocket
  /ws/jobs/{id}                         live progress + preview frames while a job runs
  /ws/live                              send JPEG frames (e.g. webcam), receive detections + latency
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from visioninspect import CLASSES, __version__
from visioninspect.live import LiveInspector
from visioninspect.pipeline import InspectConfig, draw_detections, process_video
from visioninspect.registry import available, default_model, get_detector
from visioninspect.reporting.report import write_csv, write_json, write_pdf

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = Path(os.environ.get("VI_JOBS_DIR", ROOT / "jobs"))
DEMO = Path(os.environ.get("VI_DEMO_VIDEO", ROOT / "data" / "demo" / "conveyor.mp4"))
OUTPUTS = ROOT / "outputs"
MAX_UPLOAD_MB = int(os.environ.get("VI_MAX_UPLOAD_MB", 200))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VisionInspect API", version=__version__,
              description="Real-time defect inspection: detect, localize, classify, track and report.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("VI_CORS", "*").split(","), allow_methods=["*"], allow_headers=["*"])

_pool = ThreadPoolExecutor(max_workers=1)      # one video at a time keeps latency numbers honest on small machines
_jobs: dict[str, dict] = {}
_cancel: dict[str, bool] = {}
_lock = threading.Lock()
_counters = {"images": 0, "jobs": 0, "frames": 0, "defects": 0}


# ----------------------------------------------------------------------------- helpers
def _job_dir(job_id: str) -> Path:
    if not job_id.replace("-", "").isalnum():
        raise HTTPException(400, "bad job id")
    return JOBS_DIR / job_id


def _public(j: dict) -> dict:
    return {k: v for k, v in j.items() if k != "preview"}


def _load_existing_jobs():
    for d in JOBS_DIR.iterdir():
        meta = d / "job.json"
        if meta.exists():
            try:
                j = json.loads(meta.read_text())
                if j.get("status") in ("queued", "running"):
                    j["status"] = "failed"
                    j["error"] = "server restarted"
                _jobs[j["id"]] = j
            except Exception:
                pass


def _save(j: dict):
    (JOBS_DIR / j["id"] / "job.json").write_text(json.dumps(_public(j)))


def _run_job(job_id: str):
    j = _jobs[job_id]
    d = JOBS_DIR / job_id
    j.update(status="running", started_at=time.time())
    try:
        det = get_detector(j["model"])
        cfg = InspectConfig(conf=j["conf"], frame_stride=j["frame_stride"], alert_defect_rate=j["alert_rate"])

        def progress(p):
            j["progress"] = {k: v for k, v in p.items() if k != "preview"}
            j["preview"] = p["preview"]

        res = process_video(d / "input.mp4", d, det, j["model"], cfg, progress=progress, cancel=lambda: _cancel.get(job_id, False))
        if _cancel.get(job_id):
            j.update(status="cancelled")
        else:
            write_json(d, res["summary"], res["events"], res["frame_stats"])
            write_csv(d, res["events"])
            write_pdf(d, res["summary"], res["frame_stats"], "Inspection report", j.get("filename", ""))
            j.update(status="done", summary=res["summary"], finished_at=time.time())
            with _lock:
                _counters["jobs"] += 1
                _counters["frames"] += res["summary"]["video"]["frames"]
                _counters["defects"] += res["summary"]["defects"]["unique"]
    except Exception as e:  # noqa: BLE001
        j.update(status="failed", error=str(e))
    finally:
        j["preview"] = None
        _save(j)


def _decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    return img


def _b64jpg(img, q=85) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()


@app.on_event("startup")
def _startup():
    _load_existing_jobs()
    try:  # warm the default model so the first request is not slow
        d = get_detector(default_model())
        if hasattr(d, "warmup"):
            d.warmup(2)
    except Exception:
        pass


# ----------------------------------------------------------------------------- meta
@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "default_model": default_model(), "models": [m["id"] for m in available()],
            "classes": CLASSES, "jobs": len(_jobs)}


@app.get("/api/models")
def models():
    return {"default": default_model(), "models": available()}


@app.get("/api/benchmarks")
def benchmarks():
    out = {}
    for name, rel in [("optimization", "benchmark/benchmark.json"), ("comparison", "compare/results.json"), ("video", "video_eval/video_eval_int8.json")]:
        p = OUTPUTS / rel
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [f"visioninspect_{k}_total {v}" for k, v in _counters.items()]
    lines.append(f"visioninspect_jobs_active {sum(1 for j in _jobs.values() if j['status'] in ('queued', 'running'))}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- single image
@app.post("/api/predict/image")
async def predict_image(file: UploadFile = File(...), model: str | None = Form(None), conf: float = Form(0.45), annotate: bool = Form(True)):
    img = _decode_image(await file.read())
    try:
        det = get_detector(model)
    except KeyError:
        raise HTTPException(404, f"unknown model {model}")
    t = time.perf_counter()
    raw = await asyncio.get_running_loop().run_in_executor(None, lambda: det.predict_batch([img])[0])
    lat = (time.perf_counter() - t) * 1000
    dets = raw[raw[:, 4] >= conf] if len(raw) else raw
    with _lock:
        _counters["images"] += 1
    res = {"model": model or default_model(), "latency_ms": round(lat, 2), "width": img.shape[1], "height": img.shape[0],
           "defective": bool(len(dets)),
           "detections": [{"class": CLASSES[int(d[5])], "confidence": round(float(d[4]), 4), "box": [round(float(v), 1) for v in d[:4]]} for d in dets]}
    if annotate:
        res["annotated_jpeg_b64"] = _b64jpg(draw_detections(img.copy(), dets))
    return res


# ----------------------------------------------------------------------------- video jobs
@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile = File(...), model: str | None = Form(None), conf: float = Form(0.45),
                     frame_stride: int = Form(1), alert_rate: float = Form(0.30)):
    model = model or default_model()
    if model not in {m["id"] for m in available()}:
        raise HTTPException(404, f"unknown model {model}")
    if not (0 < conf < 1) or not (1 <= frame_stride <= 10):
        raise HTTPException(422, "conf must be in (0,1) and frame_stride in [1,10]")
    job_id = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job_id
    d.mkdir(parents=True)
    size = 0
    with open(d / "input.mp4", "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1 << 20:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    cap = cv2.VideoCapture(str(d / "input.mp4"))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "not a readable video file")
    j = {"id": job_id, "status": "queued", "filename": file.filename, "model": model, "conf": conf, "frame_stride": frame_stride,
         "alert_rate": alert_rate, "created_at": time.time(), "progress": {"frame": 0, "total": 0}, "summary": None, "error": None, "preview": None}
    _jobs[job_id] = j
    _save(j)
    _pool.submit(_run_job, job_id)
    return _public(j)


@app.post("/api/jobs/demo", status_code=202)
async def create_demo_job(model: str | None = Form(None), conf: float = Form(0.45), frame_stride: int = Form(1)):
    """Run the built-in conveyor clip so the demo works without uploading anything."""
    if not DEMO.exists():
        raise HTTPException(404, "demo clip not found")
    model = model or default_model()
    job_id = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job_id
    d.mkdir(parents=True)
    shutil.copy(DEMO, d / "input.mp4")
    j = {"id": job_id, "status": "queued", "filename": DEMO.name, "model": model, "conf": conf, "frame_stride": frame_stride,
         "alert_rate": 0.30, "created_at": time.time(), "progress": {"frame": 0, "total": 0}, "summary": None, "error": None, "preview": None}
    _jobs[job_id] = j
    _save(j)
    _pool.submit(_run_job, job_id)
    return _public(j)


@app.get("/api/jobs")
def list_jobs():
    return sorted((_public(j) | {"summary": None} for j in _jobs.values()), key=lambda j: -j["created_at"])


def _get(job_id: str) -> dict:
    j = _jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _public(_get(job_id))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    j = _get(job_id)
    if j["status"] in ("queued", "running"):
        _cancel[job_id] = True
        return {"status": "cancelling"}
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    _jobs.pop(job_id, None)
    return {"status": "deleted"}


def _file(job_id: str, name: str, media: str, filename: str | None = None):
    j = _get(job_id)
    p = _job_dir(job_id) / name
    if j["status"] != "done" or not p.exists():
        raise HTTPException(409 if j["status"] in ("queued", "running") else 404, f"{name} not available (job {j['status']})")
    return FileResponse(p, media_type=media, filename=filename)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    return _file(job_id, "annotated.mp4", "video/mp4")


@app.get("/api/jobs/{job_id}/report.json")
def report_json(job_id: str):
    return _file(job_id, "report.json", "application/json", f"inspection_{job_id}.json")


@app.get("/api/jobs/{job_id}/report.csv")
def report_csv(job_id: str):
    return _file(job_id, "report.csv", "text/csv", f"inspection_{job_id}.csv")


@app.get("/api/jobs/{job_id}/report.pdf")
def report_pdf(job_id: str):
    return _file(job_id, "report.pdf", "application/pdf", f"inspection_{job_id}.pdf")


@app.get("/api/jobs/{job_id}/snapshots/{name}")
def snapshot(job_id: str, name: str):
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    return _file(job_id, f"snapshots/{name}", "image/jpeg")


@app.websocket("/ws/jobs/{job_id}")
async def ws_job(ws: WebSocket, job_id: str):
    await ws.accept()
    j = _jobs.get(job_id)
    if not j:
        await ws.close(code=4404)
        return
    try:
        last = -1
        while True:
            p = dict(j.get("progress") or {})
            msg = {"status": j["status"], **p}
            if j.get("preview") and p.get("frame") != last:
                msg["preview_b64"] = base64.b64encode(j["preview"]).decode()
                last = p.get("frame")
            if j["status"] == "done":
                msg["summary"] = j["summary"]
            await ws.send_json(msg)
            if j["status"] in ("done", "failed", "cancelled"):
                break
            await asyncio.sleep(0.15)
        await ws.close()
    except WebSocketDisconnect:
        pass


# ----------------------------------------------------------------------------- live
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket, model: str | None = None, conf: float = 0.45):
    """Client sends binary JPEG frames; server answers with JSON (and optionally an annotated JPEG)."""
    await ws.accept()
    try:
        det = get_detector(model)
    except KeyError:
        await ws.close(code=4404)
        return
    insp = LiveInspector(det, model or default_model(), InspectConfig(conf=conf))
    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await ws.receive_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                await ws.send_json({"error": "bad frame"})
                continue
            out, info = await loop.run_in_executor(None, lambda: insp.step(img, draw=True))
            info["annotated_b64"] = _b64jpg(out, 75)
            await ws.send_json(info)
    except WebSocketDisconnect:
        pass


@app.get("/api/stream/demo")
def stream_demo(model: str | None = None, conf: float = 0.45, loop: bool = True):
    """MJPEG stream of the demo conveyor processed live. Open in an <img> tag."""
    if not DEMO.exists():
        raise HTTPException(404, "demo clip not found")
    try:
        det = get_detector(model)
    except KeyError:
        raise HTTPException(404, "unknown model")

    def gen():
        while True:
            cap = cv2.VideoCapture(str(DEMO))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            insp = LiveInspector(det, model or default_model(), InspectConfig(conf=conf))
            t_next = time.perf_counter()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out, _ = insp.step(frame)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes() + b"\r\n"
                t_next += 1 / fps
                time.sleep(max(0.0, t_next - time.perf_counter()))
            cap.release()
            if not loop:
                break

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
