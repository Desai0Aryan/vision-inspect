import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["VI_JOBS_DIR"] = tempfile.mkdtemp()

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
if not (ROOT / "models" / "yolo_int8.onnx").exists():
    pytest.skip("trained models not present", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from visioninspect.data.synth import render_sample  # noqa: E402
from visioninspect.data.video import make_demo_video  # noqa: E402
from visioninspect.tracking.tracker import DefectTracker  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "clip.mp4"
    make_demo_video(p, seconds=5, n_parts=2, seed=3)
    return p


def test_health_and_models(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "int8" in h["models"]
    assert client.get("/api/models").json()["default"] == "int8"
    assert "visioninspect_images_total" in client.get("/metrics").text


def test_predict_image(client):
    img, labels = render_sample(np.random.default_rng(5), spec=None)
    ok, buf = cv2.imencode(".jpg", img)
    r = client.post("/api/predict/image", files={"file": ("a.jpg", buf.tobytes(), "image/jpeg")}, data={"conf": "0.4"})
    assert r.status_code == 200
    j = r.json()
    assert j["latency_ms"] > 0 and isinstance(j["detections"], list) and j["annotated_jpeg_b64"]
    assert j["defective"] == (len(labels) > 0) or len(labels) == 0


def test_bad_inputs_rejected(client):
    assert client.post("/api/predict/image", files={"file": ("a.jpg", b"not an image", "image/jpeg")}).status_code == 400
    assert client.post("/api/jobs", files={"file": ("a.mp4", b"garbage", "video/mp4")}).status_code == 400
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/../etc").status_code in (404, 400, 422)


def test_job_lifecycle_and_reports(client, clip):
    r = client.post("/api/jobs", files={"file": ("clip.mp4", clip.read_bytes(), "video/mp4")}, data={"frame_stride": "2"})
    assert r.status_code == 202
    jid = r.json()["id"]
    for _ in range(120):
        s = client.get(f"/api/jobs/{jid}").json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(1)
    assert s["status"] == "done", s.get("error")
    sm = s["summary"]
    assert sm["video"]["frames"] > 100 and sm["parts"]["inspected"] >= 1
    assert client.get(f"/api/jobs/{jid}/video").headers["content-type"] == "video/mp4"
    assert client.get(f"/api/jobs/{jid}/report.pdf").content[:4] == b"%PDF"
    assert client.get(f"/api/jobs/{jid}/report.csv").text.startswith("frame,time_s,track_id")
    assert "summary" in client.get(f"/api/jobs/{jid}/report.json").json()
    assert client.delete(f"/api/jobs/{jid}").json()["status"] == "deleted"


def test_websocket_live(client):
    img, _ = render_sample(np.random.default_rng(9))
    buf = cv2.imencode(".jpg", img)[1].tobytes()
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(buf)
        msg = ws.receive_json()
    assert msg["latency_ms"] > 0 and "annotated_b64" in msg


def test_tracker_counts_one_defect_once():
    tr = DefectTracker(min_hits=3)
    for f in range(40):
        d = np.array([[100 + f * 3, 50, 140 + f * 3, 90, 0.9, 1]], np.float32)   # moves 3 px / frame
        tr.update(d, f)
    conf = tr.confirmed()
    assert len(conf) == 1 and conf[0].hits == 40 and conf[0].cls == 1


def test_tracker_separates_two_defects_and_survives_missed_frames():
    tr = DefectTracker(min_hits=3, max_missed=8)
    for f in range(30):
        dets = [[10, 10, 40, 40, 0.9, 0], [300, 300, 340, 340, 0.8, 2]]
        if f in (10, 11, 12):          # detector drops the first defect for 3 frames
            dets = dets[1:]
        tr.update(np.array(dets, np.float32), f)
    ids = {t.id for t in tr.confirmed()}
    assert len(ids) == 2
