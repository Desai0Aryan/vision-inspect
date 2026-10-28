import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["VI_JOBS_DIR"] = tempfile.mkdtemp()

pytest.importorskip("fastapi")
if not (ROOT / "models" / "yolo_int8.onnx").exists():
    pytest.skip("trained models not present", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_sample_part_is_deterministic_and_matches_truth(client):
    a = client.get("/api/sample/7/image")
    b = client.get("/api/sample/7/image")
    assert a.status_code == 200 and a.content[:2] == b"\xff\xd8" and a.content == b.content
    t = client.get("/api/sample/7/truth").json()
    assert t["seed"] == 7 and all(d["class"] in ("scratch", "crack", "stain", "missing_part") and len(d["box"]) == 4 for d in t["defects"])
    assert client.get("/api/sample/8/image").content != a.content


def test_model_finds_defects_on_generated_parts(client):
    """End-to-end check of what the Single part page does: generated image in, defects out."""
    found = total = 0
    for seed in range(100, 108):
        truth = client.get(f"/api/sample/{seed}/truth").json()["defects"]
        img = client.get(f"/api/sample/{seed}/image").content
        r = client.post("/api/predict/image", files={"file": ("p.jpg", img, "image/jpeg")}, data={"annotate": "false"}).json()
        total += len(truth)
        found += min(len(truth), len(r["detections"]))
        assert r["defective"] == (len(r["detections"]) > 0)
    assert total > 0 and found / total > 0.8


@pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(), reason="frontend not built")
def test_built_dashboard_is_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200 and "VisionInspect" in r.text
    assert client.get("/api/health").json()["status"] == "ok"   # API routes still win over the static mount
