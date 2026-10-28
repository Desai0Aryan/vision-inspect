"""Packaging checks: Dockerfile/compose/CI are coherent and the slim runtime stays torch-free."""
import glob
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
yaml = pytest.importorskip("yaml")


def test_dockerfile_copy_sources_exist():
    text = (ROOT / "Dockerfile").read_text()
    for line in text.splitlines():
        m = re.match(r"COPY\s+(?!--from)(.+)\s+\S+$", line.strip())
        if not m:
            continue
        for src in m.group(1).split():
            assert glob.glob(str(ROOT / src)), f"Dockerfile copies missing path: {src}"


def test_compose_and_ci_are_valid_yaml():
    comp = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert "app" in comp["services"] and comp["services"]["app"]["ports"] == ["8000:8000"]
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    assert {"python", "frontend", "docker"} <= set(ci["jobs"])


def test_runtime_requirements_have_no_torch():
    reqs = "\n".join(l for l in (ROOT / "requirements-runtime.txt").read_text().lower().splitlines() if not l.startswith("#"))
    for bad in ("torch", "ultralytics", "onnx>="):
        assert bad not in reqs


def test_registry_hides_pytorch_without_ultralytics(monkeypatch):
    from visioninspect import registry

    real = importlib.util.find_spec
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda n, *a: None if n == "ultralytics" else real(n, *a))
    assert "pytorch" not in [m["id"] for m in registry.available()]
    assert "classical" in [m["id"] for m in registry.available()]
