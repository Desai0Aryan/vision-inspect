import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visioninspect import CLASSES  # noqa: E402
from visioninspect.baselines.classical import ClassicalDetector  # noqa: E402
from visioninspect.data.synth import render_sample  # noqa: E402
from visioninspect.metrics import average_precision, evaluate, iou_matrix  # noqa: E402


def test_iou_basic():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    iou = iou_matrix(a, b)[0]
    assert iou[0] == np.float32(1.0) or abs(iou[0] - 1) < 1e-6
    assert abs(iou[1] - 1 / 3) < 1e-6
    assert iou[2] == 0


def test_perfect_predictions_give_map_one():
    gts = [np.array([[10, 10, 50, 50, 0], [60, 60, 90, 90, 1]], np.float32)]
    preds = [np.array([[10, 10, 50, 50, 0.9, 0], [60, 60, 90, 90, 0.8, 1]], np.float32)]
    rep = evaluate(preds, gts, CLASSES[:2])
    assert rep["mAP50"] > 0.99 and rep["mAP50_95"] > 0.99
    assert rep["precision"] == 1.0 and rep["recall"] == 1.0


def test_no_predictions_give_zero():
    gts = [np.array([[10, 10, 50, 50, 0]], np.float32)]
    ap, *_ = average_precision([np.zeros((0, 6), np.float32)], gts, 0)
    assert ap == 0.0


def test_false_positives_lower_precision():
    gts = [np.array([[10, 10, 50, 50, 0]], np.float32)]
    preds = [np.array([[10, 10, 50, 50, 0.9, 0], [100, 100, 140, 140, 0.95, 0]], np.float32)]
    rep = evaluate(preds, gts, CLASSES[:1])
    assert rep["precision"] == 0.5 and rep["recall"] == 1.0


def test_generator_is_deterministic_and_labels_in_bounds():
    a, la = render_sample(np.random.default_rng(3))
    b, lb = render_sample(np.random.default_rng(3))
    assert np.array_equal(a, b) and np.array_equal(la, lb)
    assert a.shape == (640, 640, 3)
    if len(la):
        assert (la[:, :4] >= 0).all() and (la[:, :4] <= 640).all()
        assert (la[:, 2] > la[:, 0]).all() and (la[:, 3] > la[:, 1]).all()


def test_baseline_detects_something_on_defective_parts():
    det = ClassicalDetector()
    hits = n = 0
    rng = np.random.default_rng(11)
    for _ in range(20):
        img, lab = render_sample(rng)
        if len(lab):
            n += 1
            hits += int(len(det.predict(img)) > 0)
    assert n > 0 and hits / n > 0.7


def test_baseline_output_format():
    img, _ = render_sample(np.random.default_rng(5))
    out = ClassicalDetector().predict(img)
    assert out.ndim == 2 and out.shape[1] == 6
