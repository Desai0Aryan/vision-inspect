import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visioninspect.inference.onnx_detector import letterbox, postprocess, preprocess  # noqa: E402


def test_letterbox_square_and_ratio():
    img = np.zeros((540, 960, 3), np.uint8)
    out, r, px, py = letterbox(img, 512)
    assert out.shape == (512, 512, 3)
    assert abs(r - 512 / 960) < 1e-9 and px == 0 and py > 0


def test_preprocess_shape_and_range():
    x, meta = preprocess([np.full((300, 400, 3), 255, np.uint8)] * 3, 256)
    assert x.shape == (3, 3, 256, 256) and x.dtype == np.float32
    assert x.max() <= 1.0 and len(meta) == 3


def test_postprocess_maps_boxes_back_to_original_pixels():
    # one detection in letterboxed space for a 960x540 frame at imgsz 512
    r, px, py = 512 / 960, 0, (512 - round(540 * 512 / 960)) // 2
    cx, cy, w, h = 256.0, 256.0, 64.0, 32.0
    out = np.zeros((1, 8, 1), np.float32)
    out[0, :4, 0] = [cx, cy, w, h]
    out[0, 4 + 2, 0] = 0.9           # class 2
    det = postprocess(out, [(r, px, py)], conf=0.25, iou=0.5, nc=4)[0]
    assert det.shape == (1, 6) and int(det[0, 5]) == 2
    assert abs((det[0, 0] + det[0, 2]) / 2 - 480) < 1.0    # centre x in original frame
    assert abs((det[0, 2] - det[0, 0]) - 64 / r) < 1.0


def test_postprocess_nms_suppresses_duplicates_but_keeps_other_classes():
    out = np.zeros((1, 8, 3), np.float32)
    out[0, :4, 0] = [100, 100, 40, 40]; out[0, 4, 0] = 0.9
    out[0, :4, 1] = [102, 101, 40, 40]; out[0, 4, 1] = 0.8      # duplicate, same class
    out[0, :4, 2] = [101, 100, 40, 40]; out[0, 5, 2] = 0.7      # overlapping, different class
    det = postprocess(out, [(1.0, 0, 0)], conf=0.25, iou=0.5, nc=4)[0]
    assert sorted(det[:, 5].astype(int).tolist()) == [0, 1]


def test_postprocess_empty():
    out = np.zeros((2, 8, 10), np.float32)
    res = postprocess(out, [(1.0, 0, 0)] * 2, conf=0.25, iou=0.5, nc=4)
    assert all(r.shape == (0, 6) for r in res)
