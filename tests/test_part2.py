import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visioninspect import CLASSES  # noqa: E402

compare = pytest.importorskip("matplotlib")


def _load_compare():
    import importlib.util
    spec = importlib.util.spec_from_file_location("compare_models", Path(__file__).resolve().parents[1] / "scripts" / "compare_models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_confusion_counts_hits_confusions_and_misses():
    m = _load_compare()
    gts = [np.array([[0, 0, 10, 10, 0], [50, 50, 80, 80, 1], [100, 100, 120, 120, 2]], np.float32)]
    preds = [np.array([[0, 0, 10, 10, 0.9, 0],       # correct scratch
                       [50, 50, 80, 80, 0.8, 2]],    # crack predicted as stain
                      np.float32)]
    M = m.confusion(preds, gts, thr=0.5)
    assert M[0, 0] == 1
    assert M[1, 2] == 1
    assert M[2, len(CLASSES)] == 1          # missed stain
    assert M.sum() == 3


def test_best_conf_returns_value_in_grid():
    m = _load_compare()
    gts = [np.array([[0, 0, 10, 10, 0]], np.float32)]
    preds = [np.array([[0, 0, 10, 10, 0.7, 0], [30, 30, 50, 50, 0.2, 0]], np.float32)]
    t = m.best_conf(preds, gts)
    assert 0.2 < t <= 0.7
