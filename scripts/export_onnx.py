"""Export the trained YOLO to ONNX (FP32 and FP16 weights) and quantize to INT8.

    python scripts/export_onnx.py --weights models/yolo_pretrained.pt --imgsz 512 --calib 128

INT8 uses ONNX Runtime *static* quantization (QDQ format, per-channel weights) calibrated on
training images. The final detection-head convolutions can optionally be kept in float
(`--keep-head-fp32`), which usually recovers most of the accuracy lost to quantization.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visioninspect.data.dataset import load_split  # noqa: E402
from visioninspect.inference.onnx_detector import preprocess  # noqa: E402


class CalibReader:
    def __init__(self, data, n, imgsz, input_name):
        self.data, self.n, self.imgsz, self.input_name = data, n, imgsz, input_name
        self.rewind()

    def get_next(self):
        try:
            _, img, _ = next(self.it)
        except StopIteration:
            return None
        x, _ = preprocess([img], self.imgsz)
        return {self.input_name: x}

    def rewind(self):
        import itertools
        self.it = itertools.islice(load_split(self.data, "train"), self.n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/yolo_pretrained.pt")
    ap.add_argument("--data", default="data/inspect")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--out", default="models")
    ap.add_argument("--keep-head-fp32", action="store_true", help="keep every node of the detect head (/model.22/) in float")
    ap.add_argument("--exclude", default="", help="comma-separated substrings; matching nodes stay in float (e.g. /model.22/cv3)")
    ap.add_argument("--naive", action="store_true",
                    help="ablation: quantize EVERYTHING (breaks YOLO: class scores collapse to 0, mAP = 0). "
                         "Default keeps the final 1x1 box/class predictors and the decode in float.")
    ap.add_argument("--tag", default="", help="suffix for the INT8 file name")
    ap.add_argument("--method", default="minmax", choices=["minmax", "entropy", "percentile"],
                    help="calibration method; entropy/percentile keep many histograms and need much more RAM")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    from ultralytics import YOLO

    p = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, dynamic=True, simplify=True, opset=17)
    fp32 = out / "yolo_fp32.onnx"
    shutil.move(p, fp32)
    print("FP32 ->", fp32, f"{fp32.stat().st_size/1e6:.1f} MB")

    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = out / "yolo_fp32_pre.onnx"
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    m = onnx.load(str(pre))
    input_name = m.graph.input[0].name

    exclude = []
    pats = ["/model.22/"] if a.keep_head_fp32 else [x for x in a.exclude.split(",") if x]
    if pats:
        exclude = [n.name for n in m.graph.node if any(p in n.name for p in pats)]
        print(f"keeping {len(exclude)} nodes in FP32 (patterns: {pats})")

    if not a.naive and not a.keep_head_fp32 and not a.exclude:
        import re
        pred = re.compile(r"/cv[23]\.\d/cv[23]\.\d\.2/")
        feat = re.compile(r"/cv[23]\.\d/cv[23]\.\d\.[01]/")
        exclude = [n.name for n in m.graph.node
                   if n.name.startswith("/model.22/") and (pred.search(n.name) or not feat.search(n.name))]
        print(f"keeping {len(exclude)} nodes in FP32 (final predictors + decode)")

    int8 = out / f"yolo_int8{a.tag}.onnx"
    reader = CalibReader(a.data, a.calib, a.imgsz, input_name)
    quantize_static(str(pre), str(int8), reader, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method={"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy, "percentile": CalibrationMethod.Percentile}[a.method], nodes_to_exclude=exclude,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    print("INT8 ->", int8, f"{int8.stat().st_size/1e6:.1f} MB")
    pre.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
