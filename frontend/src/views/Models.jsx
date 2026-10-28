import { useEffect, useState } from "react";
import { api } from "../api.js";
import { GroupedBars, HLatency, Legend } from "../components/Charts.jsx";
import { CLASSES, CLASS_LABEL } from "../format.js";

const C = { classical: "#8a99a3", yolo: "#2a7547", pt: "#8a99a3", fp32: "#2874e8", int8: "#2a7547" };

export default function Models() {
  const [b, setB] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    api.benchmarks().then(setB).catch((e) => setErr(e.message));
  }, []);

  if (err) return <p className="err">{err}</p>;
  if (!b) return <p className="muted">Loading results…</p>;
  const cmp = b.comparison;
  const opt = b.optimization;
  const vid = b.video;

  return (
    <div>
      <h1 className="title">What the models can do</h1>
      <p className="lede">Every number here was measured on a held-out test set of 150 images that the models never saw while training, using one shared evaluator.</p>

      {cmp?.classical && cmp?.yolo_pretrained ? (
        <div className="model-grid">
          <div>
            <h2 className="h2">Learned detector vs hand-written rules</h2>
            <Legend items={[{ name: "Classical OpenCV", color: C.classical }, { name: "YOLOv8n, pretrained", color: C.yolo }]} />
            <GroupedBars
              cats={["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]}
              series={[
                { name: "Classical", color: C.classical, values: [cmp.classical.mAP50, cmp.classical.mAP50_95, cmp.classical.precision, cmp.classical.recall] },
                { name: "YOLO", color: C.yolo, values: [cmp.yolo_pretrained.mAP50, cmp.yolo_pretrained.mAP50_95, cmp.yolo_pretrained.precision, cmp.yolo_pretrained.recall] },
              ]}
            />
          </div>
          <div>
            <h2 className="h2">Where the rules fall short</h2>
            <Legend items={[{ name: "Classical OpenCV", color: C.classical }, { name: "YOLOv8n, pretrained", color: C.yolo }]} />
            <GroupedBars
              cats={CLASSES.map((c) => CLASS_LABEL[c])}
              series={[
                { name: "Classical", color: C.classical, values: CLASSES.map((c) => cmp.classical.per_class?.[c]?.AP50) },
                { name: "YOLO", color: C.yolo, values: CLASSES.map((c) => cmp.yolo_pretrained.per_class?.[c]?.AP50) },
              ]}
            />
            <p className="muted small">Average precision at IoU 0.5, per defect type. Cracks are the hardest for rules, because they look like scratches and stains to a threshold.</p>
          </div>
        </div>
      ) : (
        <div className="empty">Comparison results are missing. Run <code>make compare</code> to create them.</div>
      )}

      {opt?.pytorch_fp32 ? (
        <div className="model-grid">
          <div>
            <h2 className="h2">Making it fast</h2>
            <HLatency
              rows={[
                { label: "PyTorch", value: opt.pytorch_fp32.latency["1"].mean_ms, color: C.pt, note: `${opt.pytorch_fp32.size_mb.toFixed(1)} MB` },
                { label: "ONNX Runtime, FP32", value: opt.onnx_fp32.latency["1"].mean_ms, color: C.fp32, note: `${opt.onnx_fp32.latency["1"].speedup_vs_pytorch_bs1.toFixed(1)}x faster` },
                { label: "ONNX Runtime, INT8", value: opt.onnx_int8.latency["1"].mean_ms, color: C.int8, note: `${opt.onnx_int8.latency["1"].speedup_vs_pytorch_bs1.toFixed(1)}x faster, ${opt.onnx_int8.size_mb.toFixed(1)} MB` },
              ]}
            />
            <p className="muted small">Milliseconds per image on a 2-core CPU, including preprocessing and box filtering. INT8 gives up {(opt.onnx_fp32.mAP50 - opt.onnx_int8.mAP50).toFixed(3)} mAP@0.5 for the speed.</p>
          </div>
          <div>
            <h2 className="h2">Batching</h2>
            <Legend items={[{ name: "PyTorch", color: C.pt }, { name: "ONNX FP32", color: C.fp32 }, { name: "ONNX INT8", color: C.int8 }]} />
            <GroupedBars
              cats={["1", "2", "4", "8"].map((n) => `batch ${n}`)}
              max={Math.ceil(opt.pytorch_fp32.latency["1"].mean_ms / 10) * 10}
              fmtv={(v) => v.toFixed(0)}
              series={[
                { name: "PyTorch", color: C.pt, values: ["1", "2", "4", "8"].map((n) => opt.pytorch_fp32.latency[n]?.mean_ms) },
                { name: "FP32", color: C.fp32, values: ["1", "2", "4", "8"].map((n) => opt.onnx_fp32.latency[n]?.mean_ms) },
                { name: "INT8", color: C.int8, values: ["1", "2", "4", "8"].map((n) => opt.onnx_int8.latency[n]?.mean_ms) },
              ]}
            />
            <p className="muted small">Milliseconds per image. Batching helps a little on a CPU and would help much more on a GPU.</p>
          </div>
        </div>
      ) : (
        <div className="empty" style={{ marginTop: 30 }}>Speed results are missing. Run <code>make bench</code> to create them.</div>
      )}

      {vid && (
        <div style={{ marginTop: 36 }}>
          <h2 className="h2">The whole line, scored against the truth</h2>
          <table className="table" style={{ maxWidth: 640 }}>
            <tbody>
              <tr><td>Parts counted</td><td><b>{vid.system_level.parts_counted}</b> of {vid.system_level.parts_truth}</td></tr>
              <tr><td>Pass or reject verdicts correct</td><td><b>{(vid.system_level.verdict_accuracy * 100).toFixed(0)}%</b></td></tr>
              <tr><td>Unique defects counted</td><td><b>{vid.system_level.unique_defects_pred}</b> of {vid.system_level.unique_defect_instances_truth_approx}</td></tr>
              <tr><td>Frame-level mAP@0.5</td><td><b>{vid.frame_level.mAP50.toFixed(3)}</b></td></tr>
            </tbody>
          </table>
        </div>
      )}

      <div className="callout" style={{ marginTop: 36 }}>
        <b>Two things worth knowing.</b> Quantizing every layer to 8 bits broke the detector completely (mAP 0.000): the class scores collapsed to zero. Keeping the final prediction layers in float fixed it at almost no cost.
        And the model is sensitive to part size: with the camera pulled back so parts look 40% smaller, accuracy drops sharply. Both findings are in the README.
      </div>
    </div>
  );
}
