"""Latency / throughput / accuracy benchmark across inference backends and batch sizes.

    python scripts/benchmark.py --split test --imgsz 512

Reports, per backend: mAP@0.5, mAP@0.5:0.95, model size, and for each batch size the
per-image latency (mean / p50 / p95), throughput (images/s) and speed-up vs the PyTorch baseline.
Everything is measured end to end (preprocess + model + NMS), on the same machine, in one process.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visioninspect import CLASSES  # noqa: E402
from visioninspect.data.dataset import load_split  # noqa: E402
from visioninspect.metrics import evaluate  # noqa: E402

COL = {"pytorch_fp32": "#8a8f98", "onnx_fp32": "#2f81f7", "onnx_int8": "#2da44e"}


def time_batches(det, imgs, bs, reps=3):
    """Per-image latency (ms) measured over whole batches, repeated `reps` times."""
    batches = [imgs[i:i + bs] for i in range(0, len(imgs) - bs + 1, bs)]
    det.predict_batch(batches[0])  # warmup
    lat = []
    for _ in range(reps):
        for b in batches:
            t = time.perf_counter()
            det.predict_batch(b)
            lat.append((time.perf_counter() - t) * 1000 / len(b))
    return np.array(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inspect")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--pt", default="models/yolo_pretrained.pt")
    ap.add_argument("--fp32", default="models/yolo_fp32.onnx")
    ap.add_argument("--int8", default="models/yolo_int8.onnx")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--conf", type=float, default=0.45, help="operating threshold for P/R")
    ap.add_argument("--out", default="outputs/benchmark")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bss = [int(x) for x in a.batches.split(",")]

    items = list(load_split(a.data, a.split))
    imgs = [i for _, i, _ in items]
    gts = [g for _, _, g in items]

    from visioninspect.inference.onnx_detector import OnnxDetector
    from visioninspect.models.yolo import YoloDetector

    backends = {"pytorch_fp32": (YoloDetector(a.pt, imgsz=a.imgsz), a.pt)}
    if Path(a.fp32).exists():
        backends["onnx_fp32"] = (OnnxDetector(a.fp32, imgsz=a.imgsz), a.fp32)
    if Path(a.int8).exists():
        backends["onnx_int8"] = (OnnxDetector(a.int8, imgsz=a.imgsz), a.int8)

    res = {}
    for key, (det, path) in backends.items():
        print(f"[{key}] accuracy ...", flush=True)
        preds = []
        for i in range(0, len(imgs), 8):
            preds += det.predict_batch(imgs[i:i + 8])
        rep = evaluate(preds, gts, CLASSES, conf_thr=a.conf)
        r = {"size_mb": os.path.getsize(path) / 1e6, "mAP50": rep["mAP50"], "mAP50_95": rep["mAP50_95"],
             "precision": rep["precision"], "recall": rep["recall"], "f1": rep["f1"],
             "per_class_AP50": {c: v["AP50"] for c, v in rep["per_class"].items()}, "latency": {}}
        for bs in bss:
            print(f"[{key}] latency batch={bs} ...", flush=True)
            lat = time_batches(det, imgs[:48], bs)
            r["latency"][bs] = {"mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)),
                                "p95_ms": float(np.percentile(lat, 95)), "throughput_ips": float(1000 / lat.mean())}
        res[key] = r
        print(f"   mAP50={r['mAP50']:.3f}  size={r['size_mb']:.1f}MB  bs1={r['latency'][bss[0]]['mean_ms']:.1f}ms", flush=True)

    base = res["pytorch_fp32"]["latency"][bss[0]]["mean_ms"]
    for k, r in res.items():
        for bs, l in r["latency"].items():
            l["speedup_vs_pytorch_bs1"] = base / l["mean_ms"]
    res["_meta"] = {"cpu_threads": os.cpu_count(), "imgsz": a.imgsz, "n_images": len(imgs), "conf": a.conf}
    (out / "benchmark.json").write_text(json.dumps(res, indent=2))

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    keys = [k for k in res if not k.startswith("_")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        v = [res[k]["latency"][b]["mean_ms"] for b in bss]
        bars = axs[0].bar(np.arange(len(bss)) + i * w - 0.4 + w / 2, v, w, label=k, color=COL.get(k))
        axs[0].bar_label(bars, fmt="%.0f", fontsize=8, padding=1)
    axs[0].set_xticks(range(len(bss)), [f"bs={b}" for b in bss])
    axs[0].set_ylabel("ms / image (lower is better)")
    axs[0].set_title("Latency per image")
    axs[0].set_ylim(0, max(l["mean_ms"] for r in res.values() if isinstance(r, dict) and "latency" in r for l in r["latency"].values()) * 1.25)
    axs[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    for k in keys:
        axs[1].plot(bss, [res[k]["latency"][b]["throughput_ips"] for b in bss], "-o", label=k, color=COL.get(k), lw=2)
    axs[1].set_xlabel("batch size")
    axs[1].set_ylabel("images / second")
    axs[1].set_title("Throughput")
    axs[1].set_xticks(bss)
    for k in keys:
        axs[2].scatter(res[k]["latency"][1]["mean_ms"], res[k]["mAP50_95"], s=140, color=COL.get(k), label=k, zorder=3)
        axs[2].annotate(f"{res[k]['mAP50_95']:.3f}", (res[k]["latency"][1]["mean_ms"], res[k]["mAP50_95"]), textcoords="offset points", xytext=(8, 6), fontsize=9)
    axs[2].set_xlabel("latency @ bs=1 (ms)")
    axs[2].set_ylabel("mAP@0.5:0.95")
    axs[2].set_title("Accuracy vs speed (up-left is best)")
    fig.tight_layout()
    fig.savefig(out / "latency_throughput.png", dpi=150)
    plt.close(fig)

    lines = ["# Inference benchmark", "", f"{len(imgs)} `{a.split}` images, imgsz {a.imgsz}, {os.cpu_count()} CPU threads, end-to-end (preprocess + model + NMS).", "",
             "| Backend | Size (MB) | mAP@0.5 | mAP@0.5:0.95 | " + " | ".join(f"ms/img bs={b}" for b in bss) + " | Best throughput (img/s) | Speed-up (bs1) |",
             "|---|---|---|---|" + "---|" * len(bss) + "---|---|"]
    for k in keys:
        r = res[k]
        best = max(l["throughput_ips"] for l in r["latency"].values())
        lines.append(f"| {k} | {r['size_mb']:.1f} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} | " +
                     " | ".join(f"{r['latency'][b]['mean_ms']:.1f}" for b in bss) + f" | {best:.1f} | {r['latency'][bss[0]]['speedup_vs_pytorch_bs1']:.2f}x |")
    (out / "benchmark.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
