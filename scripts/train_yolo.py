"""Fine-tune a COCO-pretrained YOLO on the inspection dataset.

    python scripts/train_yolo.py --weights yolov8n.pt --epochs 20 --imgsz 640
    python scripts/train_yolo.py --weights yolov8n.pt --scratch   # ablation: no pretraining
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inspect/data.yaml")
    ap.add_argument("--weights", default="models/yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="yolo_pretrained")
    ap.add_argument("--scratch", action="store_true", help="random init (same architecture) for a pretraining ablation")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.weights.replace(".pt", ".yaml") if a.scratch else a.weights)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=a.device,
                project="runs", name=a.name, exist_ok=True, pretrained=not a.scratch, cache="ram", patience=50, plots=True,
                # light augmentation: parts are upright, so no vertical flips / heavy rotation
                fliplr=0.5, flipud=0.0, degrees=0.0, mosaic=0.5, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, seed=0, deterministic=True)
    best = Path("runs") / a.name / "weights" / "best.pt"
    Path("models").mkdir(exist_ok=True)
    if best.exists():
        shutil.copy(best, Path("models") / f"{a.name}.pt")
        print("saved", Path("models") / f"{a.name}.pt")
    print(json.dumps({"run": a.name, "epochs": a.epochs}))


if __name__ == "__main__":
    main()
