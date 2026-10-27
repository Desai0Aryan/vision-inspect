"""Generate the conveyor-belt demo clip used by the live-inspection demo.

    python scripts/make_demo_video.py --out data/demo/conveyor.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visioninspect.data.video import make_demo_video  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo/conveyor.mp4")
    ap.add_argument("--seconds", type=int, default=18)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=float, default=1.0, help="part size relative to training scale; 0.62 = far camera stress test")
    a = ap.parse_args()
    print(make_demo_video(a.out, seconds=a.seconds, seed=a.seed, scale=a.scale))
