"""Generate the synthetic inspection dataset (YOLO format).

    python scripts/make_dataset.py --out data/inspect --train 600 --val 150 --test 150
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visioninspect.data.dataset import build_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/inspect")
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    last = {"s": None}

    def prog(split, i, n):
        if i == n or i % 50 == 0:
            print(f"  {split:5s} {i:4d}/{n}", flush=True)

    stats = build_dataset(a.out, a.train, a.val, a.test, a.size, a.seed, prog)
    print("done:", stats)


if __name__ == "__main__":
    main()
