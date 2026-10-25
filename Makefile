PY ?= python3

.PHONY: data video baseline train compare export bench test clean

data:
	$(PY) scripts/make_dataset.py --out data/inspect --train 600 --val 150 --test 150

video:
	$(PY) scripts/make_demo_video.py --out data/demo/conveyor.mp4

baseline:
	$(PY) scripts/eval_baseline.py --data data/inspect --split test

train:
	$(PY) scripts/train_yolo.py --weights models/yolov8n.pt --epochs 12 --imgsz 512 --name yolo_pretrained

compare:
	$(PY) scripts/compare_models.py --split test --yolo models/yolo_pretrained.pt --imgsz 512

export:
	$(PY) scripts/export_onnx.py --weights models/yolo_pretrained.pt --imgsz 512 --calib 64

bench:
	$(PY) scripts/benchmark.py --split test

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/inspect outputs
