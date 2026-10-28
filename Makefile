PY ?= python3

# Docker
.PHONY: docker docker-full up down
.PHONY: data video baseline train compare export bench serve eval-video ui-build ui-dev ui-smoke test clean

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

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

eval-video:
	$(PY) scripts/eval_video.py --video data/demo/conveyor.mp4 --model int8 --stride 3

ui-build:
	cd frontend && npm install --no-audit --no-fund && npm run build

ui-dev:
	cd frontend && npm run dev

ui-smoke:
	$(PY) scripts/ui_smoke.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/inspect outputs

docker:
	docker build -t visioninspect .

docker-full:
	docker build --target full -t visioninspect:full .

up:
	docker compose up --build

down:
	docker compose down
