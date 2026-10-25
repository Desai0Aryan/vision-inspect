PY ?= python3

.PHONY: data video baseline test clean

data:
	$(PY) scripts/make_dataset.py --out data/inspect --train 600 --val 150 --test 150

video:
	$(PY) scripts/make_demo_video.py --out data/demo/conveyor.mp4

baseline:
	$(PY) scripts/eval_baseline.py --data data/inspect --split test

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/inspect outputs
