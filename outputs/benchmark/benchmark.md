# Inference benchmark

150 `test` images, imgsz 512, 2 CPU threads, end-to-end (preprocess + model + NMS).

| Backend | Size (MB) | mAP@0.5 | mAP@0.5:0.95 | ms/img bs=1 | ms/img bs=2 | ms/img bs=4 | ms/img bs=8 | Best throughput (img/s) | Speed-up (bs1) |
|---|---|---|---|---|---|---|---|---|---|
| pytorch_fp32 | 6.5 | 0.991 | 0.830 | 68.6 | 59.8 | 57.2 | 59.3 | 17.5 | 1.00x |
| onnx_fp32 | 12.9 | 0.991 | 0.830 | 38.5 | 37.8 | 35.8 | 38.0 | 28.0 | 1.78x |
| onnx_int8 | 3.8 | 0.988 | 0.807 | 25.7 | 23.7 | 21.8 | 22.7 | 45.9 | 2.67x |