# Nail Try-On Segmentation (YOLOv8s-seg + ONNX Runtime CPU)

[English](./README.md) | [中文](./README-CN.md)

A baseline project for nail instance segmentation and lightweight try-on workflows.

## Features

- ONNX model inference with `onnxruntime` (CPU)
- Local CLI inference scripts
- Gradio web demo with upload, segmentation, and examples
- Output artifacts for mask and overlay visualization

## Project Structure

```text
.
├── models/
│   └── nail-seg.onnx
├── scripts/
│   ├── onnx_seg_utils.py
│   ├── infer_seg_onnx.py
│   └── demo_gradio.py
├── examples/
│   ├── input/
│   ├── output_mask/
│   ├── output_overlay/
│   └── output_tryon/
├── pyproject.toml
├── uv.lock
├── .gitignore
├── README.md
└── README-CN.md
```

## Requirements

- Python 3.10+
- `uv`

## Setup (uv)

```bash
uv python install 3.10
uv venv --python 3.10
source .venv/bin/activate
uv sync --frozen
```

## Demo Results

| Original | Mask | Overlay |
|---|---|---|
| ![1 input](./examples/input/1.jpg) | ![1 mask](./examples/output_mask/1_mask.png) | ![1 overlay](./examples/output_overlay/1_overlay.jpg) |
| ![2 input](./examples/input/2.jpg) | ![2 mask](./examples/output_mask/2_mask.png) | ![2 overlay](./examples/output_overlay/2_overlay.jpg) |

## Benchmark

Environment:

- CPU: Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz (6 cores, 12 threads)
- Runtime: ONNX Runtime CPUExecutionProvider
- Session defaults: `intra_op_num_threads=0`, `inter_op_num_threads=0`, `execution_mode=ORT_SEQUENTIAL`, `graph_optimization_level=ORT_ENABLE_ALL`

Input images:

- `examples/input/1.jpg`: `752 x 984`
- `examples/input/2.jpg`: `902 x 1088`

Method:

- Warmup: 1 run
- Measured runs: 10 (same process, alternating 2 images)
- Model: `models/nail-seg.onnx`
- Args: `conf=0.25`, `iou=0.45`, `mask_thr=0.5`, `min_area_ratio=0.01`

Latency (ms):

- Mean: `290.10`
- P50: `291.06`
- P90: `303.75`
- Min/Max: `274.40 / 304.76`
- Throughput (from mean): `3.45 FPS`

> Notes: results are measured on the local machine and may vary by CPU model, load, and runtime settings.

## CLI Inference

Script: `scripts/infer_seg_onnx.py`

```bash
python scripts/infer_seg_onnx.py \
  --model models/nail-seg.onnx \
  --input examples/input \
  --out outputs \
  --conf 0.25 \
  --iou 0.45
```

Outputs:

- Binary masks in `output_mask/`
- Instance overlays in `output_overlay/`

## Gradio Demo

Script: `scripts/demo_gradio.py`

```bash
python scripts/demo_gradio.py \
  --model models/nail-seg.onnx \
  --example_dir examples/input \
  --host 127.0.0.1 \
  --port 7860
```

Open: [http://127.0.0.1:7860](http://127.0.0.1:7860)

## Notes

- Intended for consumer-level visual effects (e.g., color try-on pre-processing)
- Not for medical use
- Performance may degrade under strong reflections, heavy occlusion, or extreme poses
