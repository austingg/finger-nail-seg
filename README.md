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

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Demo Results

| Original | Mask | Overlay |
|---|---|---|
| ![1 input](./examples/input/1.jpg) | ![1 mask](./examples/output_mask/1_mask.png) | ![1 overlay](./examples/output_overlay/1_overlay.jpg) |
| ![2 input](./examples/input/2.jpg) | ![2 mask](./examples/output_mask/2_mask.png) | ![2 overlay](./examples/output_overlay/2_overlay.jpg) |

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
