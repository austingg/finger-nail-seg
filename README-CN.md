# Nail Try-On Segmentation（YOLOv8s-seg + ONNX Runtime CPU）

[English](./README.md) | [中文](./README-CN.md)

这是一个用于指甲实例分割与轻量试妆流程的基线项目。

## 功能特性

- 使用 `onnxruntime`（CPU）进行 ONNX 模型推理
- 提供本地 CLI 推理脚本
- 提供 Gradio Web Demo（上传、分割、示例）
- 支持导出 mask 与 overlay 可视化结果

## 项目结构

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

## 环境要求

- Python 3.10+
- `uv`

## 安装

```bash
uv venv
source .venv/bin/activate
uv sync
```

## CLI 推理

脚本：`scripts/infer_seg_onnx.py`

```bash
python scripts/infer_seg_onnx.py \
  --model models/nail-seg.onnx \
  --input examples/input \
  --out outputs \
  --conf 0.25 \
  --iou 0.45
```

输出结果：

- 二值 mask：`output_mask/`
- 实例叠加图：`output_overlay/`

## Gradio Demo

脚本：`scripts/demo_gradio.py`

```bash
python scripts/demo_gradio.py \
  --model models/nail-seg.onnx \
  --example_dir examples/input \
  --host 127.0.0.1 \
  --port 7860
```

浏览器访问：[http://127.0.0.1:7860](http://127.0.0.1:7860)

## 说明

- 适用于消费级视觉效果（如试色前处理）
- 非医疗用途
- 在强反光、重遮挡、极端姿态下效果可能下降
