import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from onnx_seg_utils import YoloV8SegONNX, render_instances


def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_examples(example_dir: Path):
    if not example_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in example_dir.iterdir() if p.suffix.lower() in exts])
    return [[str(p)] for p in files[:12]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/nail-seg.onnx")
    parser.add_argument("--example_dir", type=str, default="examples/input")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    inferencer = YoloV8SegONNX(args.model, conf_thres=args.conf, iou_thres=args.iou)

    def predict(image_np, conf_thr=None, iou_thr=None, mask_thr=None, min_area_ratio=None):
        if image_np is None:
            return None, None, "Please upload an image first."

        conf_thr = float(args.conf if conf_thr is None else conf_thr)
        iou_thr = float(args.iou if iou_thr is None else iou_thr)
        mask_thr = float(0.5 if mask_thr is None else mask_thr)
        min_area_ratio = float(0.01 if min_area_ratio is None else min_area_ratio)

        inferencer.conf_thres = float(conf_thr)
        inferencer.iou_thres = float(iou_thr)

        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        instances = inferencer.infer(
            image_bgr,
            mask_thr=float(mask_thr),
            min_area_ratio=float(min_area_ratio),
        )
        mask, overlay = render_instances(image_bgr, instances)
        msg = (
            f"Detected instances: {len(instances)} | "
            f"conf={conf_thr:.2f}, iou={iou_thr:.2f}, mask_thr={mask_thr:.2f}, min_area_ratio={min_area_ratio:.3f}"
        )

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return to_rgb(overlay), mask_rgb, msg

    example_data = load_examples(Path(args.example_dir))

    with gr.Blocks(title="Nail Segmentation Demo (ONNX Runtime CPU)") as demo:
        gr.Markdown("## Nail Segmentation Demo (YOLOv8s-seg ONNX Runtime CPU)")
        gr.Markdown("Upload an image, then click **Segment** to run instance segmentation.")

        with gr.Row():
            in_image = gr.Image(type="numpy", label="Input Image")
        with gr.Row():
            conf_slider = gr.Slider(0.1, 0.95, value=args.conf, step=0.01, label="Confidence")
            iou_slider = gr.Slider(0.1, 0.9, value=args.iou, step=0.01, label="NMS IoU")
        with gr.Row():
            mask_thr_slider = gr.Slider(0.2, 0.8, value=0.5, step=0.01, label="Mask Threshold")
            min_area_slider = gr.Slider(0.001, 0.05, value=0.01, step=0.001, label="Min Area Ratio")
        with gr.Row():
            btn = gr.Button("Segment", variant="primary")
        with gr.Row():
            out_overlay = gr.Image(type="numpy", label="Overlay (Instances)")
            out_mask = gr.Image(type="numpy", label="Mask (Binary)")
        out_text = gr.Textbox(label="Result")

        btn.click(fn=predict, inputs=[in_image, conf_slider, iou_slider, mask_thr_slider, min_area_slider], outputs=[out_overlay, out_mask, out_text])

        if example_data:
            gr.Examples(examples=example_data, inputs=[in_image], label="Examples")

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
