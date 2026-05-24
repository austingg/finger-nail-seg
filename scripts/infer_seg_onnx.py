import argparse
from pathlib import Path

import cv2

from onnx_seg_utils import YoloV8SegONNX, render_instances


def list_images(path: Path):
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/nail-seg.onnx")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--mask_thr", type=float, default=0.5)
    parser.add_argument("--min_area_ratio", type=float, default=0.01)
    args = parser.parse_args()

    inferencer = YoloV8SegONNX(args.model, conf_thres=args.conf, iou_thres=args.iou)

    input_path = Path(args.input)
    files = list_images(input_path)
    if not files:
        raise SystemExit(f"No images found under: {input_path}")

    out_root = Path(args.out)
    out_mask = out_root / "output_mask"
    out_overlay = out_root / "output_overlay"
    out_mask.mkdir(parents=True, exist_ok=True)
    out_overlay.mkdir(parents=True, exist_ok=True)

    for p in files:
        image = cv2.imread(str(p))
        if image is None:
            print(f"[WARN] skip unreadable image: {p}")
            continue

        inferencer.conf_thres = float(args.conf)
        inferencer.iou_thres = float(args.iou)
        instances = inferencer.infer(
            image,
            mask_thr=float(args.mask_thr),
            min_area_ratio=float(args.min_area_ratio),
        )
        mask, overlay = render_instances(image, instances)

        stem = p.stem
        cv2.imwrite(str(out_mask / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(out_overlay / f"{stem}_overlay.jpg"), overlay)
        print(f"[OK] {p.name}: {len(instances)} instances")

    print(f"Done. Results saved in: {out_root}")


if __name__ == "__main__":
    main()
