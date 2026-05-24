import cv2
import numpy as np
import onnxruntime as ort


def letterbox(image, new_shape=(768, 768), color=(114, 114, 114)):
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y




def crop_mask(mask, box):
    x1, y1, x2, y2 = box
    h, w = mask.shape
    x1 = max(0, min(w, int(np.floor(x1))))
    y1 = max(0, min(h, int(np.floor(y1))))
    x2 = max(0, min(w, int(np.ceil(x2))))
    y2 = max(0, min(h, int(np.ceil(y2))))
    out = np.zeros_like(mask)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def keep_large_components(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out


def nms(boxes, scores, iou_thres=0.45):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


class YoloV8SegONNX:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_h = int(shape[2])
        self.input_w = int(shape[3])

    def preprocess(self, image_bgr):
        img, ratio, dwdh = letterbox(image_bgr, (self.input_h, self.input_w))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(rgb, (2, 0, 1))[None]
        return x, ratio, dwdh

    def infer(self, image_bgr, mask_thr=0.5, min_area_ratio=0.01):
        x, ratio, (dw, dh) = self.preprocess(image_bgr)
        preds, protos = self.session.run(None, {self.input_name: x})

        pred = preds[0].T
        proto = protos[0]
        if proto.ndim == 4:
            proto = proto[0]
        nm = int(proto.shape[0])
        nc = int(pred.shape[1] - 4 - nm)
        if nc <= 0:
            raise ValueError(
                f"Invalid ONNX output shape: pred_dim={pred.shape[1]}, mask_dim={nm}, inferred_nc={nc}"
            )

        boxes = pred[:, :4]
        cls_raw = pred[:, 4 : 4 + nc]
        cls_scores = cls_raw.max(axis=1)
        cls_ids = cls_raw.argmax(axis=1)
        mask_coeff = pred[:, 4 + nc : 4 + nc + nm]

        keep = cls_scores > self.conf_thres
        boxes, scores, cls_ids, mask_coeff = boxes[keep], cls_scores[keep], cls_ids[keep], mask_coeff[keep]

        if boxes.shape[0] == 0:
            return []

        boxes = xywh2xyxy(boxes)
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= ratio

        h0, w0 = image_bgr.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0 - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0 - 1)

        keep_idx = nms(boxes, scores, self.iou_thres)

        c, mh, mw = proto.shape
        proto_flat = proto.reshape(c, -1)

        results = []
        left = int(round(dw))
        right = int(round(self.input_w - dw))
        top = int(round(dh))
        bottom = int(round(self.input_h - dh))
        gain_x = mw / self.input_w
        gain_y = mh / self.input_h

        for i in keep_idx:
            box = boxes[i]
            box_l = box.copy()
            box_l[[0, 2]] = box_l[[0, 2]] * ratio + dw
            box_l[[1, 3]] = box_l[[1, 3]] * ratio + dh

            m = 1.0 / (1.0 + np.exp(-(mask_coeff[i] @ proto_flat)))
            m = m.reshape(mh, mw)

            box_p = np.array([
                box_l[0] * gain_x,
                box_l[1] * gain_y,
                box_l[2] * gain_x,
                box_l[3] * gain_y,
            ], dtype=np.float32)
            m = crop_mask(m, box_p)

            m = cv2.resize(m, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            m = m[top:bottom, left:right]
            m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_LINEAR)
            mask = (m > float(mask_thr)).astype(np.uint8)

            x1, y1, x2, y2 = box.astype(int)
            hard_clip = np.zeros_like(mask, dtype=np.uint8)
            hard_clip[max(0, y1):min(h0, y2), max(0, x1):min(w0, x2)] = 1
            mask = (mask * hard_clip).astype(np.uint8)

            box_area = max(1, (max(0, x2 - x1) * max(0, y2 - y1)))
            min_area = max(8, int(box_area * float(min_area_ratio)))
            mask = keep_large_components(mask, min_area=min_area)

            if np.any(mask):
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
                mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
                mask = (mask > 0.35).astype(np.uint8)
                mask = (mask * hard_clip).astype(np.uint8)

            results.append(
                {
                    "box": box.astype(int),
                    "score": float(scores[i]),
                    "cls": int(cls_ids[i]),
                    "mask": mask,
                }
            )
        return results


def render_instances(image_bgr, instances):
    overlay = image_bgr.copy()
    colors = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (255, 165, 0),
        (138, 43, 226),
    ]
    mask_canvas = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    for idx, ins in enumerate(instances):
        color = colors[idx % len(colors)]
        mask = ins["mask"].astype(bool)
        mask_canvas[mask] = 255

        color_arr = np.array(color, dtype=np.uint8)
        overlay[mask] = ((overlay[mask] * 0.45) + (color_arr * 0.55)).astype(np.uint8)

        x1, y1, x2, y2 = ins["box"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"nail {ins['score']:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return mask_canvas, overlay
