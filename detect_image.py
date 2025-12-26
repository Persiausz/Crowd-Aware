#main.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------
# CONFIG
# ----------------------
IMAGE_PATH = "/Users/chidimon/Dev/Crowd-Aware/source camera/image copy 2.png"
WEIGHTS = "yolov8n.pt"

ROWS, COLS = 3, 3
YOLO_CONF = 0.25
YOLO_IOU  = 0.5
NMS_IOU   = 0.5

# ----------------------
# NMS (numpy)
# ----------------------
def nms_boxes(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
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

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

# ----------------------
# วาดข้อความมุมซ้ายบน (พื้นหลังทึบ)
# ----------------------
def draw_topleft_label(img_bgr, text, pad=8):
    img = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.95
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0, y0 = 10, 10  # มุมซ้ายบน
    # กล่องพื้นหลัง
    cv2.rectangle(img, (x0, y0), (x0 + tw + pad*2, y0 + th + pad*2), (0, 0, 0), -1)
    # ข้อความ (สีขาว)
    cv2.putText(img, text, (x0 + pad, y0 + th + pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

# ----------------------
# ตรวจจับคนแบบ Tiling-only
# ----------------------
def detect_persons_tiled(img_bgr, model, rows=3, cols=3, conf=0.25, yolo_iou=0.5, nms_iou=0.5):
    H, W = img_bgr.shape[:2]
    tile_h, tile_w = H // rows, W // cols

    # หา class index ของ "person"
    person_class_index = None
    for k, v in model.names.items():
        if v == "person":
            person_class_index = k
            break
    if person_class_index is None:
        raise RuntimeError("ไม่พบคลาส 'person' ในโมเดลที่โหลดมา")

    all_boxes, all_scores = [], []

    for r in range(rows):
        for c in range(cols):
            y0 = r * tile_h
            x0 = c * tile_w
            y1 = H if r == rows - 1 else (r + 1) * tile_h
            x1 = W if c == cols - 1 else (c + 1) * tile_w

            tile = img_bgr[y0:y1, x0:x1]
            res = model.predict(source=tile, conf=conf, iou=yolo_iou, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue

            cls = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            xyxy = res.boxes.xyxy.cpu().numpy()

            mask = (cls == float(person_class_index))
            if np.any(mask):
                boxes = xyxy[mask]
                scores = confs[mask]
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                all_boxes.append(boxes)
                all_scores.append(scores)

    if len(all_boxes) == 0:
        out0 = draw_topleft_label(img_bgr, "persons: 0")
        return out0, 0

    all_boxes = np.vstack(all_boxes)
    all_scores = np.concatenate(all_scores)

    keep_idx = nms_boxes(all_boxes, all_scores, iou_thresh=nms_iou)
    kept_boxes = all_boxes[keep_idx]

    out = img_bgr.copy()
    for (x1, y1, x2, y2) in kept_boxes.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # ใส่จำนวนคนมุมซ้ายบน
    out = draw_topleft_label(out, f"persons: {len(kept_boxes)}")
    return out, len(kept_boxes)

def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"ไม่พบไฟล์รูป: {IMAGE_PATH}")

    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise RuntimeError("อ่านรูปไม่สำเร็จ (อาจเป็นฟอร์แมตที่ OpenCV ไม่รองรับ)")

    model = YOLO(WEIGHTS)

    out_img, count = detect_persons_tiled(
        img_bgr, model, rows=ROWS, cols=COLS, conf=YOLO_CONF, yolo_iou=YOLO_IOU, nms_iou=NMS_IOU
    )

    root, ext = os.path.splitext(IMAGE_PATH)
    out_path = f"{root}_tiled.jpg"
    cv2.imwrite(out_path, out_img)
    print(f"[Tiled {ROWS}x{COLS}] พบคนทั้งหมด = {count}")
    print(f"บันทึกผลที่: {out_path}")

if __name__ == "__main__":
    main()
