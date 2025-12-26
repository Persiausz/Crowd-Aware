#index.py
import os, time, json, asyncio
import cv2
import numpy as np
import websockets
from ultralytics import YOLO

# =============== CONFIG ===============
VIDEO_PATH = "/Users/chidimon/Dev/Crowd-Aware/source camera/source.mp4"
WEIGHTS    = "yolov8n.pt"
DEVICE     = None          # 'cuda' | 'mps' | None
ROWS, COLS = 3, 3
YOLO_CONF  = 0.25
YOLO_IOU   = 0.5
NMS_IOU    = 0.5
PROCESS_FPS = 2            # ประมวลผล ~2 fps
WS_HOST, WS_PORT = "localhost", 8765
# ======================================

def nms_boxes(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0: return []
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

def detect_persons_tiled(img_bgr, model, person_idx, rows, cols, conf, yolo_iou, nms_iou):
    H, W = img_bgr.shape[:2]
    tile_h, tile_w = H // rows, W // cols
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
            mask = (cls == float(person_idx))
            if np.any(mask):
                boxes = xyxy[mask]
                scores = confs[mask]
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                all_boxes.append(boxes)
                all_scores.append(scores)

    if len(all_boxes) == 0:
        return 0

    all_boxes = np.vstack(all_boxes)
    all_scores = np.concatenate(all_scores)
    keep_idx = nms_boxes(all_boxes, all_scores, iou_thresh=nms_iou)
    return len(keep_idx)

def format_ts_and_phase(t_sec: float):
    mm = int(t_sec // 60)
    ss = int(t_sec % 60)
    ts = f"{mm:02d}:{ss:02d}"
    if t_sec < 5:      phase = "Dense (0–5s)"
    elif t_sec < 10:   phase = "Medium (5–10s)"
    else:              phase = "Sparse (10–15s)"
    return ts, phase

async def stream_counts(websocket):
    if not os.path.exists(VIDEO_PATH):
        await websocket.send(json.dumps({"error": f"video not found: {VIDEO_PATH}"}))
        return

    model = YOLO(WEIGHTS)
    if DEVICE is not None:
        try:
            model.to(DEVICE)
        except Exception as e:
            print("[WARN] move device failed:", e)

    # person class idx
    person_idx = None
    for k, v in model.names.items():
        if v == "person":
            person_idx = k
            break
    if person_idx is None:
        await websocket.send(json.dumps({"error": "person class not found"}))
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        await websocket.send(json.dumps({"error": "cannot open video"}))
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / PROCESS_FPS)))
    frame_idx = 0

    print(f"[INFO] streaming at ~{src_fps/step:.2f} fps over WS ws://{WS_HOST}:{WS_PORT}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        t_cur = frame_idx / src_fps
        ts, phase = format_ts_and_phase(t_cur)

        n = detect_persons_tiled(
            frame, model, person_idx,
            rows=ROWS, cols=COLS,
            conf=YOLO_CONF, yolo_iou=YOLO_IOU, nms_iou=NMS_IOU
        )

        msg = {"count": n, "ts": ts, "phase": phase}
        await websocket.send(json.dumps(msg))

        frame_idx += 1
        if t_cur >= 15.0:   # ตัดที่ 15 วิพอดี
            break

    cap.release()
    print("[INFO] stream finished")

async def ws_handler(websocket):
    # ตัวอย่างนี้: client ไม่ต้องส่งอะไรมา แค่เชื่อมแล้วรอรับ
    await stream_counts(websocket)

async def main():
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT, ping_interval=20, ping_timeout=20):
        print(f"[WS] listening on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
