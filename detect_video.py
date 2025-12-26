#home.py
import os, time, csv
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
VIDEO_PATH = "/Users/chidimon/Dev/Crowd-Aware/source camera/source.mp4"
WEIGHTS    = "yolov8n.pt"      # หรือ yolov8s.pt
DEVICE     = None              # None / 'mps' (Mac Apple Silicon) / 'cuda' (NVIDIA) / 'cpu'
ROWS, COLS = 3, 3              # Tiling 3x3
YOLO_CONF  = 0.25
YOLO_IOU   = 0.5
NMS_IOU    = 0.5

# ประมวลผลกี่เฟรมต่อวินาที (เพื่อให้ลื่น) — จะข้ามเฟรมอัตโนมัติ
PROCESS_FPS = 7

# บันทึกผลเป็น CSV ท้ายรัน
SAVE_CSV = True
CSV_PATH = "crowd_counts.csv"

# =========================
# NMS (numpy)
# =========================
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

# =========================
# วาดป้ายมุมซ้ายบน
# =========================
def draw_topleft_label(img_bgr, lines, pad=8):
    img = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    th = 2

    # คำนวณกล่องครอบรวมหลายบรรทัด
    widths, heights = [], []
    for t in lines:
        (tw, th_text), base = cv2.getTextSize(t, font, scale, th)
        widths.append(tw); heights.append(th_text)
    box_w, box_h = max(widths) + pad * 2, sum(heights) + pad * (len(lines) + 1)

    x0, y0 = 10, 10
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)

    y = y0 + pad + heights[0]
    for t in lines:
        cv2.putText(img, t, (x0 + pad, y), font, scale, (255, 255, 255), th, cv2.LINE_AA)
        y += heights[0] + pad
    return img

# =========================
# ตรวจจับคนแบบ Tiling-only (คืนภาพวาดกล่อง + จำนวนคน)
# =========================
def detect_persons_tiled(img_bgr, model, person_idx, rows=3, cols=3, conf=0.25, yolo_iou=0.5, nms_iou=0.5):
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
        return draw_topleft_label(img_bgr, ["persons: 0"]), 0

    all_boxes = np.vstack(all_boxes)
    all_scores = np.concatenate(all_scores)
    keep_idx = nms_boxes(all_boxes, all_scores, iou_thresh=nms_iou)
    kept = all_boxes[keep_idx].astype(int)

    out = img_bgr.copy()
    for (x1, y1, x2, y2) in kept:
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

    out = draw_topleft_label(out, [f"persons: {len(kept)}"])
    return out, len(kept)

# =========================
# Helper: แปลงวินาทีเป็น mm:ss และชื่อช่วง (0–5, 5–10, 10–15)
# =========================
def format_ts_and_phase(t_sec):
    mm = int(t_sec // 60)
    ss = int(t_sec % 60)
    ts = f"{mm:02d}:{ss:02d}"
    if t_sec < 5:
        phase = "Dense (0–5s)"
    elif t_sec < 10:
        phase = "Medium (5–10s)"
    else:
        phase = "Sparse (10–15s)"
    return ts, phase

# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"ไม่พบไฟล์วิดีโอ: {VIDEO_PATH}")

    # โหลดโมเดล
    model = YOLO(WEIGHTS)
    if DEVICE is not None:
        try:
            model.to(DEVICE)
        except Exception as e:
            print(f"เตือน: ย้ายอุปกรณ์ไป {DEVICE} ไม่สำเร็จ → ใช้ดีฟอลต์แทน ({e})")

    # หา index ของคลาส person
    person_idx = None
    for k, v in model.names.items():
        if v == "person":
            person_idx = k
            break
    if person_idx is None:
        raise RuntimeError("ไม่พบคลาส 'person' ในโมเดลนี้")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("เปิดวิดีโอไม่สำเร็จ")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0
    step = max(1, int(round(src_fps / PROCESS_FPS)))

    print(f"[INFO] Video FPS ~ {src_fps:.2f}, ประมวลผลที่ ~{src_fps/step:.2f} FPS (ทุก {step} เฟรม)")

    # เก็บสถิติ
    counts = []
    per_phase = defaultdict(list)
    frame_idx = 0
    t_start = time.time()

    if SAVE_CSV:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "timestamp", "phase", "count"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ข้ามเฟรมตาม step
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # เวลาปัจจุบันของวิดีโอ (วินาที)
        t_cur = frame_idx / src_fps
        ts, phase = format_ts_and_phase(t_cur)

        t0 = time.time()
        vis, n = detect_persons_tiled(
            frame, model, person_idx,
            rows=ROWS, cols=COLS,
            conf=YOLO_CONF, yolo_iou=YOLO_IOU, nms_iou=NMS_IOU
        )
        dt = time.time() - t0
        rt_fps = 1.0 / max(dt, 1e-6)

        # วาด overlay เพิ่มเติม (เวลา/เฟส/FPS)
        lines = [
            f"persons: {n}",
            f"time: {ts}  |  {phase}",
            f"proc fps: {rt_fps:.1f}"
        ]
        vis = draw_topleft_label(vis, lines)

        # แสดงผล
        cv2.imshow("Crowd-Aware (press 'q' to quit)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # บันทึกสถิติ
        counts.append((t_cur, n))
        per_phase[phase].append(n)
        if SAVE_CSV:
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{t_cur:.3f}", ts, phase, n])

        # พิมพ์คอนโซลย่อ
        print(f"{ts} | {phase} | persons={n} | {rt_fps:.1f} FPS")

        frame_idx += 1

        # ถ้าคลิป 15 วิ อยากหยุดที่ 15 วิพอดี (กันเผื่อคลิปยาวกว่านี้)
        if t_cur >= 15.0:
            break

    cap.release()
    cv2.destroyAllWindows()

    # สรุปผล
    dur = time.time() - t_start
    if counts:
        overall_avg = sum(n for _, n in counts) / len(counts)
    else:
        overall_avg = 0.0

    print("\n=== SUMMARY ===")
    print(f"Processed frames: {len(counts)} in {dur:.2f}s")
    print(f"Overall avg persons: {overall_avg:.2f}")
    for phase in ["Dense (0–5s)", "Medium (5–10s)", "Sparse (10–15s)"]:
        vals = per_phase.get(phase, [])
        avg = sum(vals) / len(vals) if vals else 0.0
        print(f"{phase:<20} avg persons: {avg:.2f}  (n={len(vals)})")

    if SAVE_CSV:
        print(f"CSV saved: {CSV_PATH}")

if __name__ == "__main__":
    main()
