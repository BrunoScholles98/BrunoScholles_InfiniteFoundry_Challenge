from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import torch
import time
from tqdm import tqdm
from collections import deque
from typing import Dict, Optional, Tuple

# -------------------------
# CONSTANTS
# -------------------------
MODEL_PATH = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/trained_models/yolov12n_hands/yolov12_hands_run/weights/best.pt")
INPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/tarefas_cima_Trim.mp4")
OUTPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/outupt_video/output_video_detected.mp4")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5


# -------------------------
# HELPERS
# -------------------------
def count_piece_in_box(num_hands: int,
                       only_one_hand_frames: int,
                       cooldown_frames: int,
                       counter: int,
                       fps: float,
                       alert: Dict) -> Tuple[int, int, int, Dict]:
    cooldown_duration = int(2 * fps)
    only_one_hand_frames = only_one_hand_frames + 1 if num_hands == 1 else 0
    cooldown_frames = max(0, cooldown_frames - 1)

    if only_one_hand_frames >= 14 and cooldown_frames == 0:
        counter += 1
        cooldown_frames = cooldown_duration
        only_one_hand_frames = 0
        alert["active"] = True
        alert["start_time"] = time.time()

    return counter, only_one_hand_frames, cooldown_frames, alert


def draw_piece_in_box_alert(frame: np.ndarray, alert: Dict) -> np.ndarray:
    if not alert["active"]:
        return frame
    elapsed = time.time() - alert["start_time"]
    if elapsed > alert["duration"]:
        alert["active"] = False
        return frame

    h, _ = frame.shape[:2]
    arrow_h = 160
    color = (255, 0, 0)
    thickness = 25

    tip_y = h - 50
    base_y = tip_y - arrow_h
    base_x = 200
    tip = (base_x, tip_y)
    cv2.line(frame, (base_x, base_y), tip, color, thickness)
    pts = np.array([(base_x - 50, tip_y - 70), (base_x + 50, tip_y - 70), tip], np.int32)
    cv2.fillPoly(frame, [pts], color)
    cv2.putText(frame, "PIECE IN THE BOX", (base_x - 100, base_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 5, cv2.LINE_AA)
    return frame


def identify_hands(boxes: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    if boxes is None or len(boxes) == 0:
        return {"left": None, "right": None}
    boxes = np.asarray(boxes)
    if len(boxes) == 1:
        return {"left": None, "right": boxes[0].astype(int)}

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    idx = np.argsort(cx)[:2]
    left_box, right_box = boxes[idx[0]].astype(int), boxes[idx[1]].astype(int)
    return {"left": left_box, "right": right_box}


def count_pieces_picked(hands, picked_count, above_prev, line_color):
    above_now = above_prev
    if hands["left"] is not None:
        x1, y1, x2, y2 = hands["left"]
        y_centroid = (y1 + y2) // 2
        above_now = y_centroid < 715
        line_color = (170, 255, 200) if above_now else (255, 105, 180)
        if above_prev and not above_now:
            picked_count += 1
    return picked_count, above_now, line_color


# -------------------------
# NEW: OPERATION TIMER (separate from main)
# -------------------------
def update_operation_timer(picked_count: int, in_box_count: int, op_state: Dict) -> Dict:
    """
    Tracks and updates the average duration of a complete operation.

    Definition (per request):
    - Start: when the hand crosses the line to pick the piece (we use the increment in 'Pieces Picked').
    - End: when the piece is dropped into the box (we use the increment in 'Pieces in the box').

    The function:
    - Detects start/end events by comparing current counters to the last seen.
    - On a completed cycle, appends the duration to a rolling buffer and updates the average.
    """
    now = time.time()

    # Detect start (hand crosses the line -> picked_count increment)
    if picked_count > op_state["last_picked_count"]:
        # Only start a new operation if one isn't already running
        if op_state["current_start_time"] is None:
            op_state["current_start_time"] = now

    # Detect end (piece dropped into box -> in_box_count increment)
    if in_box_count > op_state["last_in_box_count"]:
        if op_state["current_start_time"] is not None:
            duration = now - op_state["current_start_time"]
            op_state["durations"].append(duration)
            # Update average time across completed operations
            if len(op_state["durations"]) > 0:
                op_state["avg_time"] = float(np.mean(op_state["durations"]))
            else:
                op_state["avg_time"] = 0.0
            # Reset current operation
            op_state["current_start_time"] = None

    # Update last seen counters
    op_state["last_picked_count"] = picked_count
    op_state["last_in_box_count"] = in_box_count

    return op_state


# -------------------------
# PEN MARK DETECTOR
# -------------------------
def detect_pen_mark(hands, img, state):
    def left_stable(hist_esq):
        if len(hist_esq) < 5:
            return False
        arr = np.asarray(hist_esq, dtype=np.float32)
        return np.mean(np.std(arr, axis=0)) < 10

    def oscillatory(dist_hist):
        if len(dist_hist) < 10:
            return False
        arr = np.asarray(dist_hist, dtype=np.float32)
        diffs = np.diff(arr)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        amplitude = np.ptp(arr)
        return (2 <= sign_changes <= 6) and (amplitude <= 120)

    now = time.time()
    left, right = hands["left"], hands["right"]

    if left is not None and right is not None:
        x1l, y1l, x2l, y2l = left
        x1r, y1r, x2r, y2r = right
        c_left = np.array([(x1l + x2l) // 2, (y1l + y2l) // 2])
        c_right = np.array([(x1r + x2r) // 2, (y1r + y2r) // 2])

        dist = float(np.linalg.norm(c_left - c_right))
        state["dist_hist"].append(dist)
        state["left_hist"].append(c_left)

        if dist <= 500:
            cv2.line(img, tuple(c_left), tuple(c_right), (0, 165, 255), 3)
            if left_stable(state["left_hist"]):
                if now - state["last_detection_time"] >= 1.0 and oscillatory(state["dist_hist"]):
                    state["count"] += 2
                    state["last_detection_time"] = now
                    state["last_highlight_time"] = now
                    state["dist_hist"].clear()
                    cv2.putText(img, "Pen Mark Detected!", (700, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA)
        else:
            state["dist_hist"].clear()
            state["left_hist"].clear()
    else:
        state["dist_hist"].clear()
        state["left_hist"].clear()

    if (now - state["last_highlight_time"]) <= 1.0 and right is not None:
        x1r, y1r, x2r, y2r = right
        cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 165, 255), 5)
        cv2.putText(img, "hand_Right_pen_scratch", (x1r, y1r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3, cv2.LINE_AA)

    return state, img


# -------------------------
# PROBE PASS DETECTOR
# -------------------------
def detect_probe_pass(hands, img, state):
    now = time.time()
    left, right = hands["left"], hands["right"]

    state.setdefault("delay_frames", 0)
    state.setdefault("prev_crop", None)
    state.setdefault("phase0_green_frames", 0)
    state.setdefault("roi_green_until", 0.0)
    state.setdefault("hand_phase1_green_until", 0.0)

    if now < state.get("cooldown_until", 0):
        return state, img

    if left is None or right is None:
        state["phase"] = 0
        state["monitoring"] = False
        state["delay_frames"] = 0
        state["prev_crop"] = None
        state["phase0_green_frames"] = 0
        return state, img

    x1l, y1l, x2l, y2l = left
    x1r, y1r, x2r, y2r = right
    c_left = np.array([(x1l + x2l)//2, (y1l + y2l)//2])
    c_right = np.array([(x1r + x2r)//2, (y1r + y2r)//2])
    v_tl_right = np.array([x1r, y1r])
    centers_dist = float(np.linalg.norm(c_left - c_right))

    if centers_dist <= 500 and not state.get("monitoring", False):
        state["monitoring"] = True
        state["phase"] = 0
        state["delay_frames"] = 0
        state["prev_crop"] = None

    if centers_dist > 500 and state.get("phase", 0) == 0:
        state["monitoring"] = False
        state["phase"] = 0
        state["delay_frames"] = 0
        state["prev_crop"] = None
        state["phase0_green_frames"] = 0
        return state, img

    if state.get("monitoring", False) and state.get("phase", 0) in (0, 1):
        edge_points = [
            np.array([x2l, int(y1l + frac * (y2l - y1l))])
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        for pt in edge_points:
            cv2.line(img, tuple(v_tl_right), tuple(pt), (0, 255, 100), 2)

    # ----- FASE 0 -----
    if state["phase"] == 0 and state.get("monitoring", False):
        phase0_hand_should_be_red = True
        dx = max(0, x1l - v_tl_right[0], v_tl_right[0] - x2l)
        dy = max(0, y1l - v_tl_right[1], v_tl_right[1] - y2l)
        dist_to_left_box = float(np.hypot(dx, dy))

        if dist_to_left_box <= 15:
            state["count"] += 1
            state["phase"] = 1
            state["delay_frames"] = 0
            state["prev_crop"] = None
            state["last_detection_time"] = now
            state["phase0_green_frames"] = 14
            cv2.putText(img, "Probe Pass +1 (phase0)", (700, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3, cv2.LINE_AA)

        if state["phase"] == 0 and phase0_hand_should_be_red and right is not None:
            cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 0, 255), 5)

    # ----- FASE 1 -----
    if state["phase"] == 1 and state.get("monitoring", False):
        state["delay_frames"] += 1

        if state["phase0_green_frames"] > 0 and right is not None:
            cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 255, 0), 5)
            state["phase0_green_frames"] -= 1
        else:
            if right is not None:
                cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 0, 255), 5)

        # --- ROI só aparece após o cooldown de 14 frames ---
        if state["delay_frames"] > 14:
            roi_size = 80
            cx, cy = int(v_tl_right[0]), int(v_tl_right[1])
            x1c = max(cx - roi_size//2, 0)
            y1c = max(cy - roi_size//2, 0)
            x2c = min(cx + roi_size//2, img.shape[1])
            y2c = min(cy + roi_size//2, img.shape[0])
            crop = img[y1c:y2c, x1c:x2c].copy()

            roi_is_green = now <= state["roi_green_until"]
            roi_color = (0, 255, 0) if roi_is_green else (0, 0, 255)
            cv2.rectangle(img, (x1c, y1c), (x2c, y2c), roi_color, 2)
            cv2.putText(img, "roi", (x1c, max(0, y1c - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2, cv2.LINE_AA)

            if state["prev_crop"] is not None and crop.size > 0:
                diff = cv2.absdiff(crop, state["prev_crop"])
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                motion = np.sum(gray) / (gray.shape[0] * gray.shape[1])
                if motion > 25:
                    state["count"] += 1
                    state["phase"] = 2
                    state["cooldown_until"] = now + 2.0
                    state["monitoring"] = False
                    state["delay_frames"] = 0
                    state["prev_crop"] = None
                    state["last_detection_time"] = now
                    state["hand_phase1_green_until"] = now + 1.0
                    state["roi_green_until"] = now + 1.0
                    cv2.putText(img, "Probe Pass +1 (phase1 - insertion)", (700, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3, cv2.LINE_AA)

            state["prev_crop"] = crop

    # ----- DESTACA MÃO VERDE PÓS-DETECÇÃO -----
    if (now - state.get("last_detection_time", 0)) <= 1.0 and right is not None:
        x1r, y1r, x2r, y2r = right
        cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 255, 0), 5)
        cv2.putText(img, "hand_Right_probe_pass", (x1r, y1r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

    return state, img


# -------------------------
# MAIN
# -------------------------
def main():
    print(f"[OK] Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    model.to(DEVICE)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {INPUT_VIDEO}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

    in_box_count = 0
    only_one_hand_frames = 0
    cooldown_frames = 0
    picked_count = 0
    above_prev = False
    line_color = (255, 105, 180)

    alert = {"active": False, "start_time": 0.0, "duration": 1.0}

    penmark_state = {
        "count": 0,
        "dist_hist": deque(maxlen=60),
        "left_hist": deque(maxlen=60),
        "last_detection_time": 0.0,
        "last_highlight_time": -10.0,
    }

    probe_state = {
        "count": 0,
        "monitoring": False,
        "phase": 0,
        "followup_active": False,
        "last_detection_time": 0.0,
        "cooldown_until": 0.0,
        "phase0_green_frames": 0,
        "roi_green_until": 0.0,
        "hand_phase1_green_until": 0.0,
        "delay_frames": 0,
        "prev_crop": None,
    }

    # NEW: state to track average operation time (rolling buffer to smooth noise)
    op_state = {
        "current_start_time": None,
        "last_picked_count": 0,
        "last_in_box_count": 0,
        "durations": deque(maxlen=100),  # guarda as últimas 100 operações completas
        "avg_time": 0.0,
    }

    progress = tqdm(total=total, desc="Processing video", unit="frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, device=DEVICE, conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 4))
        hands = identify_hands(boxes)

        if hands["right"] is not None:
            x1r, _, x2r, _ = hands["right"]
            if x2r < 850:
                hands["right"] = None

        num_hands = (hands["left"] is not None) + (hands["right"] is not None)

        in_box_count, only_one_hand_frames, cooldown_frames, alert = count_piece_in_box(
            num_hands, only_one_hand_frames, cooldown_frames, in_box_count, fps, alert
        )

        picked_count, above_prev, line_color = count_pieces_picked(hands, picked_count, above_prev, line_color)
        annotated = frame

        if hands["left"] is not None:
            x1, y1, x2, y2 = hands["left"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(annotated, "hand_Left", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)

        if hands["right"] is not None:
            x1, y1, x2, y2 = hands["right"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 3)
            cv2.putText(annotated, "hand_Right", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 3, cv2.LINE_AA)

        penmark_state, annotated = detect_pen_mark(hands, annotated, penmark_state)
        probe_state, annotated = detect_probe_pass(hands, annotated, probe_state)

        cv2.line(annotated, (0, 715), (width, 715), line_color, 3)
        cv2.putText(annotated, "pick_line", (50, 705),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, line_color, 3, cv2.LINE_AA)
        
        annotated = draw_piece_in_box_alert(annotated, alert)

        # NEW: update average operation time based on counters
        op_state = update_operation_timer(picked_count, in_box_count, op_state)

        cv2.rectangle(annotated, (50, 50), (500, 350), (255, 255, 255), -1)

        cv2.putText(annotated, f"Pieces in the box - {in_box_count}", (70, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, f"Pieces Picked - {picked_count}", (70, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 105, 180), 3, cv2.LINE_AA)
        cv2.putText(annotated, f"Pen Mark - {penmark_state['count']}", (70, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated, f"Probe Passes - {probe_state['count']}", (70, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 3, cv2.LINE_AA)
        cv2.putText(annotated, f"Avg Complete Operation Time - {op_state['avg_time']:.2f}s", (70, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        out.write(annotated)
        progress.update(1)

    progress.close()
    cap.release()
    out.release()
    print(f"[✅] Video saved at: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
