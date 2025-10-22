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

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_AA = cv2.LINE_AA

# Colors (BGR)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
GOLD = (0, 215, 255)
GREEN = (0, 255, 100)
MINT = (170, 255, 200)
PINK = (255, 105, 180)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

PILE_LINE_Y = 715


# -------------------------
# HELPERS
# -------------------------
def count_piece_in_box(num_hands: int,
                       only_one_hand_frames: int,
                       cooldown_frames: int,
                       counter: int,
                       fps: float,
                       alert: Dict) -> Tuple[int, int, int, Dict]:
    """Counts a piece dropped in the box when only one hand is visible for ~14 frames with cooldown."""
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
    """Draws blue arrow + text when a piece is detected in the box."""
    if not alert["active"]:
        return frame

    elapsed = time.time() - alert["start_time"]
    if elapsed > alert["duration"]:
        alert["active"] = False
        return frame

    h, w = frame.shape[:2]

    arrow_h = 160
    color = BLUE
    thickness = 25

    tip_y = h - 50
    base_y = tip_y - arrow_h
    base_x = 200
    tip = (base_x, tip_y)

    cv2.line(frame, (base_x, base_y), tip, color, thickness)
    pts = np.array([(base_x - 50, tip_y - 70), (base_x + 50, tip_y - 70), tip], np.int32)
    cv2.fillPoly(frame, [pts], color)
    cv2.putText(frame, "PIECE IN THE BOX", (base_x - 100, base_y - 40), FONT, 1.4, color, 5, LINE_AA)

    return frame


def identify_hands(boxes: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    """Returns dict with 'left' and 'right' hands (xyxy, ints). If one hand: assign to right."""
    if boxes is None or len(boxes) == 0:
        return {"left": None, "right": None}
    boxes = np.asarray(boxes)

    if len(boxes) == 1:
        return {"left": None, "right": boxes[0].astype(int)}

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    idx = np.argsort(cx)[:2]
    left_box, right_box = boxes[idx[0]].astype(int), boxes[idx[1]].astype(int)
    return {"left": left_box, "right": right_box}


def count_pieces_picked(hands: Dict[str, Optional[np.ndarray]],
                        picked_count: int,
                        above_prev: bool,
                        line_color: Tuple[int, int, int]) -> Tuple[int, bool, Tuple[int, int, int]]:
    """Counts a pick when left-hand centroid crosses the pile line from above to below."""
    above_now = above_prev
    if hands["left"] is not None:
        x1, y1, x2, y2 = hands["left"]
        y_centroid = (y1 + y2) // 2
        above_now = y_centroid < PILE_LINE_Y
        line_color = MINT if above_now else PINK
        if above_prev and not above_now:
            picked_count += 1
    return picked_count, above_now, line_color


# -------------------------
# PEN MARK DETECTOR
# -------------------------
def detect_pen_mark(hands: Dict[str, Optional[np.ndarray]],
                    img: np.ndarray,
                    state: Dict) -> Tuple[Dict, np.ndarray]:
    """Detects pen-mark gesture: right-hand oscillation near a stable left hand."""
    # Tunables
    DIST_THRESHOLD = 500
    WINDOW = 60
    MIN_SIGN_CHANGES = 2
    MAX_SIGN_CHANGES = 6
    AMP_THRESHOLD = 120
    LEFT_STABLE_STD = 10
    COOLDOWN_TIME = 1.0
    HIGHLIGHT_TIME = 1.0

    def left_stable(hist_esq: deque) -> bool:
        if len(hist_esq) < 5:
            return False
        arr = np.asarray(hist_esq, dtype=np.float32)
        return np.mean(np.std(arr, axis=0)) < LEFT_STABLE_STD

    def oscillatory(dist_hist: deque) -> bool:
        if len(dist_hist) < 10:
            return False
        arr = np.asarray(dist_hist, dtype=np.float32)
        diffs = np.diff(arr)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        amplitude = np.ptp(arr)
        return (MIN_SIGN_CHANGES <= sign_changes <= MAX_SIGN_CHANGES) and (amplitude <= AMP_THRESHOLD)

    now = time.time()

    left, right = hands["left"], hands["right"]
    if left is not None and right is not None:
        x1l, y1l, x2l, y2l = left
        x1r, y1r, x2r, y2r = right
        c_left = np.array([(x1l + x2l) // 2, (y1l + y2l) // 2], dtype=np.int32)
        c_right = np.array([(x1r + x2r) // 2, (y1r + y2r) // 2], dtype=np.int32)

        dist = float(np.linalg.norm(c_left - c_right))
        state["dist_hist"].append(dist)
        state["left_hist"].append(c_left)

        if dist <= DIST_THRESHOLD:
            cv2.line(img, tuple(c_left), tuple(c_right), ORANGE, 3)
            if left_stable(state["left_hist"]):
                if now - state["last_detection_time"] >= COOLDOWN_TIME and oscillatory(state["dist_hist"]):
                    state["count"] += 2
                    state["last_detection_time"] = now
                    state["last_highlight_time"] = now
                    state["dist_hist"].clear()
                    cv2.putText(img, "Pen Mark Detected!", (700, 80), FONT, 1.2, ORANGE, 3, LINE_AA)
        else:
            state["dist_hist"].clear()
            state["left_hist"].clear()
    else:
        state["dist_hist"].clear()
        state["left_hist"].clear()

    if (now - state["last_highlight_time"]) <= HIGHLIGHT_TIME and right is not None:
        x1r, y1r, x2r, y2r = right
        cv2.rectangle(img, (x1r, y1r), (x2r, y2r), ORANGE, 5)
        cv2.putText(img, "hand_Right_pen_scratch", (x1r, y1r - 10), FONT, 1.0, ORANGE, 3, LINE_AA)

    return state, img


# -------------------------
# PROBE PASS DETECTOR (with visual trace)
# -------------------------
def detect_probe_pass(hands: Dict[str, Optional[np.ndarray]],
                      img: np.ndarray,
                      state: Dict) -> Tuple[Dict, np.ndarray]:
    """Detects a two-phase probe pass interaction when hands are close, with follow-up trace."""
    now = time.time()
    left, right = hands["left"], hands["right"]

    if left is not None and right is not None:
        x1l, y1l, x2l, y2l = left
        x1r, y1r, x2r, y2r = right

        c_left = np.array([(x1l + x2l) // 2, (y1l + y2l) // 2], dtype=np.int32)
        c_right = np.array([(x1r + x2r) // 2, (y1r + y2r) // 2], dtype=np.int32)
        centers_dist = float(np.linalg.norm(c_left - c_right))

        # Points of interest: top-left vertex of right hand -> mid of right edge of left hand
        v_tl_right = np.array([x1r, y1r], dtype=np.int32)
        mid_right_edge_left = np.array([x2l, (y1l + y2l) // 2], dtype=np.int32)

        if "last_vertex" not in state:
            state["last_vertex"] = v_tl_right.copy()
            state["prev_dist"] = centers_dist
            state["trace"] = []

        # Start/stop monitoring by proximity
        if centers_dist <= 500 and not state["monitoring"]:
            state["monitoring"] = True
            state["phase"] = 0
            state["trace"].clear()

        if centers_dist > 500:
            state["monitoring"] = False
            state["phase"] = 0
            state["followup_active"] = False
            state["trace"].clear()
            return state, img

        if state["monitoring"]:
            # Distance from right's top-left vertex to left-hand box (0 if inside)
            dx = max(0, x1l - v_tl_right[0], v_tl_right[0] - x2l)
            dy = max(0, y1l - v_tl_right[1], v_tl_right[1] - y2l)
            dist_to_left_box = float(np.hypot(dx, dy))

            # Phase 0: first contact
            if state["phase"] == 0:
                if dist_to_left_box <= 15:
                    state["count"] += 1
                    state["phase"] = 1
                    state["last_detection_time"] = now
                    state["dist_after_first"] = centers_dist
                    state["followup_active"] = True
                    cv2.putText(img, "Probe Pass +1 (phase1)", (700, 130), FONT, 1.2, GREEN, 3, LINE_AA)
                else:
                    # Green guide: right TL vertex -> mid of left right edge (as requested)
                    cv2.line(img, tuple(v_tl_right), tuple(mid_right_edge_left), GREEN, 2)

            # Phase 1: follow-up separation + vertex movement
            elif state["phase"] == 1 and state["followup_active"]:
                moved_apart = centers_dist - state["dist_after_first"] >= 40
                vertex_moved = float(np.linalg.norm(v_tl_right - state["last_vertex"])) >= 10

                state["trace"].append(tuple(v_tl_right))
                if len(state["trace"]) > 2:
                    for i in range(1, len(state["trace"])):
                        cv2.line(img, state["trace"][i - 1], state["trace"][i], GREEN, 3)

                if moved_apart and vertex_moved and centers_dist <= 500:
                    state["count"] += 1
                    state["phase"] = 2
                    state["cooldown_until"] = now + 1.5
                    state["followup_active"] = False
                    state["trace"].clear()
                    cv2.putText(img, "Probe Pass +1 (phase2)", (700, 160), FONT, 1.2, GREEN, 3, LINE_AA)

            # Follow-up timeout
            if state["followup_active"] and (now - state["last_detection_time"]) > 1.5:
                state["followup_active"] = False
                state["phase"] = 0
                state["monitoring"] = False
                state["trace"].clear()

        # Cooldown reset
        if state["phase"] == 2 and now > state["cooldown_until"]:
            state["phase"] = 0
            state["monitoring"] = False

        state["last_vertex"] = v_tl_right.copy()
        state["prev_dist"] = centers_dist

    # Highlight right hand briefly after detection
    if (now - state["last_detection_time"]) <= 1.0 and hands["right"] is not None:
        x1r, y1r, x2r, y2r = hands["right"]
        cv2.rectangle(img, (x1r, y1r), (x2r, y2r), GREEN, 5)
        cv2.putText(img, "hand_Right_probe_pass", (x1r, y1r - 10), FONT, 1.0, GREEN, 3, LINE_AA)

    return state, img


# -------------------------
# MAIN
# -------------------------
def main() -> None:
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

    # Counters/state
    in_box_count = 0
    only_one_hand_frames = 0
    cooldown_frames = 0
    picked_count = 0
    above_prev = False
    line_color = PINK

    alert = {"active": False, "start_time": 0.0, "duration": 1.5}

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
    }

    progress = tqdm(total=total, desc="Processing video", unit="frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run model (silent)
        results = model.predict(source=frame, device=DEVICE, conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 4))

        hands = identify_hands(boxes)
        num_hands = (hands["left"] is not None) + (hands["right"] is not None)

        # Count: piece in box
        in_box_count, only_one_hand_frames, cooldown_frames, alert = count_piece_in_box(
            num_hands, only_one_hand_frames, cooldown_frames, in_box_count, fps, alert
        )

        # Count: pieces picked
        picked_count, above_prev, line_color = count_pieces_picked(hands, picked_count, above_prev, line_color)

        annotated = frame  # draw in-place (no extra copy)

        # Draw hands
        if hands["left"] is not None:
            x1, y1, x2, y2 = hands["left"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), YELLOW, 3)
            cv2.putText(annotated, "hand_Left", (x1, y1 - 10), FONT, 1.0, YELLOW, 3, LINE_AA)

        if hands["right"] is not None:
            x1, y1, x2, y2 = hands["right"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 3)
            cv2.putText(annotated, "hand_Right", (x1, y1 - 10), FONT, 1.0, (255, 200, 0), 3, LINE_AA)

        # Detectors
        penmark_state, annotated = detect_pen_mark(hands, annotated, penmark_state)
        probe_state, annotated = detect_probe_pass(hands, annotated, probe_state)

        # Reference line
        cv2.line(annotated, (0, PILE_LINE_Y), (width, PILE_LINE_Y), line_color, 3)

        # Alert (piece in box)
        annotated = draw_piece_in_box_alert(annotated, alert)

        # Stats panel
        cv2.rectangle(annotated, (50, 50), (550, 360), WHITE, -1)
        cv2.putText(annotated, f"Pieces in the box - {in_box_count}", (70, 100), FONT, 1.2, BLUE, 3, LINE_AA)
        cv2.putText(annotated, f"Pieces Picked - {picked_count}", (70, 180), FONT, 1.2, PINK, 3, LINE_AA)
        cv2.putText(annotated, f"Pen Mark - {penmark_state['count']}", (70, 260), FONT, 1.2, ORANGE, 3, LINE_AA)
        cv2.putText(annotated, f"Probe Passes - {probe_state['count']}", (70, 340), FONT, 1.2, GREEN, 3, LINE_AA)

        out.write(annotated)
        progress.update(1)

    progress.close()
    cap.release()
    out.release()
    print(f"[✅] Video saved at: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()