from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import torch
import time
from tqdm import tqdm
from collections import deque
from typing import Dict, Optional, Tuple
import argparse


# =========================
# CONSTANTS
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5


# =========================
# HELPERS
# =========================
def update_operation_metrics(
    picked_count: int,
    in_box_count: int,
    penmark_count: int,
    probepass_count: int,
    t_video_sec: float,
    op_state: Dict
) -> Dict:
    # Count deltas only (robust against duplicate frames or re-draws)
    if penmark_count > op_state["last_penmark_count"]:
        op_state["penmark_actions"] += (penmark_count - op_state["last_penmark_count"])
    if probepass_count > op_state["last_probepass_count"]:
        op_state["probepass_actions"] += (probepass_count - op_state["last_probepass_count"])
    if picked_count > op_state["last_picked_count"]:
        op_state["picked_actions"] += (picked_count - op_state["last_picked_count"])
    if in_box_count > op_state["last_in_box_count"]:
        op_state["in_box_actions"] += (in_box_count - op_state["last_in_box_count"])

    # Operation starts on first pick after idle (pairs pick → place)
    if picked_count > op_state["last_picked_count"]:
        if op_state["current_start_time_sec"] is None:
            op_state["current_start_time_sec"] = t_video_sec
            # Snapshot baselines to assess if this operation included penmark/probe
            op_state["op_snapshot"] = {"penmark": penmark_count, "probepass": probepass_count}

    # Operation ends when a piece is placed in the box after a start
    if in_box_count > op_state["last_in_box_count"]:
        if op_state["current_start_time_sec"] is not None:
            duration = max(0.0, t_video_sec - op_state["current_start_time_sec"])
            op_state["durations_sec"].append(duration)
            op_state["avg_time"] = float(np.mean(op_state["durations_sec"])) if len(op_state["durations_sec"]) > 0 else 0.0

            op_state["total_complete_operations"] += 1
            # If counters advanced during this op, mark success per operation
            if penmark_count - op_state["op_snapshot"]["penmark"] > 0:
                op_state["ops_with_penmark"] += 1
            if probepass_count - op_state["op_snapshot"]["probepass"] > 0:
                op_state["ops_with_probepass"] += 1

            # Reset to wait for next pick
            op_state["current_start_time_sec"] = None
            op_state["op_snapshot"] = {"penmark": penmark_count, "probepass": probepass_count}

    # Persist last seen totals for next delta computation
    op_state["last_picked_count"] = picked_count
    op_state["last_in_box_count"] = in_box_count
    op_state["last_penmark_count"] = penmark_count
    op_state["last_probepass_count"] = probepass_count

    return op_state


def identify_hands(boxes: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    # Select up to two boxes by x-center; left-most → "left", next → "right"
    if boxes is None or len(boxes) == 0:
        return {"left": None, "right": None}
    boxes = np.asarray(boxes)
    if len(boxes) == 1:
        return {"left": None, "right": boxes[0].astype(int)}
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    idx = np.argsort(cx)[:2]
    left_box, right_box = boxes[idx[0]].astype(int), boxes[idx[1]].astype(int)
    return {"left": left_box, "right": right_box}


def draw_metrics_panel(
    frame: np.ndarray,
    picked_count: int,
    in_box_count: int,
    penmark_count: int,
    probepass_count: int,
    op_state: Dict,
    line_color: Tuple[int, int, int]
) -> np.ndarray:
    # Consolidate live metrics and draw a stable HUD on top of the video frame
    total_actions = (
        op_state["penmark_actions"]
        + op_state["probepass_actions"]
        + op_state["picked_actions"]
        + op_state["in_box_actions"]
    )
    penmark_pct_actions = (op_state["penmark_actions"] / total_actions * 100.0) if total_actions > 0 else 0.0
    probe_pct_actions = (op_state["probepass_actions"] / total_actions * 100.0) if total_actions > 0 else 0.0

    total_ops = op_state["total_complete_operations"]
    penmark_pct_ops = (op_state["ops_with_penmark"] / total_ops * 100.0) if total_ops > 0 else 0.0
    probe_pct_ops = (op_state["ops_with_probepass"] / total_ops * 100.0) if total_ops > 0 else 0.0

    # Semi-transparent panel for readability on varying backgrounds
    overlay = frame.copy()
    panel_top_left = (50, 50)
    panel_bottom_right = (650, 480)
    cv2.rectangle(overlay, panel_top_left, panel_bottom_right, (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Vertically spaced rows
    base_y = 100
    dy = 35

    # Event counters (color-coded to match on-frame cues)
    cv2.putText(frame, f"Pieces Picked - {picked_count}", (70, base_y + 0 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 105, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Probe Pass Events - {probepass_count}", (70, base_y + 1 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Pen Mark Events - {penmark_count}", (70, base_y + 2 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Pieces Placed in the Box - {in_box_count}", (70, base_y + 3 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)

    y0 = base_y + 4 * dy
    cv2.putText(frame, f"Average Operation Duration - {op_state['avg_time']:.2f}s", (70, y0 + 0 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Marking Rate (per Action) - {penmark_pct_actions:.1f}%", (70, y0 + 1 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Marking Success Rate (per Operation) - {penmark_pct_ops:.1f}%", (70, y0 + 2 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Probe Pass Rate (per Action) - {probe_pct_actions:.1f}%", (70, y0 + 3 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Probe Pass Success Rate (per Operation) - {probe_pct_ops:.1f}%", (70, y0 + 4 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total Detected Actions - {total_actions}", (70, y0 + 5 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total Completed Operations - {total_ops}", (70, y0 + 6 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    return frame


def draw_piece_in_box_alert(frame: np.ndarray, alert: Dict) -> np.ndarray:
    # Early exit if no fresh "placed" event is active
    if not alert["active"]:
        return frame

    # Auto-expire the visual cue after a short duration
    elapsed = time.time() - alert["start_time"]
    if elapsed > alert["duration"]:
        alert["active"] = False
        return frame

    # Arrow points down to the box area to draw the operator's attention
    h, _ = frame.shape[:2]
    arrow_h = 160                      # tall enough to be visible over clutter
    color = (255, 0, 0)
    thickness = 25

    # Fixed anchor at bottom-left quadrant keeps the cue stable across frames
    tip_y = h - 50
    base_y = tip_y - arrow_h
    base_x = 200
    tip = (base_x, tip_y)

    # Shaft
    cv2.line(frame, (base_x, base_y), tip, color, thickness)

    # Triangular head
    pts = np.array([(base_x - 50, tip_y - 70), (base_x + 50, tip_y - 70), tip], np.int32)
    cv2.fillPoly(frame, [pts], color)

    # Short text label; large font for readability on video
    cv2.putText(
        frame, "PIECE IN THE BOX", (base_x - 100, base_y - 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 5, cv2.LINE_AA
    )
    return frame


def detect_pieces_in_box(
    num_hands: int,
    only_one_hand_frames: int,
    cooldown_frames: int,
    counter: int,
    fps: float,
    alert: Dict
) -> Tuple[int, int, int, Dict]:

    cooldown_duration = int(2 * fps)      # ~2 seconds at the current FPS
    only_one_hand_frames = only_one_hand_frames + 1 if num_hands == 1 else 0
    cooldown_frames = max(0, cooldown_frames - 1)

    # Single-hand streak qualifies as a "placed in box" event when cooldown is over
    if only_one_hand_frames >= 14 and cooldown_frames == 0:
        counter += 1
        cooldown_frames = cooldown_duration
        only_one_hand_frames = 0

        # Arm a short-lived UI cue for operator feedback
        alert["active"] = True
        alert["start_time"] = time.time()

    return counter, only_one_hand_frames, cooldown_frames, alert


def detect_pieces_picked(hands, picked_count, above_prev, line_color):
    above_now = above_prev
    if hands["left"] is not None:
        x1, y1, x2, y2 = hands["left"]
        y_centroid = (y1 + y2) // 2

        # True while hand center is above the gate line
        above_now = y_centroid < 715

        # Visual feedback: greenish above, pink below
        line_color = (170, 255, 200) if above_now else (255, 105, 180)

        # Edge-trigger: only count the downward transition
        if above_prev and not above_now:
            picked_count += 1

    return picked_count, above_now, line_color


# =========================
# PEN MARK DETECTOR
# =========================
def detect_pen_mark(hands, img, state):

    def left_stable(hist_esq):
        # Stability gate to avoid counting during large drifts
        if len(hist_esq) < 5:
            return False
        arr = np.asarray(hist_esq, dtype=np.float32)
        return np.mean(np.std(arr, axis=0)) < 10  # tight positional dispersion

    def oscillatory(dist_hist):
        # Look for a few sign changes and limited amplitude (small, quick wiggles)
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

        # Proximity + stability + oscillation → mark event
        if dist <= 500:
            cv2.line(img, tuple(c_left), tuple(c_right), (0, 165, 255), 3)
            if left_stable(state["left_hist"]):
                if now - state["last_detection_time"] >= 1.0 and oscillatory(state["dist_hist"]):
                    state["count"] += 2  # two scratches per cycle in this heuristic
                    state["last_detection_time"] = now
                    state["last_highlight_time"] = now
                    state["dist_hist"].clear()
                    cv2.putText(img, "Pen Mark Detected!", (700, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA)
        else:
            # Reset if hands separate
            state["dist_hist"].clear()
            state["left_hist"].clear()
    else:
        # Reset when one of the hands disappears
        state["dist_hist"].clear()
        state["left_hist"].clear()

    # Brief highlight on the right hand after detection
    if (now - state["last_highlight_time"]) <= 1.0 and right is not None:
        x1r, y1r, x2r, y2r = right
        cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 165, 255), 5)
        cv2.putText(img, "hand_Right_pen_scratch", (x1r, y1r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3, cv2.LINE_AA)

    return state, img


# =========================
# PROBE PASS DETECTOR
# =========================
def detect_probe_pass(hands, img, state):
    now = time.time()
    left, right = hands["left"], hands["right"]

    # Ensure all state keys exist (idempotent on first call)
    state.setdefault("delay_frames", 0)
    state.setdefault("prev_crop", None)
    state.setdefault("phase0_green_frames", 0)
    state.setdefault("roi_green_until", 0.0)
    state.setdefault("hand_phase1_green_until", 0.0)

    # Global cooldown
    if now < state.get("cooldown_until", 0):
        return state, img

    # Need both hands to reason about relative geometry
    if left is None or right is None:
        state["phase"] = 0
        state["monitoring"] = False
        state["delay_frames"] = 0
        state["prev_crop"] = None
        state["phase0_green_frames"] = 0
        state["roi_green_until"] = 0.0
        return state, img

    x1l, y1l, x2l, y2l = left
    x1r, y1r, x2r, y2r = right
    c_left = np.array([(x1l + x2l)//2, (y1l + y2l)//2])
    c_right = np.array([(x1r + x2r)//2, (y1r + y2r)//2])
    v_tl_right = np.array([x1r, y1r])  # visual anchor near right-hand top-left
    centers_dist = float(np.linalg.norm(c_left - c_right))

    # Start monitoring only when hands are sufficiently close
    if centers_dist <= 500 and not state.get("monitoring", False):
        state["monitoring"] = True
        state["phase"] = 0
        state["delay_frames"] = 0
        state["prev_crop"] = None

    # If they separate during Phase 0, cancel to reduce false positives
    if centers_dist > 500 and state.get("phase", 0) == 0:
        state["monitoring"] = False
        state["phase"] = 0
        state["delay_frames"] = 0
        state["prev_crop"] = None
        state["phase0_green_frames"] = 0
        state["roi_green_until"] = 0.0
        return state, img

    # Draw guidance lines while tracking (helps visual debugging on the overlay)
    if state.get("monitoring", False) and state.get("phase", 0) in (0, 1):
        edge_points = [
            np.array([x2l, int(y1l + frac * (y2l - y1l))])
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        for pt in edge_points:
            cv2.line(img, tuple(v_tl_right), tuple(pt), (0, 255, 100), 2)

    # ----- PHASE 0 -----
    if state["phase"] == 0 and state.get("monitoring", False):
        # Touch proximity between right-hand top-left and left-hand bbox → +1
        dx = max(0, x1l - v_tl_right[0], v_tl_right[0] - x2l)
        dy = max(0, y1l - v_tl_right[1], v_tl_right[1] - y2l)
        dist_to_left_box = float(np.hypot(dx, dy))

        if dist_to_left_box <= 15:
            state["count"] += 1
            state["phase"] = 1
            state["delay_frames"] = 0
            state["prev_crop"] = None
            state["last_detection_time"] = now
            state["phase0_green_frames"] = 14  # temporary green highlight
            cv2.putText(img, "Probe Pass +1 (phase0)", (700, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3, cv2.LINE_AA)

        # Right-hand stays red until phase 0 is satisfied
        if state["phase"] == 0 and right is not None:
            cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 0, 255), 5)

    # ----- PHASE 1 -----
    if state["phase"] == 1 and state.get("monitoring", False):
        # Abort if right hand drifts too far in x relative to the left hand
        dist_x = x1r - x2l
        if dist_x > 150:
            state["phase"] = 0
            state["monitoring"] = False
            state["delay_frames"] = 0
            state["prev_crop"] = None
            state["phase0_green_frames"] = 0
            state["roi_green_until"] = 0.0
            return state, img

        state["delay_frames"] += 1

        # Visual feedback around right hand after phase 0 trigger
        if state["phase0_green_frames"] > 0 and right is not None:
            cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 255, 0), 5)
            state["phase0_green_frames"] -= 1
        else:
            if right is not None:
                cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 0, 255), 5)

        # After a small delay, analyze a tiny ROI for motion → insertion → +1
        if state["delay_frames"] > 14:
            roi_size = 80
            cx, cy = int(v_tl_right[0]), int(v_tl_right[1])
            x1c = max(cx - roi_size // 2, 0)
            y1c = max(cy - roi_size // 2, 0)
            x2c = min(cx + roi_size // 2, img.shape[1])
            y2c = min(cy + roi_size // 2, img.shape[0])
            crop = img[y1c:y2c, x1c:x2c].copy()

            roi_is_green = now <= state["roi_green_until"]
            roi_color = (0, 255, 0) if roi_is_green else (0, 0, 255)
            cv2.rectangle(img, (x1c, y1c), (x2c, y2c), roi_color, 2)
            cv2.putText(img, "roi", (x1c, max(0, y1c - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2, cv2.LINE_AA)

            if state["prev_crop"] is not None and crop.size > 0:
                # Simple frame diff; threshold tuned to this scene
                diff = cv2.absdiff(crop, state["prev_crop"])
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                motion = np.sum(gray) / (gray.shape[0] * gray.shape[1])
                if motion > 25:
                    state["count"] += 1
                    state["phase"] = 2
                    state["cooldown_until"] = now + 2.0  # global cooldown
                    state["monitoring"] = False
                    state["delay_frames"] = 0
                    state["prev_crop"] = None
                    state["last_detection_time"] = now
                    state["hand_phase1_green_until"] = now + 1.0
                    state["roi_green_until"] = now + 1.0
                    cv2.putText(img, "Probe Pass +1 (phase1 - insertion)", (700, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3, cv2.LINE_AA)

            state["prev_crop"] = crop

    # Brief success highlight on right hand
    if (now - state.get("last_detection_time", 0)) <= 1.0 and right is not None:
        x1r, y1r, x2r, y2r = right
        cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (0, 255, 0), 5)
        cv2.putText(img, "hand_Right_probe_pass", (x1r, y1r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

    return state, img


# =========================
# MAIN
# =========================
def main():
    # Command-line arguments with defaults in the script directory
    base_dir = Path(__file__).resolve().parent
    default_model = base_dir / "trained_models" / "yolov12n_hands" / "yolov12_hands_run" / "weights" / "best.pt"
    default_input = base_dir / "tarefas_cima_Trim.mp4"
    default_output = base_dir / "output_video_detections.mp4"

    parser = argparse.ArgumentParser(description="Hand operations analytics with YOLO (paths via CLI).")
    parser.add_argument("--model_path", type=Path, default=default_model, help="Path to the .pt model")
    parser.add_argument("--input_video", type=Path, default=default_input, help="Path to the input video")
    parser.add_argument("--output_video", type=Path, default=default_output, help="Path to the output video")

    args = parser.parse_args()

    MODEL_PATH: Path = args.model_path
    INPUT_VIDEO: Path = args.input_video
    OUTPUT_VIDEO: Path = args.output_video

    print(f"[OK] Model loaded from: {MODEL_PATH}")
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

    # Live counters and UI state
    in_box_count = 0
    only_one_hand_frames = 0
    cooldown_frames = 0
    picked_count = 0
    above_prev = False
    line_color = (255, 105, 180)

    # Short flash to draw attention when placing in box
    alert = {"active": False, "start_time": 0.0, "duration": 1.0}

    # Pen-mark temporal features
    penmark_state = {
        "count": 0,
        "dist_hist": deque(maxlen=60),   # inter-hand distance history
        "left_hist": deque(maxlen=60),   # left-hand center positions
        "last_detection_time": 0.0,
        "last_highlight_time": -10.0,
    }

    # Probe-pass state machine (phase 0 → phase 1)
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

    # Operation-level metrics (timestamp-based, independent of FPS)
    op_state = {
        "current_start_time_sec": None,
        "op_snapshot": {"penmark": 0, "probepass": 0},
        "last_picked_count": 0,
        "last_in_box_count": 0,
        "last_penmark_count": 0,
        "last_probepass_count": 0,
        "durations_sec": deque(maxlen=100),
        "avg_time": 0.0,
        "penmark_actions": 0,
        "probepass_actions": 0,
        "picked_actions": 0,
        "in_box_actions": 0,
        "total_complete_operations": 0,
        "ops_with_penmark": 0,
        "ops_with_probepass": 0,
    }

    progress = tqdm(total=total, desc="Processing video", unit="frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Use file timestamp to avoid drift (ms → s)
        t_video_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0

        # Run detector on current frame (single-image mode)
        results = model.predict(source=frame, device=DEVICE, conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 4))
        hands = identify_hands(boxes)

        # Discard right-hand boxes too far left (scene-specific constraint)
        if hands["right"] is not None:
            x1r, _, x2r, _ = hands["right"]
            if x2r < 850:
                hands["right"] = None

        num_hands = (hands["left"] is not None) + (hands["right"] is not None)

        # "Place in box" heuristic based on single-hand visibility window
        in_box_count, only_one_hand_frames, cooldown_frames, alert = detect_pieces_in_box(
            num_hands, only_one_hand_frames, cooldown_frames, in_box_count, fps, alert
        )

        # "Pick" counting via crossing of a static horizontal line
        picked_count, above_prev, line_color = detect_pieces_picked(hands, picked_count, above_prev, line_color)
        annotated = frame

        # Draw hands if present (helps correlate counters with visuals)
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

        # Update per-event detectors (no logic changes)
        penmark_state, annotated = detect_pen_mark(hands, annotated, penmark_state)
        probe_state, annotated = detect_probe_pass(hands, annotated, probe_state)

        # Reference line for the pick trigger
        cv2.line(annotated, (0, 715), (width, 715), line_color, 3)
        cv2.putText(annotated, "pick_line", (50, 705),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, line_color, 3, cv2.LINE_AA)

        # Short-lived alert for "placed in box"
        annotated = draw_piece_in_box_alert(annotated, alert)

        # Update operation metrics based on absolute video time
        op_state = update_operation_metrics(
            picked_count=picked_count,
            in_box_count=in_box_count,
            penmark_count=penmark_state["count"],
            probepass_count=probe_state["count"],
            t_video_sec=t_video_sec,
            op_state=op_state
        )

        # Single function to render the HUD (keeps main loop tidy)
        annotated = draw_metrics_panel(
            frame=annotated,
            picked_count=picked_count,
            in_box_count=in_box_count,
            penmark_count=penmark_state["count"],
            probepass_count=probe_state["count"],
            op_state=op_state,
            line_color=line_color
        )

        out.write(annotated)
        progress.update(1)

    progress.close()
    cap.release()
    out.release()
    print(f"Processed video saved at: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()