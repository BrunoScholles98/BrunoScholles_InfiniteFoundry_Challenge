"""
Usage example (in my case):
python main.py \
    --base_dir /mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/challenge_hands/train \
    --weights /mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/YOLOv12_Baseline_Weights/yolov12n.pt \
    --output_dir /mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/trained_models/yolov12n_hands_new
"""

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

# Import detectors from the detectors package
from detectors.penmark_detector import detect_pen_mark
from detectors.probe_pass_detector import detect_probe_pass
from detectors.box_detector import detect_pieces_in_box
from detectors.pick_detector import detect_pieces_picked

# Import helpers from utils
from utils.hands import identify_hands
from utils.metrics import update_operation_metrics
from utils.drawing import draw_metrics_panel, draw_piece_in_box_alert


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5


def main():
    
    # Determine script path to build fallback paths dynamically
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    # Command-line arguments with defaults in the script directory
    base_dir = Path(__file__).resolve().parent
    default_model = script_dir.parent.parent / "trained_models" / "yolov12n_hands" / "yolov12_hands_run" / "weights" / "best.pt"
    default_input = script_dir.parent.parent / "tarefas_cima.mp4"
    default_output = script_dir.parent.parent / "results" / "output_video_detections.mp4"

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