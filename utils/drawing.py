import cv2
import numpy as np
import time
from typing import Dict, Tuple

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