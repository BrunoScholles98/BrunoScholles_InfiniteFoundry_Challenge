import numpy as np
from typing import Dict

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
