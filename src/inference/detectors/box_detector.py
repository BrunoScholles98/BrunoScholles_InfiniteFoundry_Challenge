import time
from typing import Dict, Tuple


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
