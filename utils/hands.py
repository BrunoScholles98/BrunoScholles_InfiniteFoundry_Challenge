import numpy as np
from typing import Dict, Optional

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