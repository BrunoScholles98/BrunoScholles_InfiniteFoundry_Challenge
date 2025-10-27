import time
import numpy as np
from collections import deque
import cv2


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