import time
import numpy as np
import cv2


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