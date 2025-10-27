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