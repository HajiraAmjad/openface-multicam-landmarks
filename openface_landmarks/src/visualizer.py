"""
Landmark Visualisation
Draws 68-point landmarks on frames with group colour coding,
bounding boxes, indices, and connection lines — matches the style
of OpenFace's visualisation output.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# Colour palette per landmark group (BGR)
GROUP_COLORS = {
    "jaw":           (200, 200, 200),   # light grey
    "left_eyebrow":  (0, 200, 255),     # yellow
    "right_eyebrow": (0, 200, 255),
    "nose_bridge":   (0, 255, 128),     # green
    "nose_tip":      (0, 255, 128),
    "left_eye":      (255, 100, 0),     # blue
    "right_eye":     (255, 100, 0),
    "outer_lip":     (0, 60, 255),      # red
    "inner_lip":     (0, 100, 255),
}

# Connection lines per group — landmark index pairs
CONNECTIONS = {
    "jaw":           list(zip(range(0, 16), range(1, 17))),
    "left_eyebrow":  list(zip(range(17, 21), range(18, 22))),
    "right_eyebrow": list(zip(range(22, 26), range(23, 27))),
    "nose_bridge":   list(zip(range(27, 30), range(28, 31))),
    "nose_tip":      list(zip(range(31, 35), range(32, 36))) + [(35, 31)],
    "left_eye":      list(zip(range(36, 41), range(37, 42))) + [(41, 36)],
    "right_eye":     list(zip(range(42, 47), range(43, 48))) + [(47, 42)],
    "outer_lip":     list(zip(range(48, 59), range(49, 60))) + [(59, 48)],
    "inner_lip":     list(zip(range(60, 67), range(61, 68))) + [(67, 60)],
}

from src.landmark_extractor import LANDMARK_GROUPS


def draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,        # (68, 2)
    bbox: Optional[Tuple] = None,
    draw_indices: bool = False,
    draw_connections: bool = True,
    dot_radius: int = 2,
    line_thickness: int = 1,
    conf: Optional[float] = None,
) -> np.ndarray:
    """Return a copy of frame with landmarks drawn."""
    out = frame.copy()
    pts = landmarks.reshape(68, 2).astype(int)

    # Bounding box
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (100, 255, 100), 1)
        if conf is not None:
            cv2.putText(out, f"conf:{conf:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

    # Connection lines
    if draw_connections:
        for group, pairs in CONNECTIONS.items():
            color = GROUP_COLORS[group]
            for a, b in pairs:
                cv2.line(out, tuple(pts[a]), tuple(pts[b]), color, line_thickness)

    # Landmark dots
    for group, indices in LANDMARK_GROUPS.items():
        color = GROUP_COLORS[group]
        for idx in indices:
            cv2.circle(out, tuple(pts[idx]), dot_radius, color, -1)

    # Optional index numbers
    if draw_indices:
        for i, (px, py) in enumerate(pts):
            cv2.putText(out, str(i), (px + 2, py - 2),
                        cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
    return out


def draw_multi_face(
    frame: np.ndarray,
    landmarks_list: List[np.ndarray],
    bboxes: Optional[List[Tuple]] = None,
    **kwargs,
) -> np.ndarray:
    """Draw landmarks for multiple faces."""
    out = frame.copy()
    for i, lm in enumerate(landmarks_list):
        bbox = bboxes[i] if bboxes and i < len(bboxes) else None
        out = draw_landmarks(out, lm, bbox=bbox, **kwargs)
    return out


def create_landmark_grid(
    frames_with_lm: List[Tuple[np.ndarray, List[np.ndarray]]],
    grid_cols: int = 4,
    cell_size: Tuple[int, int] = (200, 200),
) -> np.ndarray:
    """
    Creates a grid image from (frame, [landmarks]) tuples.
    Useful for visualising a batch of images at once.
    """
    cells = []
    for frame, lm_list in frames_with_lm:
        vis = draw_multi_face(frame, lm_list)
        cells.append(cv2.resize(vis, cell_size))

    # Pad to fill the grid
    rows_needed = (len(cells) + grid_cols - 1) // grid_cols
    blank = np.zeros((*cell_size[::-1], 3), dtype=np.uint8)
    while len(cells) < rows_needed * grid_cols:
        cells.append(blank)

    rows = [np.hstack(cells[i:i + grid_cols])
            for i in range(0, len(cells), grid_cols)]
    return np.vstack(rows)


def save_landmark_plot(
    frame: np.ndarray,
    landmarks: np.ndarray,
    output_path: str,
    draw_indices: bool = True,
) -> None:
    """Save a visualised landmark image to disk."""
    vis = draw_landmarks(frame, landmarks, draw_indices=draw_indices)
    cv2.imwrite(output_path, vis)
    print(f"[Visualiser] Saved → {output_path}")
