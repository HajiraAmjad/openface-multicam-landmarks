"""
OpenFace-Compatible CSV Output
Writes landmark data in the exact column format produced by OpenFace's
FeatureExtraction executable so downstream tools (MATLAB/R/Python AU analysis
pipelines) work without modification.

OpenFace output columns:
  frame, face_id, timestamp, confidence, success,
  x_0..x_67, y_0..y_67,              (2D landmarks, pixel coords)
  X_0..X_67, Y_0..Y_67, Z_0..Z_67    (3D landmarks, normalised — set to 0
                                       when 3D estimation is unavailable)
"""

import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any


# Build the canonical OpenFace column header (matches FeatureExtraction output)
_BASE_COLS = ["frame", "face_id", "timestamp", "confidence", "success"]
_2D_X = [f" x_{i}" for i in range(68)]
_2D_Y = [f" y_{i}" for i in range(68)]
_3D_X = [f" X_{i}" for i in range(68)]
_3D_Y = [f" Y_{i}" for i in range(68)]
_3D_Z = [f" Z_{i}" for i in range(68)]

OPENFACE_COLUMNS = _BASE_COLS + _2D_X + _2D_Y + _3D_X + _3D_Y + _3D_Z


def landmarks_to_row(
    frame_idx: int,
    face_id: int,
    timestamp: float,
    landmarks_2d: np.ndarray,          # shape (68, 2)
    confidence: float = 1.0,
    success: int = 1,
    landmarks_3d: Optional[np.ndarray] = None,  # shape (68, 3), optional
) -> Dict[str, Any]:
    """
    Convert a (68, 2) landmark array to an OpenFace-format dict row.
    """
    row: Dict[str, Any] = {
        "frame":      frame_idx,
        "face_id":    face_id,
        "timestamp":  round(timestamp, 4),
        "confidence": round(confidence, 4),
        "success":    success,
    }

    pts = landmarks_2d.reshape(68, 2)
    for i in range(68):
        row[f" x_{i}"] = round(float(pts[i, 0]), 2)
        row[f" y_{i}"] = round(float(pts[i, 1]), 2)

    if landmarks_3d is not None:
        pts3 = landmarks_3d.reshape(68, 3)
        for i in range(68):
            row[f" X_{i}"] = round(float(pts3[i, 0]), 4)
            row[f" Y_{i}"] = round(float(pts3[i, 1]), 4)
            row[f" Z_{i}"] = round(float(pts3[i, 2]), 4)
    else:
        for i in range(68):
            row[f" X_{i}"] = 0.0
            row[f" Y_{i}"] = 0.0
            row[f" Z_{i}"] = 0.0

    return row


class OpenFaceCSVWriter:
    """
    Streaming CSV writer — call write_frame() for each processed frame,
    then close().  Mimics the file OpenFace's FeatureExtraction produces.
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=OPENFACE_COLUMNS)
        self._writer.writeheader()
        self._frame_count = 0

    def write_frame(
        self,
        landmarks_list: List[np.ndarray],   # one (68,2) array per face
        timestamp: float,
        confidences: Optional[List[float]] = None,
    ) -> None:
        """Write one CSV row per detected face in this frame."""
        for face_id, lm in enumerate(landmarks_list):
            conf = confidences[face_id] if confidences else 1.0
            row = landmarks_to_row(
                frame_idx=self._frame_count,
                face_id=face_id,
                timestamp=timestamp,
                landmarks_2d=lm,
                confidence=conf,
                success=1,
            )
            self._writer.writerow(row)
        # If no face detected, write a "failure" row
        if not landmarks_list:
            row = {col: 0 for col in OPENFACE_COLUMNS}
            row.update({"frame": self._frame_count, "timestamp": round(timestamp, 4),
                        "success": 0, "confidence": 0.0})
            self._writer.writerow(row)
        self._frame_count += 1

    def close(self) -> None:
        self._file.close()
        print(f"[CSV] Saved {self._frame_count} frames → {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_openface_csv(csv_path: str) -> pd.DataFrame:
    """Load an OpenFace-format CSV and return a DataFrame."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def get_landmark_array(df: pd.DataFrame, frame: int,
                       face_id: int = 0) -> Optional[np.ndarray]:
    """
    Extract a (68, 2) landmark array for a specific frame from a loaded DataFrame.
    Returns None if the frame has success=0.
    """
    mask = (df["frame"] == frame) & (df["face_id"] == face_id)
    rows = df[mask]
    if rows.empty or rows.iloc[0]["success"] == 0:
        return None
    row = rows.iloc[0]
    x_cols = [f"x_{i}" for i in range(68)]
    y_cols = [f"y_{i}" for i in range(68)]
    xs = row[x_cols].values.astype(np.float32)
    ys = row[y_cols].values.astype(np.float32)
    return np.stack([xs, ys], axis=1)
