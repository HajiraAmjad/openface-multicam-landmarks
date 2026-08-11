"""
Face Landmark Extractor
Supports three backends (in order of preference):

  1. OpenCV FacemarkLBF  — 68-point iBUG landmarks, same set used by OpenFace.
     Model: lbfmodel.yaml (≈54 MB)
     Download: https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml

  2. OpenCV FacemarkKazemi — 68-point alternative, faster but less accurate.
     Model: face_landmark_model.dat (≈98 MB)
     Download: https://github.com/opencv/opencv_3rdparty/raw/contrib_face_alignment_20170818/face_landmark_model.dat

  3. Geometric fallback — derives approximate landmark positions from the face
     bounding box using proportional offsets (no model required). Useful for
     pipeline testing when model files are not yet available.

OpenFace landmark index convention (iBUG 68-point scheme):
  0–16   Jaw line
  17–21  Left eyebrow
  22–26  Right eyebrow
  27–30  Nose bridge
  31–35  Nose tip
  36–41  Left eye
  42–47  Right eye
  48–59  Outer lip
  60–67  Inner lip
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# Landmark group indices — mirrors OpenFace conventions
LANDMARK_GROUPS = {
    "jaw":           list(range(0, 17)),
    "left_eyebrow":  list(range(17, 22)),
    "right_eyebrow": list(range(22, 27)),
    "nose_bridge":   list(range(27, 31)),
    "nose_tip":      list(range(31, 36)),
    "left_eye":      list(range(36, 42)),
    "right_eye":     list(range(42, 48)),
    "outer_lip":     list(range(48, 60)),
    "inner_lip":     list(range(60, 68)),
}

N_LANDMARKS = 68


# ---------------------------------------------------------------------------
# Geometric fallback (no model required)
# ---------------------------------------------------------------------------
_JAW_XN  = [0.0, 0.08, 0.15, 0.22, 0.30, 0.37, 0.44, 0.50,
             0.56, 0.63, 0.70, 0.78, 0.85, 0.92, 0.95, 0.97, 1.00]
_JAW_YN  = [0.55, 0.65, 0.74, 0.82, 0.88, 0.93, 0.97, 1.00,
             0.97, 0.93, 0.88, 0.82, 0.74, 0.65, 0.60, 0.57, 0.55]

_PROTO_68 = np.array([
    # Jaw (17)
    *zip(_JAW_XN, _JAW_YN),
    # Left eyebrow (5): 17-21
    (0.20, 0.30), (0.28, 0.24), (0.37, 0.22), (0.45, 0.24), (0.50, 0.28),
    # Right eyebrow (5): 22-26
    (0.55, 0.28), (0.60, 0.24), (0.68, 0.22), (0.77, 0.24), (0.85, 0.30),
    # Nose bridge (4): 27-30
    (0.50, 0.32), (0.50, 0.40), (0.50, 0.48), (0.50, 0.56),
    # Nose tip (5): 31-35
    (0.40, 0.60), (0.44, 0.63), (0.50, 0.65), (0.56, 0.63), (0.60, 0.60),
    # Left eye (6): 36-41
    (0.28, 0.35), (0.34, 0.32), (0.41, 0.32), (0.47, 0.35),
    (0.41, 0.38), (0.34, 0.38),
    # Right eye (6): 42-47
    (0.53, 0.35), (0.59, 0.32), (0.66, 0.32), (0.72, 0.35),
    (0.66, 0.38), (0.59, 0.38),
    # Outer lip (12): 48-59
    (0.38, 0.72), (0.43, 0.70), (0.50, 0.69), (0.57, 0.70), (0.62, 0.72),
    (0.58, 0.77), (0.50, 0.80), (0.42, 0.77),
    (0.38, 0.72), (0.43, 0.73), (0.50, 0.74), (0.57, 0.73),
    # Inner lip (8): 60-67
    (0.43, 0.72), (0.50, 0.71), (0.57, 0.72),
    (0.57, 0.76), (0.50, 0.78), (0.43, 0.76),
    (0.43, 0.72), (0.57, 0.72),
], dtype=np.float32)

assert len(_PROTO_68) == N_LANDMARKS, f"Prototype has {len(_PROTO_68)} points"


def _geometric_landmarks(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Return Nx2 float32 landmark coordinates from a face bounding box."""
    x, y, w, h = bbox
    pts = _PROTO_68.copy()
    pts[:, 0] = x + pts[:, 0] * w
    pts[:, 1] = y + pts[:, 1] * h
    return pts


# ---------------------------------------------------------------------------
# OpenCV Facemark backends
# ---------------------------------------------------------------------------
class LBFLandmarkExtractor:
    """
    OpenCV FacemarkLBF — 68-point iBUG landmarks.
    Identical landmark set to OpenFace's CLM/CE-CLM tracker.
    """

    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"LBF model not found at {model_path}.\n"
                "Run  python scripts/download_models.py  or:\n"
                "  wget https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
                " -O models/lbfmodel.yaml"
            )
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(str(model_path))
        print(f"[LBF] Model loaded from {model_path}")

    def extract(self, frame: np.ndarray,
                bboxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Return list of (68, 2) float32 arrays, one per detected face.
        Returns empty list if no faces or detection fails.
        """
        if not bboxes:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        rect_list = [list(b) for b in bboxes]
        ok, landmarks = self.facemark.fit(gray, np.array(rect_list))
        if not ok:
            return []
        # landmarks shape: (n_faces, 1, 68, 2)
        return [lm[0].reshape(N_LANDMARKS, 2) for lm in landmarks]


class KazemiLandmarkExtractor:
    """
    OpenCV FacemarkKazemi — 68-point alternative backend.
    Model: face_landmark_model.dat
    """

    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Kazemi model not found at {model_path}.\n"
                "Download from OpenCV contrib releases."
            )
        self.facemark = cv2.face.createFacemarkKazemi()
        self.facemark.loadModel(str(model_path))
        print(f"[Kazemi] Model loaded from {model_path}")

    def extract(self, frame: np.ndarray,
                bboxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        if not bboxes:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        rect_list = [list(b) for b in bboxes]
        ok, landmarks = self.facemark.fit(gray, np.array(rect_list))
        if not ok:
            return []
        return [lm[0].reshape(N_LANDMARKS, 2) for lm in landmarks]


class GeometricLandmarkExtractor:
    """
    Fallback extractor — proportional geometric landmarks.
    No model file required.  Good for pipeline testing and CI.
    NOT suitable for real research measurements.
    """

    def extract(self, frame: np.ndarray,
                bboxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        return [_geometric_landmarks(bbox) for bbox in bboxes]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_extractor(model_dir: str = "models", backend: str = "auto"):
    """
    Returns the best available landmark extractor.
    backend: 'auto' | 'lbf' | 'kazemi' | 'geometric'
    """
    model_dir = Path(model_dir)

    if backend in ("auto", "lbf"):
        lbf_path = model_dir / "lbfmodel.yaml"
        if lbf_path.exists():
            return LBFLandmarkExtractor(str(lbf_path))
        if backend == "lbf":
            raise FileNotFoundError(f"lbfmodel.yaml not found in {model_dir}")

    if backend in ("auto", "kazemi"):
        kazemi_path = model_dir / "face_landmark_model.dat"
        if kazemi_path.exists():
            return KazemiLandmarkExtractor(str(kazemi_path))
        if backend == "kazemi":
            raise FileNotFoundError(f"face_landmark_model.dat not found in {model_dir}")

    print("[LandmarkExtractor] No model found — using geometric fallback.")
    print("  Run:  python scripts/download_models.py  to get real landmarks.")
    return GeometricLandmarkExtractor()
