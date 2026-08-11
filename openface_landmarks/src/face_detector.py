"""
Face Detection Module
Supports: Haar Cascade (bundled w/ OpenCV) and OpenCV YuNet DNN detector.
Returns bounding boxes in (x, y, w, h) format — same convention used by OpenFace.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class HaarFaceDetector:
    """Haar Cascade detector — zero external dependencies, ships with OpenCV."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (30, 30)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) face bounding boxes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [tuple(f) for f in faces] if len(faces) else []


class YuNetFaceDetector:
    """
    OpenCV YuNet DNN detector — faster and more accurate than Haar.
    Requires: face_detection_yunet_2023mar.onnx
    Download: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.7,
                 nms_threshold: float = 0.3):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found at {model_path}.\n"
                "Download it with:\n"
                "  wget https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx "
                "-O models/face_detection_yunet_2023mar.onnx"
            )
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        results = []
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            results.append((x, y, fw, fh))
        return results


def get_detector(model_dir: str = "models", backend: str = "haar") -> object:
    """
    Factory — returns the best available face detector.
    backend: 'haar' | 'yunet' | 'auto'
    """
    model_dir = Path(model_dir)
    if backend == "auto":
        yunet_path = model_dir / "face_detection_yunet_2023mar.onnx"
        if yunet_path.exists():
            print("[FaceDetector] Using YuNet DNN detector")
            return YuNetFaceDetector(str(yunet_path))
        print("[FaceDetector] YuNet model not found, falling back to Haar cascade")
        return HaarFaceDetector()
    elif backend == "yunet":
        yunet_path = model_dir / "face_detection_yunet_2023mar.onnx"
        return YuNetFaceDetector(str(yunet_path))
    else:
        return HaarFaceDetector()
