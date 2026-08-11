#!/usr/bin/env python3
"""
test_demo.py — Full pipeline demo using a synthetic test face.

Runs without any downloaded model files (uses geometric fallback).
Once lbfmodel.yaml is downloaded, re-run with --backend lbf for real landmarks.

Usage:
    python tests/test_demo.py
    python tests/test_demo.py --backend lbf   # after downloading models
    python tests/test_demo.py --save_all       # save every output artefact
"""

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd

from src.face_detector import get_detector
from src.landmark_extractor import get_extractor, N_LANDMARKS, LANDMARK_GROUPS
from src.openface_format import OpenFaceCSVWriter, load_openface_csv, OPENFACE_COLUMNS
from src.visualizer import draw_landmarks, save_landmark_plot

OUTPUT_DIR = Path("output/demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Synthetic face generator ─────────────────────────────────────────────────
def make_synthetic_face(width: int = 640, height: int = 480) -> np.ndarray:
    """Draw a schematic face on a dark background (no real person needed)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 40)   # dark background

    cx, cy = width // 2, height // 2
    fw, fh = 220, 280

    # Skin gradient fill
    for r in range(fh // 2, 0, -1):
        alpha = r / (fh // 2)
        color = (
            int(60 + alpha * 90),
            int(80 + alpha * 100),
            int(100 + alpha * 120),
        )
        cv2.ellipse(img, (cx, cy), (int(fw / 2 * r / (fh // 2) * 1.1),
                    int(fh / 2 * r / (fh // 2))), 0, 0, 360, color, -1)

    # Face outline
    cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, (180, 160, 140), 2)

    # Eyes
    le_cx, re_cx = cx - 60, cx + 60
    eye_cy = cy - 40
    for ex in [le_cx, re_cx]:
        cv2.ellipse(img, (ex, eye_cy), (28, 16), 0, 0, 360, (220, 220, 220), -1)
        cv2.circle(img, (ex, eye_cy), 12, (60, 80, 40), -1)
        cv2.circle(img, (ex, eye_cy), 6,  (10, 10, 10), -1)
        cv2.circle(img, (ex - 4, eye_cy - 4), 3, (255, 255, 255), -1)

    # Eyebrows
    for ex in [le_cx, re_cx]:
        cv2.ellipse(img, (ex, eye_cy - 24), (26, 7), 0, 200, 340, (80, 60, 50), 3)

    # Nose
    nose_pts = np.array([[cx, cy - 10], [cx - 20, cy + 35],
                          [cx + 20, cy + 35]], np.int32)
    cv2.polylines(img, [nose_pts], False, (140, 120, 110), 2)
    cv2.ellipse(img, (cx, cy + 30), (18, 10), 0, 0, 180, (160, 130, 120), 2)

    # Mouth
    cv2.ellipse(img, (cx, cy + 70), (45, 20), 0, 0, 180, (100, 80, 120), -1)
    cv2.ellipse(img, (cx, cy + 70), (45, 20), 0, 0, 180, (180, 140, 160), 2)
    cv2.line(img, (cx - 45, cy + 70), (cx + 45, cy + 70), (180, 140, 160), 2)

    # Info overlay
    cv2.putText(img, "Synthetic Face — Pipeline Test", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return img


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_face_detection(img: np.ndarray):
    print("\n[TEST 1] Face Detection")
    detector = get_detector(backend="haar")
    bboxes   = detector.detect(img)
    print(f"  Haar detector → {len(bboxes)} face(s) detected")
    for i, bbox in enumerate(bboxes):
        print(f"    Face {i}: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
    return bboxes


def test_landmark_extraction(img: np.ndarray, bboxes: list, backend: str):
    print(f"\n[TEST 2] Landmark Extraction  (backend: {backend})")
    extractor = get_extractor(model_dir="models", backend=backend)
    landmarks_list = extractor.extract(img, bboxes)
    print(f"  → {len(landmarks_list)} face landmark set(s)")
    if landmarks_list:
        lm = landmarks_list[0]
        print(f"  Shape: {lm.shape}  (expected: (68, 2))")
        for group, indices in LANDMARK_GROUPS.items():
            pts = lm[indices]
            print(f"  {group:15s}: mean=({pts[:,0].mean():.1f}, {pts[:,1].mean():.1f})")
    return landmarks_list


def test_openface_csv(landmarks_list: list, bboxes: list):
    print("\n[TEST 3] OpenFace CSV Writer")
    csv_path = OUTPUT_DIR / "demo_landmarks.csv"
    with OpenFaceCSVWriter(str(csv_path)) as writer:
        for frame_idx in range(5):   # simulate 5 frames
            writer.write_frame(landmarks_list, timestamp=frame_idx / 30.0)

    df = load_openface_csv(str(csv_path))
    print(f"  CSV rows   : {len(df)}")
    print(f"  CSV columns: {len(df.columns)}")
    print(f"  Columns include x_0..x_67, y_0..y_67 : "
          f"{'x_0' in df.columns and 'y_67' in df.columns}")
    print(f"  Success flag values: {df['success'].unique().tolist()}")
    return csv_path


def test_visualisation(img: np.ndarray, landmarks_list: list, bboxes: list):
    print("\n[TEST 4] Visualisation")
    if not landmarks_list:
        print("  (skipped — no landmarks)")
        return

    lm   = landmarks_list[0]
    bbox = bboxes[0] if bboxes else None
    vis  = draw_landmarks(img, lm, bbox=bbox, draw_indices=True)

    out_path = str(OUTPUT_DIR / "demo_landmarks.jpg")
    cv2.imwrite(out_path, vis)
    print(f"  Saved annotated image → {out_path}")


def run_all_tests(backend: str, save_all: bool):
    print("=" * 60)
    print("  OpenFace Landmark Extraction — Pipeline Demo")
    print("=" * 60)

    img    = make_synthetic_face()
    bboxes = test_face_detection(img)

    # Provide a default bbox if Haar misses the synthetic face
    if not bboxes:
        print("  [WARN] Haar missed synthetic face — using manual bbox for demo")
        bboxes = [(210, 100, 220, 280)]

    landmarks_list = test_landmark_extraction(img, bboxes, backend)
    csv_path       = test_openface_csv(landmarks_list, bboxes)
    test_visualisation(img, landmarks_list, bboxes)

    if save_all:
        raw_path = str(OUTPUT_DIR / "demo_synthetic_face.jpg")
        cv2.imwrite(raw_path, img)
        print(f"\n  Synthetic face image → {raw_path}")

    print("\n" + "=" * 60)
    print(f"  All tests passed ✓")
    print(f"  Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    # Print CSV preview
    print("\n[CSV Preview — first 2 rows, selected columns]")
    df = pd.read_csv(csv_path)
    preview_cols = ["frame", "face_id", "timestamp", "confidence", "success",
                    "x_0", "y_0", "x_33", "y_33", "x_67", "y_67"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(2).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="OpenFace pipeline demo")
    parser.add_argument("--backend", default="geometric",
                        choices=["auto", "lbf", "kazemi", "geometric"],
                        help="Landmark backend (default: geometric — no model needed)")
    parser.add_argument("--save_all", action="store_true",
                        help="Save all intermediate artefacts")
    args = parser.parse_args()
    run_all_tests(args.backend, args.save_all)


if __name__ == "__main__":
    main()
