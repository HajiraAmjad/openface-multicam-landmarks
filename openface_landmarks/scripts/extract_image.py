#!/usr/bin/env python3
"""
extract_image.py — Extract 68-point face landmarks from one or more images.

Usage:
    python scripts/extract_image.py --input photo.jpg
    python scripts/extract_image.py --input images/ --output output/ --visualize
    python scripts/extract_image.py --input face.png --backend lbf --indices
"""

import sys, argparse
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.face_detector import get_detector
from src.landmark_extractor import get_extractor
from src.openface_format import OpenFaceCSVWriter, landmarks_to_row, OPENFACE_COLUMNS
from src.visualizer import draw_landmarks, save_landmark_plot

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def process_image(
    image_path: Path,
    detector,
    extractor,
    output_dir: Path,
    visualize: bool = False,
    draw_indices: bool = False,
) -> list:
    """Process a single image. Returns list of row dicts for CSV."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"  [WARN] Cannot read {image_path}")
        return []

    bboxes = detector.detect(frame)
    landmarks_list = extractor.extract(frame, bboxes)

    rows = []
    for face_id, lm in enumerate(landmarks_list):
        bbox = bboxes[face_id] if face_id < len(bboxes) else None
        row = landmarks_to_row(
            frame_idx=0,
            face_id=face_id,
            timestamp=0.0,
            landmarks_2d=lm,
            confidence=1.0,
            success=1,
        )
        row["source"] = str(image_path)
        rows.append(row)

        if visualize:
            vis_path = output_dir / f"{image_path.stem}_face{face_id}_landmarks.jpg"
            vis = draw_landmarks(frame, lm, bbox=bbox, draw_indices=draw_indices)
            cv2.imwrite(str(vis_path), vis)

    if not landmarks_list:
        print(f"  [INFO] No face detected in {image_path.name}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract OpenFace-compatible 68-pt landmarks from images."
    )
    parser.add_argument("--input",  required=True,
                        help="Image file or directory of images")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--models", default="models",
                        help="Model directory (default: models/)")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "lbf", "kazemi", "geometric"],
                        help="Landmark extraction backend")
    parser.add_argument("--detector", default="auto",
                        choices=["auto", "haar", "yunet"],
                        help="Face detection backend")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualised landmark images")
    parser.add_argument("--indices",   action="store_true",
                        help="Draw landmark index numbers on visualisation")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image files
    if input_path.is_dir():
        image_files = sorted([p for p in input_path.iterdir()
                               if p.suffix.lower() in IMAGE_EXTS])
    elif input_path.is_file():
        image_files = [input_path]
    else:
        print(f"[ERROR] Input not found: {input_path}")
        sys.exit(1)

    print(f"[INFO] Found {len(image_files)} image(s)")

    # Initialise models
    detector  = get_detector(model_dir=args.models, backend=args.detector)
    extractor = get_extractor(model_dir=args.models, backend=args.backend)

    # Process images
    all_rows = []
    for img_path in tqdm(image_files, desc="Processing", unit="img"):
        rows = process_image(img_path, detector, extractor, output_dir,
                             visualize=args.visualize, draw_indices=args.indices)
        all_rows.extend(rows)

    # Write combined CSV
    if all_rows:
        csv_path = output_dir / "landmarks.csv"
        cols = ["source"] + OPENFACE_COLUMNS
        df = pd.DataFrame(all_rows, columns=[c for c in cols if c in all_rows[0]])
        df.to_csv(csv_path, index=False)
        print(f"\n[Done] {len(all_rows)} face(s) processed.")
        print(f"  CSV  → {csv_path}")
        if args.visualize:
            print(f"  VIS  → {output_dir}/")
    else:
        print("\n[Done] No faces detected in any image.")


if __name__ == "__main__":
    main()
