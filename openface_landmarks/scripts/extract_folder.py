#!/usr/bin/env python3
"""
extract_folder.py — Scan a folder for all .mp4 files and extract facial
landmarks from each one, writing results to a separate output sub-folder
per video.

Usage:
    python scripts/extract_folder.py --input /path/to/videos/
    python scripts/extract_folder.py --input videos/ --backend lbf --save_video
    python scripts/extract_folder.py --input videos/ --recursive --skip_frames 2

Output structure:
    output/
    ├── video1/
    │   ├── video1_landmarks.csv      ← OpenFace-format CSV (345 cols)
    │   └── video1_annotated.mp4      ← (if --save_video)
    ├── video2/
    │   ├── video2_landmarks.csv
    │   └── video2_annotated.mp4
    └── extraction_summary.csv        ← one row per video (stats)
"""

import sys, argparse, time, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

from src.face_detector import get_detector
from src.landmark_extractor import get_extractor
from src.openface_format import OpenFaceCSVWriter
from src.visualizer import draw_multi_face


# ── Per-video processor ──────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    output_root: Path,
    detector,
    extractor,
    save_video: bool = False,
    skip_frames: int = 1,
    max_faces: int = 10,
) -> dict:
    """
    Extract landmarks from a single .mp4 file.
    Returns a summary dict with stats for the summary CSV.
    """
    stem       = video_path.stem
    out_dir    = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path.name}")
        return {"video": video_path.name, "status": "error",
                "frames": 0, "faces_detected": 0, "duration_s": 0}

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0

    csv_path = out_dir / f"{stem}_landmarks.csv"

    video_writer = None
    if save_video:
        vid_out  = out_dir / f"{stem}_annotated.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(vid_out), fourcc, fps, (width, height))

    frame_idx    = 0
    faces_total  = 0
    start_t      = time.time()

    with OpenFaceCSVWriter(str(csv_path)) as writer:
        with tqdm(total=total, desc=f"  {stem[:40]}", unit="f",
                  leave=False, ncols=80) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps

                if frame_idx % skip_frames == 0:
                    bboxes         = detector.detect(frame)[:max_faces]
                    landmarks_list = extractor.extract(frame, bboxes)
                    faces_total   += len(landmarks_list)

                    writer.write_frame(landmarks_list, timestamp)

                    if video_writer:
                        vis = draw_multi_face(frame, landmarks_list, bboxes)
                        # burn-in frame info
                        cv2.putText(
                            vis,
                            f"Frame:{frame_idx}  Faces:{len(landmarks_list)}  "
                            f"t:{timestamp:.2f}s",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 120), 1,
                        )
                        video_writer.write(vis)

                frame_idx += 1
                pbar.update(1)

    cap.release()
    if video_writer:
        video_writer.release()

    elapsed     = time.time() - start_t
    proc_frames = (frame_idx + skip_frames - 1) // skip_frames

    return {
        "video":          video_path.name,
        "status":         "ok",
        "frames_total":   frame_idx,
        "frames_processed": proc_frames,
        "faces_detected": faces_total,
        "duration_s":     round(duration, 2),
        "process_time_s": round(elapsed, 2),
        "fps_proc":       round(proc_frames / elapsed, 1) if elapsed > 0 else 0,
        "csv":            str(csv_path),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan a folder for .mp4 files and extract OpenFace-compatible "
            "landmarks, one output folder per video."
        )
    )
    parser.add_argument(
        "--input", required=True,
        help="Folder containing .mp4 video files",
    )
    parser.add_argument(
        "--output", default="output",
        help="Root output directory  (default: output/)",
    )
    parser.add_argument(
        "--models", default="models",
        help="Model directory  (default: models/)",
    )
    parser.add_argument(
        "--backend", default="auto",
        choices=["auto", "lbf", "kazemi", "geometric"],
        help="Landmark extraction backend  (default: auto → lbf if model present)",
    )
    parser.add_argument(
        "--detector", default="auto",
        choices=["auto", "haar", "yunet"],
        help="Face detection backend  (default: auto)",
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="Write annotated output video alongside each CSV",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Scan sub-folders recursively for .mp4 files",
    )
    parser.add_argument(
        "--skip_frames", type=int, default=1,
        help="Process every Nth frame  (default: 1 = every frame)",
    )
    parser.add_argument(
        "--max_faces", type=int, default=10,
        help="Maximum faces to track per frame  (default: 10)",
    )
    args = parser.parse_args()

    # ── Collect .mp4 files ───────────────────────────────────────────────────
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[ERROR] Not a directory: {input_dir}")
        sys.exit(1)

    pattern   = "**/*.mp4" if args.recursive else "*.mp4"
    mp4_files = sorted(input_dir.glob(pattern))

    if not mp4_files:
        print(f"[WARN] No .mp4 files found in {input_dir}")
        sys.exit(0)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Initialise models (once, shared across all videos) ───────────────────
    print("\n=== OpenFace Batch Landmark Extraction ===")
    print(f"  Input  : {input_dir.resolve()}")
    print(f"  Output : {output_root.resolve()}")
    print(f"  Videos : {len(mp4_files)}")
    print(f"  Backend: {args.backend}  |  Detector: {args.detector}")
    print(f"  Skip   : every {args.skip_frames} frame(s)")
    print()

    detector  = get_detector(model_dir=args.models, backend=args.detector)
    extractor = get_extractor(model_dir=args.models, backend=args.backend)

    # ── Process each video ───────────────────────────────────────────────────
    summaries = []
    total_start = time.time()

    for i, vid_path in enumerate(mp4_files, 1):
        print(f"[{i}/{len(mp4_files)}] {vid_path.name}")
        summary = process_video(
            video_path   = vid_path,
            output_root  = output_root,
            detector     = detector,
            extractor    = extractor,
            save_video   = args.save_video,
            skip_frames  = args.skip_frames,
            max_faces    = args.max_faces,
        )
        summaries.append(summary)

        if summary["status"] == "ok":
            print(
                f"  ✓  {summary['frames_processed']} frames | "
                f"{summary['faces_detected']} faces | "
                f"{summary['duration_s']:.1f}s video | "
                f"{summary['process_time_s']:.1f}s process time\n"
                f"     → {summary['csv']}"
            )
        print()

    # ── Write summary CSV ────────────────────────────────────────────────────
    summary_path = output_root / "extraction_summary.csv"
    if summaries:
        keys = summaries[0].keys()
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)

    total_elapsed = time.time() - total_start
    ok_count = sum(1 for s in summaries if s["status"] == "ok")

    print("=" * 55)
    print(f"  Done: {ok_count}/{len(mp4_files)} videos processed "
          f"in {total_elapsed:.1f}s")
    print(f"  Summary → {summary_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
