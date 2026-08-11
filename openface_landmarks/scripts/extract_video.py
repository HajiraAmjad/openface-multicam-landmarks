#!/usr/bin/env python3
"""
extract_video.py — Extract 68-point landmarks from a video file or webcam.

Writes an OpenFace-compatible CSV (one row per face per frame) and
optionally an annotated output video.

Usage:
    # From webcam (camera index 0)
    python scripts/extract_video.py --input 0

    # From video file
    python scripts/extract_video.py --input recording.mp4 --save_video

    # Specify frame range and backend
    python scripts/extract_video.py --input clip.mp4 \\
        --start_frame 100 --end_frame 500 --backend lbf --save_video
"""

import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

from src.face_detector import get_detector
from src.landmark_extractor import get_extractor
from src.openface_format import OpenFaceCSVWriter
from src.visualizer import draw_multi_face


def main():
    parser = argparse.ArgumentParser(
        description="Extract OpenFace-compatible landmarks from video / webcam."
    )
    parser.add_argument("--input", required=True,
                        help="Video file path or camera index (0, 1, ...)")
    parser.add_argument("--output",      default="output",  help="Output directory")
    parser.add_argument("--models",      default="models",  help="Model directory")
    parser.add_argument("--backend",     default="auto",
                        choices=["auto", "lbf", "kazemi", "geometric"])
    parser.add_argument("--detector",    default="auto",
                        choices=["auto", "haar", "yunet"])
    parser.add_argument("--save_video",  action="store_true",
                        help="Write an annotated output video")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame",   type=int, default=-1,
                        help="-1 = process until end of video")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Process every Nth frame (1 = every frame)")
    parser.add_argument("--max_faces",   type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open capture
    try:
        src = int(args.input)
        source_name = f"webcam{src}"
        is_webcam = True
    except ValueError:
        src = args.input
        source_name = Path(args.input).stem
        is_webcam = False

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.input}")
        sys.exit(1)

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Source : {args.input}")
    print(f"[INFO] FPS    : {fps:.1f} | Resolution: {width}x{height}")
    if total > 0:
        print(f"[INFO] Frames : {total}")

    # Seek to start frame
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    # Initialise models
    detector  = get_detector(model_dir=args.models, backend=args.detector)
    extractor = get_extractor(model_dir=args.models, backend=args.backend)

    # Prepare outputs
    csv_path   = output_dir / f"{source_name}_landmarks.csv"
    video_writer = None
    if args.save_video:
        vid_path = output_dir / f"{source_name}_annotated.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (width, height))
        print(f"[INFO] Saving annotated video → {vid_path}")

    frame_idx   = args.start_frame
    proc_frames = 0
    start_t     = time.time()

    pbar_total = (args.end_frame - args.start_frame) if args.end_frame > 0 else None

    with OpenFaceCSVWriter(str(csv_path)) as csv_writer:
        with tqdm(total=pbar_total, desc="Extracting", unit="frame",
                  disable=is_webcam) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if args.end_frame > 0 and frame_idx >= args.end_frame:
                    break

                timestamp = frame_idx / fps

                # Skip frames
                if (frame_idx - args.start_frame) % args.skip_frames == 0:
                    bboxes          = detector.detect(frame)[:args.max_faces]
                    landmarks_list  = extractor.extract(frame, bboxes)

                    csv_writer.write_frame(landmarks_list, timestamp)
                    proc_frames += 1

                    if video_writer:
                        vis = draw_multi_face(frame, landmarks_list, bboxes)
                        # Overlay FPS
                        elapsed = time.time() - start_t
                        live_fps = proc_frames / elapsed if elapsed > 0 else 0
                        cv2.putText(vis, f"FPS:{live_fps:.1f}  Frame:{frame_idx}",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 100), 2)
                        video_writer.write(vis)

                    # Webcam: show live preview (press Q to quit)
                    if is_webcam:
                        vis = draw_multi_face(frame, landmarks_list, bboxes)
                        cv2.imshow("OpenFace Landmarks  [Q to quit]", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                frame_idx += 1
                pbar.update(1)

    cap.release()
    if video_writer:
        video_writer.release()
    if is_webcam:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_t
    print(f"\n[Done] {proc_frames} frames processed in {elapsed:.1f}s "
          f"({proc_frames/elapsed:.1f} fps)")
    print(f"  CSV → {csv_path}")


if __name__ == "__main__":
    main()
