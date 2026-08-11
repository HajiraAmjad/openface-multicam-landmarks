#!/usr/bin/env python3
"""
openface_wrapper.py — Calls OpenFace C++ executables and returns parsed results.
Designed to run inside the Docker container.

Usage (inside container):
    python3 /opt/openface_wrapper.py --input /data/video.mp4 --out_dir /data/output
"""

import subprocess, sys, os, argparse
from pathlib import Path
import pandas as pd


OPENFACE_BINS = {
    "video":      "FaceLandmarkVid",       # single face, video
    "video_multi":"FaceLandmarkVidMulti",  # multi-face, video
    "image":      "FaceLandmarkImg",       # image
    "extract":    "FeatureExtraction",     # full feature extraction
}


def run_openface(
    input_path: str,
    out_dir: str,
    mode: str = "extract",
    extra_args: list = None,
) -> subprocess.CompletedProcess:
    """
    Wrapper around OpenFace's FeatureExtraction / FaceLandmarkVid.

    Parameters
    ----------
    input_path : str
        Path to video or image file, or camera index as string ("0").
    out_dir : str
        Directory to write OpenFace CSV and visualisation output.
    mode : str
        One of 'video', 'video_multi', 'image', 'extract'.
    extra_args : list
        Additional CLI flags passed directly to OpenFace.

    Returns
    -------
    subprocess.CompletedProcess
    """
    bin_name = OPENFACE_BINS.get(mode, "FeatureExtraction")
    cmd = [bin_name]

    if input_path.isdigit():
        cmd += ["-device", input_path]
    elif input_path.endswith((".jpg", ".jpeg", ".png", ".bmp")):
        cmd += ["-f", input_path]
    else:
        cmd += ["-f", input_path]

    cmd += ["-out_dir", out_dir]

    # Standard landmark + AU output flags
    cmd += [
        "-2Dfp",   # 2-D facial landmarks
        "-3Dfp",   # 3-D facial landmarks
        "-pdmparams",   # PDM (face shape) parameters
        "-pose",        # head pose
        "-aus",         # action units
        "-gaze",        # gaze direction
    ]

    if extra_args:
        cmd += extra_args

    print(f"[OpenFace] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("[ERROR]", result.stderr, file=sys.stderr)
    return result


def load_openface_output(out_dir: str, stem: str) -> pd.DataFrame:
    """Load the CSV produced by OpenFace FeatureExtraction."""
    csv_path = Path(out_dir) / f"{stem}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"OpenFace output not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def get_landmarks_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the 2-D landmark columns (x_0..x_67, y_0..y_67)."""
    x_cols = [f"x_{i}" for i in range(68)]
    y_cols = [f"y_{i}" for i in range(68)]
    base   = ["frame", "face_id", "timestamp", "confidence", "success"]
    cols   = [c for c in base + x_cols + y_cols if c in df.columns]
    return df[cols]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True)
    parser.add_argument("--out_dir", default="/output")
    parser.add_argument("--mode",    default="extract",
                        choices=list(OPENFACE_BINS.keys()))
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    result = run_openface(args.input, args.out_dir, args.mode)
    sys.exit(result.returncode)
