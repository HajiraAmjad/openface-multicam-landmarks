# OpenFace Landmark Extraction + Multi-Camera Calibration

A pipeline for extracting facial landmarks from video using OpenFace, with calibration support for multi-camera setups. Developed at the Intelligent Space Robotics Laboratory (ISR Lab), Skoltech.

---

## Repository Structure

```
openface-multicam-landmarks/
├── openface_landmarks/
│   ├── scripts/
│   │   ├── extract_folder.py       # Main script — batch process a folder of videos
│   │   ├── extract_video.py        # Process a single video
│   │   ├── extract_image.py        # Process a single image
│   │   └── download_models.py      # Download required OpenFace models
│   ├── src/
│   │   ├── landmark_extractor.py   # Core landmark extraction logic
│   │   ├── face_detector.py        # Face detection module
│   │   ├── openface_format.py      # Output formatting (OpenFace-compatible)
│   │   ├── visualizer.py           # Visualization utilities
│   │   └── __init__.py
│   ├── docker/
│   │   ├── Dockerfile              # Docker environment with OpenFace
│   │   └── openface_wrapper.py     # Wrapper for running OpenFace in Docker
│   ├── tests/
│   │   └── test_demo.py
│   ├── requirements.txt
│   ├── INSTRUCTIONS.md             # Detailed setup and usage guide
│   └── README.md
└── calibration_results.json        # 3-camera calibration output
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r openface_landmarks/requirements.txt
```

### 2. Download OpenFace models

```bash
python3 openface_landmarks/scripts/download_models.py
```

### 3. Run on a folder of videos

```bash
python3 openface_landmarks/scripts/extract_folder.py \
    --input your_videos_folder/ \
    --save_video
```

For full setup instructions, Docker usage, and output format details, see [`INSTRUCTIONS.md`](openface_landmarks/INSTRUCTIONS.md).

---

## Camera Calibration

Calibrated for a 3-camera setup (Front, Left, Right) using a checkerboard pattern.

| Parameter | Value |
|---|---|
| Reprojection error | 0.086 px ✅ (threshold < 1.0 px) |
| Multi-camera error | 1.12 px |
| Frames used | 22 |

**Camera distances:**
- Front ↔ Left: 27.1 cm
- Front ↔ Right: 45.1 cm
- Left ↔ Right: 64.0 cm

**Camera angles (relative to Front):**
- Front–Left: 18.1°
- Front–Right: 34.2°
- Left–Right: 52.3°

Full calibration results in [`calibration_results.json`](calibration_results.json).

---

## Docker (Optional)

If OpenFace is not installed locally, use the provided Docker setup:

```bash
cd openface_landmarks/docker
docker build -t openface-landmarks .
docker run -v $(pwd):/data openface-landmarks python3 openface_wrapper.py
```

---

## Requirements

- Python 3.8+
- OpenFace (local install or Docker)
- See `requirements.txt` for Python dependencies

---

## Affiliation

Developed as part of multi-sensor data collection research at **ISR Lab, Skoltech**.
