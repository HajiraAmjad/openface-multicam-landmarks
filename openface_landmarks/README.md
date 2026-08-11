# OpenFace Face Landmark Extraction Toolkit

A Python virtual environment and extraction pipeline for 68-point facial landmark detection, producing output in **OpenFace-compatible CSV format** — the same column layout emitted by the original [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) C++ `FeatureExtraction` executable.

---

## Repository Structure

```
openface_landmarks/
├── src/
│   ├── __init__.py              # Public API exports
│   ├── face_detector.py         # Haar cascade + YuNet DNN face detection
│   ├── landmark_extractor.py    # OpenCV LBF / Kazemi / geometric backends
│   ├── openface_format.py       # OpenFace-compatible CSV writer & reader
│   └── visualizer.py            # Group-coloured landmark drawing utilities
│
├── scripts/
│   ├── extract_image.py         # Batch image extraction CLI
│   ├── extract_video.py         # Video / webcam extraction CLI
│   └── download_models.py       # Model file downloader
│
├── docker/
│   ├── Dockerfile               # Builds the actual OpenFace C++ framework
│   └── openface_wrapper.py      # Python wrapper for C++ executables
│
├── tests/
│   └── test_demo.py             # Full pipeline demo (no model needed)
│
├── models/                      # Downloaded model files go here
├── output/                      # Extraction results written here
└── requirements.txt
```

---

## Quick Start

### 1 — Set up the Virtual Environment

```bash
# Create venv (reuse system OpenCV/MediaPipe if present)
python3 -m venv openface_env --system-site-packages
source openface_env/bin/activate           # Linux/macOS
# openface_env\Scripts\activate            # Windows PowerShell

pip install -r requirements.txt
```

### 2 — Download Landmark Models

```bash
python scripts/download_models.py
```

This downloads into `models/`:

| File | Size | Purpose |
|------|------|---------|
| `lbfmodel.yaml` | ~54 MB | OpenCV FacemarkLBF — 68-pt iBUG landmarks (primary) |
| `face_detection_yunet_2023mar.onnx` | ~0.2 MB | YuNet DNN face detector (optional) |

If the automatic download fails (network restrictions), download manually:
```bash
wget "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml" \
     -O models/lbfmodel.yaml
```

### 3 — Run the Demo (no model required)

```bash
python tests/test_demo.py --save_all
```

Generates a synthetic face, runs detection + extraction + CSV writing + visualisation.
Output saved to `output/demo/`.

---

## Usage

### Extract from Images

```bash
# Single image — geometric fallback (no model needed)
python scripts/extract_image.py --input photo.jpg --visualize

# Directory of images — LBF backend (requires lbfmodel.yaml)
python scripts/extract_image.py --input images/ --backend lbf --visualize

# Draw landmark index numbers on output
python scripts/extract_image.py --input face.jpg --visualize --indices
```

Output: `output/landmarks.csv` + annotated JPGs (if `--visualize`).

### Extract from Video

```bash
# Process a video file
python scripts/extract_video.py --input recording.mp4 --save_video

# Process specific frame range
python scripts/extract_video.py --input clip.mp4 \
    --start_frame 100 --end_frame 500 --backend lbf --save_video

# Live webcam (camera 0) — press Q to stop
python scripts/extract_video.py --input 0
```

Output: `output/<name>_landmarks.csv` + annotated video (if `--save_video`).

---

## Output CSV Format

Matches the OpenFace `FeatureExtraction` output exactly — downstream AU
analysis pipelines (OpenFace MATLAB/R scripts, py-feat, etc.) work without
any modification.

| Column group | Columns | Description |
|---|---|---|
| Meta | `frame`, `face_id`, `timestamp`, `confidence`, `success` | Per-frame tracking info |
| 2D landmarks | `x_0`…`x_67`, `y_0`…`y_67` | Pixel coordinates |
| 3D landmarks | `X_0`…`X_67`, `Y_0`…`Y_67`, `Z_0`…`Z_67` | Normalised (0 if 3D unavailable) |

**Total: 345 columns** (5 + 68×2 + 68×3).

### Landmark Index Map (iBUG 68-point scheme)

```
Indices  0–16   → Jaw line          (17 points)
         17–21  → Left eyebrow      ( 5 points)
         22–26  → Right eyebrow     ( 5 points)
         27–30  → Nose bridge       ( 4 points)
         31–35  → Nose tip          ( 5 points)
         36–41  → Left eye          ( 6 points)
         42–47  → Right eye         ( 6 points)
         48–59  → Outer lip         (12 points)
         60–67  → Inner lip         ( 8 points)
```

### Reading the CSV in Python

```python
from src.openface_format import load_openface_csv, get_landmark_array

df  = load_openface_csv("output/recording_landmarks.csv")
lm  = get_landmark_array(df, frame=42, face_id=0)  # → np.ndarray (68, 2)

# Access individual groups
left_eye_pts = lm[36:42]    # shape (6, 2)
nose_tip_pts = lm[31:36]    # shape (5, 2)
```

---

## Backend Options

### Face Detection

| Backend | Flag | Notes |
|---------|------|-------|
| Haar Cascade | `--detector haar` | Zero dependencies, ships with OpenCV |
| YuNet DNN | `--detector yunet` | Faster & more accurate; needs `face_detection_yunet_2023mar.onnx` |
| Auto | `--detector auto` | Picks YuNet if model present, else Haar |

### Landmark Extraction

| Backend | Flag | Model file | Accuracy |
|---------|------|-----------|---------|
| LBF | `--backend lbf` | `lbfmodel.yaml` (54 MB) | ★★★★ Real landmarks |
| Kazemi | `--backend kazemi` | `face_landmark_model.dat` (98 MB) | ★★★☆ Real landmarks |
| Geometric | `--backend geometric` | None | ★☆☆☆ Proportional placeholder |
| Auto | `--backend auto` | Uses best available | — |

---

## Option B — Actual OpenFace C++ via Docker

For production research requiring OpenFace's full feature set (AU detection,
gaze, head pose, PDM parameters), use the Docker build which compiles the
original C++ codebase:

```bash
# Build the container (takes ~10 min — compiles OpenFace from source)
cd docker/
docker build -t openface:latest .

# Extract from a video file
docker run --rm \
    -v $(pwd)/data:/data \
    openface:latest \
    FeatureExtraction -f /data/video.mp4 -out_dir /data/output -2Dfp -3Dfp -aus

# Live webcam
docker run --rm --device /dev/video0 \
    -v $(pwd)/output:/output \
    openface:latest \
    FaceLandmarkVid -device 0 -out_dir /output

# Python wrapper inside the container
docker run --rm -v $(pwd)/data:/data \
    openface:latest \
    python3 /opt/openface_wrapper.py --input /data/video.mp4 --out_dir /data/output
```

### OpenFace C++ CLI Flags Reference

| Flag | Output |
|------|--------|
| `-2Dfp` | 2-D facial landmarks (pixel coords) |
| `-3Dfp` | 3-D facial landmarks (normalised) |
| `-pdmparams` | Face shape parameters |
| `-pose` | Head pose (6 DoF) |
| `-aus` | Facial Action Units (AU intensity + presence) |
| `-gaze` | Gaze direction vectors |
| `-tracked` | Write annotated video |

---

## Python API

```python
import cv2
from src import get_detector, get_extractor, OpenFaceCSVWriter, draw_landmarks

# Setup
detector  = get_detector(backend="auto")
extractor = get_extractor(backend="auto")

frame = cv2.imread("face.jpg")

# Detect + extract
bboxes         = detector.detect(frame)
landmarks_list = extractor.extract(frame, bboxes)

# Write OpenFace-compatible CSV
with OpenFaceCSVWriter("output/result.csv") as writer:
    writer.write_frame(landmarks_list, timestamp=0.0)

# Visualise
vis = draw_landmarks(frame, landmarks_list[0], bbox=bboxes[0],
                     draw_indices=True, draw_connections=True)
cv2.imwrite("output/annotated.jpg", vis)
```

---

## Troubleshooting

**`lbfmodel.yaml` downloads as 0 bytes**  
The hosting domain may be rate-limited. Try:
```bash
curl -L "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml" \
     -o models/lbfmodel.yaml
```

**Haar detector misses faces**  
Use `--detector yunet` (more robust) or lower the confidence threshold via the `YuNetFaceDetector` constructor.

**`No module named 'cv2'` in venv**  
Re-create the venv with `--system-site-packages`:
```bash
python3 -m venv openface_env --system-site-packages
```

**OpenFace Docker build fails on `download_models.sh`**  
Models require `wget`/`curl` access from inside the container. If behind a proxy, pass `--build-arg https_proxy=...`.

---

## References

- OpenFace: Baltrusaitis et al., *"OpenFace 2.0: Facial Behavior Analysis Toolkit"*, FG 2018. [GitHub](https://github.com/TadasBaltrusaitis/OpenFace)
- iBUG 68-point landmark scheme: Sagonas et al., *"300 Faces In-The-Wild Challenge"*, ICCVW 2013.
- OpenCV FacemarkLBF: Ren et al., *"Face Alignment at 3000 FPS via Regressing Local Binary Features"*, CVPR 2014.
