# Face Landmark Extraction — Instructions
**OpenFace-compatible 68-point facial landmark extraction from .mp4 videos**

---

## Requirements
- Python 3.8 or higher
- Linux / macOS / Windows

---

## Setup (run once)

```bash
# 1. Go into the project folder
cd openface_landmarks

# 2. Create virtual environment
python3 -m venv openface_env --system-site-packages

# 3. Activate it
source openface_env/bin/activate        # Linux / macOS
# openface_env\Scripts\activate         # Windows

# 4. Install dependencies
pip3 install -r requirements.txt

# 5. Download the landmark model (~54 MB, run once)
python3 scripts/download_models.py
```

---

## How to Extract Landmarks from Videos

Put your `.mp4` files in a folder (e.g. `test_videos/`) then run:

```bash
python3 scripts/extract_folder.py --input test_videos/ --save_video
```

That's it. The script will automatically find all `.mp4` files in the folder and process each one.

---

## Output

For each video, a separate folder is created inside `output/`:

```
output/
├── video1/
│   ├── video1_landmarks.csv       ← landmark data (OpenFace format)
│   └── video1_annotated.mp4       ← video with landmarks drawn on face
├── video2/
│   ├── video2_landmarks.csv
│   └── video2_annotated.mp4
└── extraction_summary.csv         ← summary of all videos processed
```

### CSV Format
Each CSV file follows the **official OpenFace output format** with **345 columns**:

| Columns | Description |
|---|---|
| `frame`, `timestamp`, `confidence`, `success` | Frame metadata |
| `x_0` … `x_67`, `y_0` … `y_67` | 68 facial landmark coordinates (pixels) |
| `X_0` … `Z_67` | 3D landmark coordinates |

### 68-Point Landmark Map (iBUG scheme — same as OpenFace)
```
 0–16   Jaw line
17–21   Left eyebrow
22–26   Right eyebrow
27–30   Nose bridge
31–35   Nose tip
36–41   Left eye
42–47   Right eye
48–59   Outer lip
60–67   Inner lip
```

---

## All Available Scripts

### 1. Extract from a folder of videos (main script)
```bash
python3 scripts/extract_folder.py --input test_videos/
python3 scripts/extract_folder.py --input test_videos/ --save_video
python3 scripts/extract_folder.py --input test_videos/ --skip_frames 2
python3 scripts/extract_folder.py --input test_videos/ --recursive
```

| Option | Description |
|---|---|
| `--input` | Folder containing .mp4 files |
| `--output` | Where to save results (default: `output/`) |
| `--save_video` | Also write annotated video with landmarks drawn |
| `--skip_frames 2` | Process every 2nd frame (faster) |
| `--recursive` | Scan sub-folders too |
| `--backend lbf` | Use LBF model (most accurate, needs download_models.py first) |

---

### 2. Extract from a single video
```bash
python3 scripts/extract_video.py --input myvideo.mp4 --save_video
```

---

### 3. Extract from webcam (live)
```bash
python3 scripts/extract_video.py --input 0
```
Press **Q** to stop.

---

### 4. Extract from images
```bash
python3 scripts/extract_image.py --input photo.jpg --visualize
python3 scripts/extract_image.py --input images_folder/ --visualize
```

---

## Every Time You Use It

```bash
cd openface_landmarks
source openface_env/bin/activate
python3 scripts/extract_folder.py --input test_videos/ --save_video
```

---

## View the Annotated Video

```bash
xdg-open output/my_face/my_face_annotated.mp4    # Linux
open output/my_face/my_face_annotated.mp4         # macOS
vlc output/my_face/my_face_annotated.mp4          # if VLC installed
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `command 'python' not found` | Use `python3` instead of `python` |
| `No module named cv2` | Run `pip3 install -r requirements.txt` again |
| No faces detected | Try better lighting or a clearer frontal face video |
| Model not found warning | Run `python3 scripts/download_models.py` |

---

## Reference
- OpenFace framework: [github.com/TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
- Landmark scheme: iBUG 300-W 68-point annotation (Sagonas et al., ICCVW 2013)
