#!/usr/bin/env python3
"""
download_models.py — Download all required model files.

Run this once before using the landmark extraction scripts:
    python scripts/download_models.py

Files downloaded (into models/):
  lbfmodel.yaml                       ~54 MB   OpenCV FacemarkLBF (68-pt landmarks)
  face_detection_yunet_2023mar.onnx   ~0.2 MB  YuNet face detector (optional, faster)
  face_landmark_model.dat             ~98 MB   Kazemi fallback (optional)
"""

import sys, urllib.request, urllib.error
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "lbfmodel.yaml": {
        "url": "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml",
        "required": True,
        "desc": "OpenCV FacemarkLBF — 68-point iBUG landmarks (primary backend)",
    },
    "face_detection_yunet_2023mar.onnx": {
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "required": False,
        "desc": "YuNet DNN face detector (optional, faster than Haar cascade)",
    },
}


def download(url: str, dest: Path) -> bool:
    """Download url to dest with a simple progress indicator."""
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    try:
        def reporthook(block, block_size, total):
            if total > 0:
                pct = block * block_size * 100 // total
                print(f"\r  Downloading {dest.name} ... {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=reporthook)
        size_mb = dest.stat().st_size / 1_048_576
        print(f"\r  ✓ {dest.name}  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\r  ✗ Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    print("=== OpenFace Landmark Model Downloader ===\n")
    success_count = 0
    fail_required = False

    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        tag  = "[required]" if info["required"] else "[optional]"
        print(f"{tag} {filename}")
        print(f"  {info['desc']}")

        if dest.exists() and dest.stat().st_size > 1_000:
            print(f"  ✓ Already present ({dest.stat().st_size // 1024} KB)")
            success_count += 1
            continue

        ok = download(info["url"], dest)
        if ok:
            success_count += 1
        elif info["required"]:
            fail_required = True
        print()

    print(f"\n=== {success_count}/{len(MODELS)} model(s) ready ===")
    if fail_required:
        print("\n[ERROR] Required model(s) failed to download.")
        print("  Manual download:")
        print("  wget 'https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml'"
              " -O models/lbfmodel.yaml")
        sys.exit(1)
    else:
        print("\n[OK] Ready to extract landmarks.")
        print("  python scripts/extract_image.py --input <image.jpg> --visualize")


if __name__ == "__main__":
    main()
