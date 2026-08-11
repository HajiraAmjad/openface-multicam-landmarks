# ============================================================
# Dockerfile — Actual OpenFace C++ Framework
# ============================================================
# Builds the original OpenFace (TadasBaltrusaitis/OpenFace) inside
# a container and exposes its CLI tools + a Python wrapper.
#
# Usage:
#   docker build -t openface:latest .
#   docker run --rm -v $(pwd)/data:/data openface:latest \
#       FeatureExtraction -f /data/face.mp4 -out_dir /data/output
#
# Or with webcam:
#   docker run --rm --device /dev/video0 -v $(pwd)/output:/output \
#       openface:latest FaceLandmarkVid -device 0 -out_dir /output
# ============================================================

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libopencv-dev libboost-all-dev \
    libdlib-dev libtbb-dev \
    python3 python3-pip \
    wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Clone OpenFace
WORKDIR /opt
RUN git clone --depth=1 https://github.com/TadasBaltrusaitis/OpenFace.git

# Download models required by OpenFace
WORKDIR /opt/OpenFace
RUN bash ./download_models.sh

# Build
WORKDIR /opt/OpenFace/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_CXX_FLAGS="-std=c++14" \
          .. \
    && make -j$(nproc)

# Install Python wrapper dependencies
RUN pip3 install opencv-python-headless numpy pandas

# Copy Python wrapper
COPY docker/openface_wrapper.py /opt/openface_wrapper.py

# Add executables to PATH
ENV PATH="/opt/OpenFace/build/bin:${PATH}"

# Default: show available tools
CMD ["bash", "-c", "echo 'OpenFace tools available:' && ls /opt/OpenFace/build/bin/"]
