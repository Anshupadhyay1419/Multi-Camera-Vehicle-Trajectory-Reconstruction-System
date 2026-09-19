# ALPR University Gate — GPU image for an x86_64 Linux PC/server.
#
# The Jetson (ARM64, JetPack) needs its own image built on an l4t base; this
# one is for the lab PC. Models, config, the database and uploads are NOT
# baked in -- docker-compose.yml mounts them from the host, so rebuilding or
# replacing the container never touches stored data.

# ── build stage: compile the venv (PyCUDA needs the CUDA headers) ────────────
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH CUDA_ROOT=/usr/local/cuda
RUN pip install --upgrade pip setuptools wheel

# Exact package set frozen from the venv the pipeline was verified in.
# --no-deps installs precisely this list: letting the resolver re-solve it
# pulls in a second OpenCV build (easyocr/ultralytics both declare
# opencv-python) that silently replaces the FFMPEG one RTSP depends on.
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-deps -r /tmp/requirements-docker.txt

# ── runtime stage ────────────────────────────────────────────────────────────
# "runtime" (not "base") because PyCUDA links libcurand, which it provides.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 libgl1 \
        ffmpeg curl tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

# The container runs as the host user (see compose), who has no home inside
# the image. Ultralytics, PyCUDA and matplotlib all cache under $HOME.
RUN mkdir -p /home/app && chmod 777 /home/app
ENV PATH=/opt/venv/bin:$PATH \
    HOME=/home/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /app
COPY . /app

# Dashboard, camera-wall MJPEG previews, REST API.
EXPOSE 8502 8765 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "src/dashboard/trajectory_app.py", \
     "--server.port=8502", "--server.address=0.0.0.0", "--server.headless=true"]
