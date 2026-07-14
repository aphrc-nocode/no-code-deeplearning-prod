# no-code-deeplearning-prod/Dockerfile

# NVIDIA CUDA base. Ubuntu 24.04 ships Python 3.12, which the pinned stack in
# requirements.txt requires (numpy 2.4 / pandas 3.0 need Python >= 3.11).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 + the OpenCV/rendering shared libs pulled in by opencv/albumentations.
RUN apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Use an isolated virtualenv. On Ubuntu 24.04 the system Python is
# externally-managed (PEP 668), so installing into it needs a venv (or
# --break-system-packages); a venv is cleaner and keeps the image reproducible.
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install the pinned dependencies. On this CUDA base image, PyPI serves the
# CUDA build of the pinned torch version.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire DL service codebase
COPY . .
