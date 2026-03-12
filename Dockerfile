# no-code-deeplearning-prod/Dockerfile

# Use NVIDIA's official CUDA base image
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies (with CUDA torch)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire DL service codebase
COPY . .