# Local Voice Reception AI - Production Dockerfile
# Supports: CUDA GPU / CPU fallback
# Note: Apple Silicon MPS is NOT supported in Docker

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage: CPU-only build
# ============================================
FROM base as cpu

COPY requirements.txt requirements-docker.txt ./

# Install CPU-only PyTorch first
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install -r requirements-docker.txt

COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 7860

CMD ["python", "src/app.py"]

# ============================================
# Stage: CUDA GPU build
# ============================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as cuda-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

FROM cuda-base as cuda

COPY requirements.txt requirements-docker.txt ./

# Install CUDA PyTorch
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip install -r requirements-docker.txt

COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 7860

CMD ["python", "src/app.py"]
