# HyperAttentionDTI Model Execution Docker Image
# Blackwell GPU (sm_120) support with PyTorch nightly + CUDA 12.8
# Build: docker build -t autodrp/hyperattentiondti:latest .
# Run: docker run --gpus all -it autodrp/hyperattentiondti:latest

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS base
SHELL ["/bin/bash", "-c"]

# User ID and Group ID arguments for dynamic UID/GID support
ARG USER_UID=1000
ARG USER_GID=1000
ENV USER_UID=${USER_UID}
ENV USER_GID=${USER_GID}

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Create virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ============================================================================
# Install PyTorch nightly + CUDA 12.8 (Blackwell GPU sm_120 support)
# ============================================================================
RUN uv pip install --python /opt/venv/bin/python \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ============================================================================
# Copy shared components (Registry pattern support)
# ============================================================================
COPY components /app/components

# ============================================================================
# Copy model source code from local src directory
# ============================================================================
COPY src /app/hyperattentiondti_models

# ============================================================================
# Install model-specific requirements
# ============================================================================
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt

# ============================================================================
# Create directories with proper permissions
# ============================================================================
RUN mkdir -p /app/hyperattentiondti /app/hyperattentiondti/checkpoints \
    /workspace/models /workspace/execution_logs && \
    mkdir -p /home/appuser && \
    groupadd -g ${USER_GID} appuser && \
    useradd -m -u ${USER_UID} -g ${USER_GID} appuser && \
    chown -R ${USER_UID}:${USER_GID} /app /workspace /home/appuser

# Set Python path (components included)
ENV PYTHONPATH="/app:/app/hyperattentiondti_models:/app/components"

# Switch to non-root user
USER appuser

# Default command
CMD ["/bin/bash"]
