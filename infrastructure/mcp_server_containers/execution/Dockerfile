# Execution MCP Server Docker Image
# Unified execution server for drug response prediction models

# ============================================================================
# Base Stage: Common dependencies
# ============================================================================
FROM python:3.11-slim AS base
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
# Note: docker.io is required for Docker-in-Docker functionality
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Build arguments for dynamic user
ARG USER_NAME=default_user
ARG USER_UID=1000
ARG USER_GID=1000

# Create group and user dynamically
RUN groupadd -g ${USER_GID} appgroup 2>/dev/null || true && \
    useradd -u ${USER_UID} -g ${USER_GID} -m -s /bin/bash ${USER_NAME} 2>/dev/null || true

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# ============================================================================
# Production Stage: Install dependencies and configure
# ============================================================================
FROM base AS production

# Create virtual environment
RUN uv venv /opt/execution-venv
ENV PATH="/opt/execution-venv/bin:$PATH"

# Copy requirements.txt for all dependencies
COPY requirements.txt /tmp/requirements.txt

# Install all dependencies from requirements.txt (MCP, ML frameworks, scientific computing, etc)
RUN uv pip install --python /opt/execution-venv/bin/python \
    -r /tmp/requirements.txt

# Build arguments needed again in this stage (for multi-stage build)
ARG USER_NAME=default_user
ARG USER_UID=1000
ARG USER_GID=1000

# Create workspace mount points and directories with proper permissions
RUN mkdir -p /workspace && \
    mkdir -p /workspace/models && \
    mkdir -p /workspace/execution_logs && \
    mkdir -p /home/${USER_NAME} && \
    chown -R ${USER_UID}:${USER_GID} /workspace && \
    chown -R ${USER_UID}:${USER_GID} /home/${USER_NAME} 2>/dev/null || true

# Add user to docker group for Docker socket access
# Note: GID 999 is typically used for docker group in Docker-in-Docker setups
RUN groupadd -g 999 docker 2>/dev/null || true && \
    usermod -aG docker ${USER_NAME} 2>/dev/null || true

# Switch to dynamic user
USER ${USER_UID}:${USER_GID}

# Set environment variables for the user
ENV HOME="/home/${USER_NAME}"

# Set Python path to include workspace and app
ENV PYTHONPATH="/workspace:/app:${PYTHONPATH}"

# Copy execution MCP source code
COPY --chown=${USER_UID}:${USER_GID} src/ /app/execution/

# Entrypoint to ensure environment is activated
ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]

# Default command starts the Execution MCP server
CMD ["python", "/app/execution/server.py"]
