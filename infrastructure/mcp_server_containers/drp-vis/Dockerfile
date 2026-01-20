FROM python:3.11-slim

# Install system dependencies for RDKit
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app:/app/src

# Set matplotlib backend to non-interactive
ENV MPLBACKEND=Agg

# Run the MCP server using FastMCP stdio transport
CMD ["python", "-u", "src/server.py"]
