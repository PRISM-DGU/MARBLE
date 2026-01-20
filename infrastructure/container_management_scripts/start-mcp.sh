#!/bin/bash

# Get current user information dynamically
CURRENT_USER=$(whoami)
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

# Get script directory and calculate PROJECT_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in infrastructure/container_management_scripts/, so go up 2 levels
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Verify PROJECT_ROOT by checking for .env file
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "⚠️ .env not found at $PROJECT_ROOT, trying to load from environment"
fi

# Set USER_ID from current user (same as DynamicMCPConfig behavior)
USER_ID=$CURRENT_USER

# Set MCP container names (same pattern as DynamicMCPConfig)
SEQUENTIAL_NAME="mcp-sequential_${USER_ID}"
DESKTOP_NAME="mcp-desktop-commander_${USER_ID}"
CONTEXT7_NAME="mcp-context7_${USER_ID}"
SERENA_NAME="mcp-serena_${USER_ID}"
ARXIV_SEMANTIC_NAME="mcp-arxiv-semantic_${USER_ID}"
DRP_VIS_NAME="drp-vis-mcp_${USER_ID}"

# All MCP containers list (6 servers)
ALL_CONTAINERS="$SEQUENTIAL_NAME $DESKTOP_NAME $CONTEXT7_NAME $SERENA_NAME $ARXIV_SEMANTIC_NAME $DRP_VIS_NAME"

# Function to check if required images exist
check_required_images() {
    local desktop_tag="desktop-commander:MARBLE-${USER_ID}"
    local serena_tag="serena:MARBLE-${USER_ID}"
    local arxiv_semantic_tag="arxiv-semantic:MARBLE-${USER_ID}"
    local sequential_tag="mcp/sequentialthinking:MARBLE-${USER_ID}"
    local context7_tag="mcp/context7:MARBLE-${USER_ID}"
    local drp_vis_tag="drp-vis-mcp:MARBLE-${USER_ID}"
    local missing_images=()

    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${desktop_tag}$"; then
        missing_images+=("$desktop_tag")
    fi

    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${serena_tag}$"; then
        missing_images+=("$serena_tag")
    fi

    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${arxiv_semantic_tag}$"; then
        missing_images+=("$arxiv_semantic_tag")
    fi

    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${sequential_tag}$"; then
        missing_images+=("$sequential_tag")
    fi

    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${context7_tag}$"; then
        missing_images+=("$context7_tag")
    fi

    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${drp_vis_tag}$"; then
        missing_images+=("$drp_vis_tag")
    fi

    if [ ${#missing_images[@]} -gt 0 ]; then
        echo "❌ Missing required Docker images:"
        for img in "${missing_images[@]}"; do
            echo "   - $img"
        done
        echo ""
        echo "📦 Please build/pull the required images:"
        echo "   - For custom images: ./infrastructure/container_management_scripts/build-mcp-images.sh"
        exit 1
    else
        echo "✅ All required images found"
    fi
}

# Function to build model execution images from docker_images/
build_model_images() {
    local DOCKER_IMAGES_DIR="$PROJECT_ROOT/docker_images"
    local models=("deeptta" "stagate" "deepst" "dlm-dti" "hyperattentiondti" "deepdr")

    echo ""
    echo "========================================="
    echo "🧬 Building Model Execution Images"
    echo "========================================="

    for model in "${models[@]}"; do
        local model_dir="$DOCKER_IMAGES_DIR/$model"
        local image_tag="marble/${model}:MARBLE-${USER_ID}"

        if [ ! -d "$model_dir" ]; then
            echo "⚠️  Model directory not found: $model_dir"
            continue
        fi

        if [ ! -f "$model_dir/Dockerfile" ]; then
            echo "⚠️  Dockerfile not found for $model"
            continue
        fi

        # Check if image already exists
        if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${image_tag}$"; then
            echo "✅ Image already exists: $image_tag (skipping)"
            continue
        fi

        echo ""
        echo "🔨 Building $model image..."
        echo "   📄 Dockerfile: $model_dir/Dockerfile"
        echo "   🏷️  Tag: $image_tag"
        echo "   👤 UID: $CURRENT_UID, GID: $CURRENT_GID"

        if docker build \
            --build-arg USER_UID=$CURRENT_UID \
            --build-arg USER_GID=$CURRENT_GID \
            -t "$image_tag" \
            -f "$model_dir/Dockerfile" \
            "$model_dir"; then
            echo "✅ Successfully built: $image_tag"
        else
            echo "❌ Failed to build $model image"
        fi
    done

    echo ""
    echo "📋 Model Images Summary:"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" | grep -E "^REPOSITORY|marble/"
}

# Define image tags (all user-specific)
DESKTOP_IMAGE_TAG="desktop-commander:MARBLE-${USER_ID}"
SERENA_IMAGE_TAG="serena:MARBLE-${USER_ID}"
DRP_VIS_IMAGE_TAG="drp-vis-mcp:MARBLE-${USER_ID}"
ARXIV_SEMANTIC_IMAGE_TAG="arxiv-semantic:MARBLE-${USER_ID}"
SEQUENTIAL_IMAGE_TAG="mcp/sequentialthinking:MARBLE-${USER_ID}"
CONTEXT7_IMAGE_TAG="mcp/context7:MARBLE-${USER_ID}"

# Model execution image tags (user-specific)
DEEPDR_IMAGE_TAG="marble/deepdr:MARBLE-${USER_ID}"
DEEPTTA_IMAGE_TAG="marble/deeptta:MARBLE-${USER_ID}"
STAGATE_IMAGE_TAG="marble/stagate:MARBLE-${USER_ID}"
DEEPST_IMAGE_TAG="marble/deepst:MARBLE-${USER_ID}"
DLM_DTI_IMAGE_TAG="marble/dlm-dti:MARBLE-${USER_ID}"
HYPERATTENTIONDTI_IMAGE_TAG="marble/hyperattentiondti:MARBLE-${USER_ID}"

# MCP 환경 시작 스크립트
echo "🚀 Starting MCP environment..."
echo "📁 Using PROJECT_ROOT: $PROJECT_ROOT"
echo "👤 Current User: $CURRENT_USER (UID: $CURRENT_UID, GID: $CURRENT_GID)"
echo "🏷️ Container naming with USER_ID: $USER_ID"
echo "📋 Container Names (6 MCP servers):"
echo "   Sequential: $SEQUENTIAL_NAME"
echo "   Desktop: $DESKTOP_NAME"
echo "   Context7: $CONTEXT7_NAME"
echo "   Serena: $SERENA_NAME"
echo "   ArXiv-Semantic: $ARXIV_SEMANTIC_NAME"
echo "   DRP-VIS: $DRP_VIS_NAME"
echo "📦 Image Tags:"
echo "   Sequential: $SEQUENTIAL_IMAGE_TAG"
echo "   Desktop: $DESKTOP_IMAGE_TAG"
echo "   Context7: $CONTEXT7_IMAGE_TAG"
echo "   Serena: $SERENA_IMAGE_TAG"
echo "   ArXiv-Semantic: $ARXIV_SEMANTIC_IMAGE_TAG"
echo "   DRP-VIS: $DRP_VIS_IMAGE_TAG"
echo "🧬 Model Execution Images:"
echo "   DeepTTA: $DEEPTTA_IMAGE_TAG"
echo "   DeepDR: $DEEPDR_IMAGE_TAG"
echo "   STAGATE: $STAGATE_IMAGE_TAG"
echo "   DeepST: $DEEPST_IMAGE_TAG"
echo "   DLM-DTI: $DLM_DTI_IMAGE_TAG"
echo "   HyperAttentionDTI: $HYPERATTENTIONDTI_IMAGE_TAG"

# Check for required images
check_required_images

# Build model execution images (6 supported models)
build_model_images

# Create required directories for ArXiv-Semantic MCP
echo "📁 Creating required directories..."
mkdir -p "$PROJECT_ROOT/experiments/arxiv/papers"
mkdir -p "$PROJECT_ROOT/experiments/arxiv/embeddings"
mkdir -p "$PROJECT_ROOT/experiments/arxiv/chromadb"
mkdir -p "$PROJECT_ROOT/experiments/arxiv/.cache"
echo "✅ Directory structure created:
   - $PROJECT_ROOT/experiments/arxiv/papers
   - $PROJECT_ROOT/experiments/arxiv/embeddings
   - $PROJECT_ROOT/experiments/arxiv/chromadb
   - $PROJECT_ROOT/experiments/arxiv/.cache"

echo "🧹 Cleaning up existing containers for user: $USER_ID..."
docker stop $ALL_CONTAINERS 2>/dev/null || true
docker rm $ALL_CONTAINERS 2>/dev/null || true
docker stop ${DRP_VIS_NAME} 2>/dev/null || true
docker rm ${DRP_VIS_NAME} 2>/dev/null || true

echo "🧠 Starting sequential thinking server..."
docker run -d --name ${SEQUENTIAL_NAME} --restart unless-stopped \
  -i -t ${SEQUENTIAL_IMAGE_TAG}

echo "🖥️ Starting desktop commander server..."
docker run -d --name ${DESKTOP_NAME} --restart unless-stopped \
  --user $CURRENT_UID:$CURRENT_GID \
  -e HOME="/home/$CURRENT_USER" \
  -e USER=$CURRENT_USER \
  -v $PROJECT_ROOT:/workspace \
  -i -t ${DESKTOP_IMAGE_TAG}

echo "📚 Starting context7 server..."
docker run -d --name ${CONTEXT7_NAME} --restart unless-stopped \
  -e MCP_TRANSPORT=stdio -i -t ${CONTEXT7_IMAGE_TAG}

echo "🔍 Starting Serena server..."
# Try GPU first, fallback to CPU if nvidia-docker not available
if docker run --rm --gpus all ubuntu:20.04 echo "GPU test" &>/dev/null 2>&1; then
    echo "   ✅ GPU available - using GPU acceleration"
    docker run -d --name ${SERENA_NAME} --restart unless-stopped \
      --user $CURRENT_UID:$CURRENT_GID \
      -e HOME="/home/$CURRENT_USER" \
      -e USER=$CURRENT_USER \
      -v $PROJECT_ROOT:/workspace \
      -e SERENA_DOCKER=1 \
      --gpus all \
      -i -t \
      ${SERENA_IMAGE_TAG} .venv/bin/serena-mcp-server --transport stdio --project /workspace
else
    echo "   ⚠️  GPU not available - running in CPU mode"
    echo "   💡 To enable GPU: install nvidia-docker2 and restart Docker daemon"
    docker run -d --name ${SERENA_NAME} --restart unless-stopped \
      --user $CURRENT_UID:$CURRENT_GID \
      -e HOME="/home/$CURRENT_USER" \
      -e USER=$CURRENT_USER \
      -v $PROJECT_ROOT:/workspace \
      -e SERENA_DOCKER=1 \
      -i -t \
      ${SERENA_IMAGE_TAG} .venv/bin/serena-mcp-server --transport stdio --project /workspace
fi

echo "🔬 Starting ArXiv-Semantic MCP server..."
docker run -d --name ${ARXIV_SEMANTIC_NAME} --restart unless-stopped \
  --user $CURRENT_UID:$CURRENT_GID \
  -e HOME="/home/$CURRENT_USER" \
  -e USER=$CURRENT_USER \
  -e PAPERS_DIR="/workspace/experiments/arxiv/papers" \
  -e EMBEDDINGS_DIR="/workspace/experiments/arxiv/embeddings" \
  -e CHROMA_DB_PATH="/workspace/experiments/arxiv/chromadb" \
  -e TRANSFORMERS_CACHE="/workspace/experiments/arxiv/.cache" \
  -v $PROJECT_ROOT:/workspace \
  -i -t \
  ${ARXIV_SEMANTIC_IMAGE_TAG}

# Calculate unique port for DRP-VIS based on UID to avoid conflicts
# Base port 10000 + UID ensures each user has a unique port
DRP_VIS_PORT=$((10000 + CURRENT_UID))

echo "🎨 Starting DRP Visualization MCP server..."
echo "   📍 Using port: $DRP_VIS_PORT (10000 + UID: $CURRENT_UID)"
docker run -d --name ${DRP_VIS_NAME} --restart unless-stopped \
  --user $CURRENT_UID:$CURRENT_GID \
  -e HOME="/home/$CURRENT_USER" \
  -e USER=$CURRENT_USER \
  -v $PROJECT_ROOT:/workspace \
  -e MPLBACKEND=Agg \
  -p ${DRP_VIS_PORT}:8080 \
  -i -t \
  ${DRP_VIS_IMAGE_TAG}

# File ownership is now handled by --user flag in docker run commands
echo "✅ File ownership automatically handled by --user flag"
echo "📁 All new files will be owned by $CURRENT_USER (UID: $CURRENT_UID)"

# 컨테이너 상태 확인
echo "📋 Checking container status..."
sleep 3
docker ps --filter "name=mcp-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
docker ps --filter "name=drp-vis" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo "🔍 Running health checks..."
# Convert space-separated string to array for health checks
read -r -a container_array <<< "$ALL_CONTAINERS"

for container in "${container_array[@]}"; do
    if docker ps --filter "name=${container}" --filter "status=running" | grep -q "${container}"; then
        echo "✅ ${container} server is running"
    else
        echo "❌ ${container} server failed to start"
        docker logs "${container}" --tail 5
    fi
done

echo "⚙️ Configuring Desktop Commander allowed directories..."
# Configure Desktop Commander to allow project directory access
if docker ps --filter "name=${DESKTOP_NAME}" --filter "status=running" | grep -q "${DESKTOP_NAME}"; then
    sleep 2  # Wait for container initialization

    echo "📁 Setting allowedDirectories to: $PROJECT_ROOT"

    # Direct config file creation for stdio MCP servers
    docker exec ${DESKTOP_NAME} sh -c "
        # Create config directory
        mkdir -p ~/.config/desktop-commander

        # Create comprehensive config file
        cat > ~/.config/desktop-commander/config.json << 'EOF'
{
  \"allowedDirectories\": [\"$PROJECT_ROOT\"],
  \"fileReadLineLimit\": 1000,
  \"fileWriteLineLimit\": 50,
  \"telemetryEnabled\": true,
  \"blockedCommands\": [
    \"mkfs\", \"format\", \"mount\", \"umount\", \"fdisk\", \"dd\", \"parted\", \"diskpart\",
    \"sudo\", \"su\", \"passwd\", \"adduser\", \"useradd\", \"usermod\", \"groupadd\", \"chsh\", \"visudo\",
    \"shutdown\", \"reboot\", \"halt\", \"poweroff\", \"init\",
    \"iptables\", \"firewall\", \"netsh\", \"sfc\", \"bcdedit\", \"reg\", \"net\", \"sc\", \"runas\", \"cipher\", \"takeown\"
  ]
}
EOF

        # Verify config file was created successfully
        if [ -f ~/.config/desktop-commander/config.json ]; then
            echo '✅ Desktop Commander config file created successfully'
            echo 'Config contents:'
            cat ~/.config/desktop-commander/config.json | head -5
        else
            echo '❌ Failed to create config file'
            exit 1
        fi
    " || echo "❌ Desktop Commander configuration failed"
else
    echo "❌ Desktop Commander container not running, skipping configuration"
fi

echo "🎉 MCP environment startup complete!"
echo "📝 All MCP servers running on stdio transport"
echo "👤 Files created will be owned by: $CURRENT_USER (UID: $CURRENT_UID)"
echo "📝 Use 'docker logs [container-name]' to check individual logs"
echo "🛑 Use './infrastructure/container_management_scripts/stop-mcp.sh' to stop all MCP services"
