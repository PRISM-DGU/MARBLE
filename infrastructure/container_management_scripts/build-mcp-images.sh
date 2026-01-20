#!/bin/bash

# ============================================================================
# build-mcp-images.sh - Docker Image Build Script for MCP Containers
# ============================================================================
# Purpose: Build Docker images for MCP containers with user-specific settings
# This script is separated from start-mcp.sh to avoid unnecessary rebuilds
# ============================================================================

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

echo "🔨 MCP Docker Image Builder"
echo "========================================"
echo "📁 Using PROJECT_ROOT: $PROJECT_ROOT"
echo "👤 Current User: $CURRENT_USER (UID: $CURRENT_UID, GID: $CURRENT_GID)"
echo "🏷️ User ID for tagging: $USER_ID"

# Define image tags (all user-specific)
DESKTOP_IMAGE_TAG="desktop-commander:AutoDRP-${USER_ID}"
SERENA_IMAGE_TAG="serena:AutoDRP-${USER_ID}"
DRP_VIS_IMAGE_TAG="drp-vis-mcp:AutoDRP-${USER_ID}"
ARXIV_SEMANTIC_IMAGE_TAG="arxiv-semantic:AutoDRP-${USER_ID}"
SEQUENTIAL_IMAGE_TAG="mcp/sequentialthinking:AutoDRP-${USER_ID}"
CONTEXT7_IMAGE_TAG="mcp/context7:AutoDRP-${USER_ID}"

# MCP server containers directory
MCP_CONTAINERS_DIR="infrastructure/mcp_server_containers"

echo "🏷️ Image Tags (all user-specific):"
echo "   Desktop: $DESKTOP_IMAGE_TAG"
echo "   Serena: $SERENA_IMAGE_TAG"
echo "   DRP-VIS: $DRP_VIS_IMAGE_TAG"
echo "   ArXiv-Semantic: $ARXIV_SEMANTIC_IMAGE_TAG"
echo "   Sequential: $SEQUENTIAL_IMAGE_TAG"
echo "   Context7: $CONTEXT7_IMAGE_TAG"
echo ""

# Function to check if image exists
check_image_exists() {
    local image_tag=$1
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${image_tag}$"; then
        return 0
    else
        return 1
    fi
}

# Function to build Docker image with progress indicator
build_image() {
    local dockerfile=$1
    local context=$2
    local tag=$3
    local name=$4

    echo "📦 Building $name image..."
    echo "   Dockerfile: $dockerfile"
    echo "   Context: $context"
    echo "   Tag: $tag"
    echo ""

    if docker build \
        --build-arg USER_NAME=$CURRENT_USER \
        --build-arg USER_UID=$CURRENT_UID \
        --build-arg USER_GID=$CURRENT_GID \
        -t "$tag" \
        -f "$dockerfile" \
        "$context"; then
        echo "✅ $name image built successfully"
        return 0
    else
        echo "❌ Failed to build $name image"
        return 1
    fi
}

# Function to pull and retag official MCP images
pull_and_retag() {
    local source_tag=$1
    local target_tag=$2
    local name=$3

    echo "📥 Pulling $name image..."
    if docker pull "$source_tag"; then
        echo "🏷️ Retagging to $target_tag..."
        docker tag "$source_tag" "$target_tag"
        echo "✅ $name image ready: $target_tag"
        return 0
    else
        echo "❌ Failed to pull $name image"
        return 1
    fi
}

# Parse command line arguments
FORCE_REBUILD=false
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_REBUILD=true
            shift
            ;;
        --skip-existing|-s)
            SKIP_EXISTING=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -f, --force        Force rebuild even if images exist"
            echo "  -s, --skip-existing Skip building if images already exist"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "If no options are provided, you will be prompted for existing images."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Pull and retag Sequential Thinking image
echo "========================================="
echo "🧠 Sequential Thinking Image"
echo "========================================="

if check_image_exists "$SEQUENTIAL_IMAGE_TAG"; then
    echo "ℹ️ Image already exists: $SEQUENTIAL_IMAGE_TAG"

    if [ "$FORCE_REBUILD" = true ]; then
        echo "🔄 Force rebuild requested"
        pull_and_retag "mcp/sequentialthinking:latest" "$SEQUENTIAL_IMAGE_TAG" "Sequential Thinking"
    elif [ "$SKIP_EXISTING" = true ]; then
        echo "⏭️ Skipping existing image"
    else
        read -p "Re-pull Sequential Thinking image? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pull_and_retag "mcp/sequentialthinking:latest" "$SEQUENTIAL_IMAGE_TAG" "Sequential Thinking"
        else
            echo "⏭️ Keeping existing image"
        fi
    fi
else
    echo "🆕 Image not found, pulling..."
    if ! pull_and_retag "mcp/sequentialthinking:latest" "$SEQUENTIAL_IMAGE_TAG" "Sequential Thinking"; then
        echo "❌ Pull failed, exiting"
        exit 1
    fi
fi

echo ""

# Pull and retag Context7 image
echo "========================================="
echo "📚 Context7 Image"
echo "========================================="

if check_image_exists "$CONTEXT7_IMAGE_TAG"; then
    echo "ℹ️ Image already exists: $CONTEXT7_IMAGE_TAG"

    if [ "$FORCE_REBUILD" = true ]; then
        echo "🔄 Force rebuild requested"
        pull_and_retag "mcp/context7:latest" "$CONTEXT7_IMAGE_TAG" "Context7"
    elif [ "$SKIP_EXISTING" = true ]; then
        echo "⏭️ Skipping existing image"
    else
        read -p "Re-pull Context7 image? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pull_and_retag "mcp/context7:latest" "$CONTEXT7_IMAGE_TAG" "Context7"
        else
            echo "⏭️ Keeping existing image"
        fi
    fi
else
    echo "🆕 Image not found, pulling..."
    if ! pull_and_retag "mcp/context7:latest" "$CONTEXT7_IMAGE_TAG" "Context7"; then
        echo "❌ Pull failed, exiting"
        exit 1
    fi
fi

echo ""

# Build Desktop Commander image
echo "========================================="
echo "🖥️ Desktop Commander Image"
echo "========================================="

if check_image_exists "$DESKTOP_IMAGE_TAG"; then
    echo "ℹ️ Image already exists: $DESKTOP_IMAGE_TAG"

    if [ "$FORCE_REBUILD" = true ]; then
        echo "🔄 Force rebuild requested"
        build_image \
            "$MCP_CONTAINERS_DIR/desktop-commander/Dockerfile.custom" \
            "$MCP_CONTAINERS_DIR/desktop-commander/" \
            "$DESKTOP_IMAGE_TAG" \
            "Desktop Commander"
    elif [ "$SKIP_EXISTING" = true ]; then
        echo "⏭️ Skipping existing image"
    else
        read -p "Rebuild Desktop Commander image? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            build_image \
                "$MCP_CONTAINERS_DIR/desktop-commander/Dockerfile.custom" \
                "$MCP_CONTAINERS_DIR/desktop-commander/" \
                "$DESKTOP_IMAGE_TAG" \
                "Desktop Commander"
        else
            echo "⏭️ Keeping existing image"
        fi
    fi
else
    echo "🆕 Image not found, building..."
    if ! build_image \
        "$MCP_CONTAINERS_DIR/desktop-commander/Dockerfile.custom" \
        "$MCP_CONTAINERS_DIR/desktop-commander/" \
        "$DESKTOP_IMAGE_TAG" \
        "Desktop Commander"; then
        echo "❌ Build failed, exiting"
        exit 1
    fi
fi

echo ""

# Build Serena image
echo "========================================="
echo "🔍 Serena Image"
echo "========================================="

if check_image_exists "$SERENA_IMAGE_TAG"; then
    echo "ℹ️ Image already exists: $SERENA_IMAGE_TAG"

    if [ "$FORCE_REBUILD" = true ]; then
        echo "🔄 Force rebuild requested"
        build_image \
            "$MCP_CONTAINERS_DIR/serena/Dockerfile.custom" \
            "$MCP_CONTAINERS_DIR/serena/" \
            "$SERENA_IMAGE_TAG" \
            "Serena"
    elif [ "$SKIP_EXISTING" = true ]; then
        echo "⏭️ Skipping existing image"
    else
        read -p "Rebuild Serena image? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            build_image \
                "$MCP_CONTAINERS_DIR/serena/Dockerfile.custom" \
                "$MCP_CONTAINERS_DIR/serena/" \
                "$SERENA_IMAGE_TAG" \
                "Serena"
        else
            echo "⏭️ Keeping existing image"
        fi
    fi
else
    echo "🆕 Image not found, building..."
    if ! build_image \
        "$MCP_CONTAINERS_DIR/serena/Dockerfile.custom" \
        "$MCP_CONTAINERS_DIR/serena/" \
        "$SERENA_IMAGE_TAG" \
        "Serena"; then
        echo "❌ Build failed, exiting"
        exit 1
    fi
fi

echo ""

# Build ArXiv-Semantic MCP image
echo "========================================="
echo "🔬 ArXiv-Semantic MCP Image"
echo "========================================="

if check_image_exists "$ARXIV_SEMANTIC_IMAGE_TAG"; then
    echo "ℹ️ Image already exists: $ARXIV_SEMANTIC_IMAGE_TAG"

    if [ "$FORCE_REBUILD" = true ]; then
        echo "🔄 Force rebuild requested"
        build_image \
            "$MCP_CONTAINERS_DIR/arxiv-semantic/Dockerfile.custom" \
            "$MCP_CONTAINERS_DIR/arxiv-semantic/" \
            "$ARXIV_SEMANTIC_IMAGE_TAG" \
            "ArXiv-Semantic MCP"
    elif [ "$SKIP_EXISTING" = true ]; then
        echo "⏭️ Skipping existing image"
    else
        read -p "Rebuild ArXiv-Semantic MCP image? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            build_image \
                "$MCP_CONTAINERS_DIR/arxiv-semantic/Dockerfile.custom" \
                "$MCP_CONTAINERS_DIR/arxiv-semantic/" \
                "$ARXIV_SEMANTIC_IMAGE_TAG" \
                "ArXiv-Semantic MCP"
        else
            echo "⏭️ Keeping existing image"
        fi
    fi
else
    echo "🆕 Image not found, building..."
    if ! build_image \
        "$MCP_CONTAINERS_DIR/arxiv-semantic/Dockerfile.custom" \
        "$MCP_CONTAINERS_DIR/arxiv-semantic/" \
        "$ARXIV_SEMANTIC_IMAGE_TAG" \
        "ArXiv-Semantic MCP"; then
        echo "❌ Failed to build ArXiv-Semantic MCP image"
        exit 1
    fi
fi

echo ""

# Build DRP-VIS MCP image
echo "========================================="
echo "🎨 DRP-VIS MCP Image"
echo "========================================="

if check_image_exists "$DRP_VIS_IMAGE_TAG"; then
    echo "ℹ️ Image already exists: $DRP_VIS_IMAGE_TAG"

    if [ "$FORCE_REBUILD" = true ]; then
        echo "🔄 Force rebuild requested"
        build_image \
            "$MCP_CONTAINERS_DIR/drp-vis/Dockerfile" \
            "$MCP_CONTAINERS_DIR/drp-vis/" \
            "$DRP_VIS_IMAGE_TAG" \
            "DRP-VIS MCP"
    elif [ "$SKIP_EXISTING" = true ]; then
        echo "⏭️ Skipping existing image"
    else
        read -p "Rebuild DRP-VIS MCP image? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            build_image \
                "$MCP_CONTAINERS_DIR/drp-vis/Dockerfile" \
                "$MCP_CONTAINERS_DIR/drp-vis/" \
                "$DRP_VIS_IMAGE_TAG" \
                "DRP-VIS MCP"
        else
            echo "⏭️ Keeping existing image"
        fi
    fi
else
    echo "🆕 Image not found, building..."
    if ! build_image \
        "$MCP_CONTAINERS_DIR/drp-vis/Dockerfile" \
        "$MCP_CONTAINERS_DIR/drp-vis/" \
        "$DRP_VIS_IMAGE_TAG" \
        "DRP-VIS MCP"; then
        echo "❌ Build failed, exiting"
        exit 1
    fi
fi

echo ""
echo "========================================="
echo "📊 Build Summary"
echo "========================================="
echo "✅ MCP image build process complete!"
echo ""
echo "📦 User-Specific Images:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" | grep -E "^REPOSITORY|AutoDRP-${USER_ID}"
echo ""
echo "💡 Next steps:"
echo "   1. Run './infrastructure/container_management_scripts/start-mcp.sh' to start containers"
echo "   2. Use '--force' flag to force rebuild: $0 --force"
echo "   3. Use '--skip-existing' flag for automated builds: $0 --skip-existing"
