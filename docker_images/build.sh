#!/bin/bash
# MARBLE Models Docker Build Script (Development Stage)
#
# Usage:
#   ./build.sh           # Build all models
#   ./build.sh deepdr    # Build specific model
#   ./build.sh deeptta deepdr  # Build multiple models

set -e

# Get current user information dynamically
CURRENT_USER=$(whoami)
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=("deeptta" "stagate" "deepst" "dlm-dti" "hyperattentiondti" "deepdr")

# 인자가 있으면 해당 모델만 빌드
if [ $# -gt 0 ]; then
    MODELS=("$@")
fi

echo "=============================================="
echo "DRP Models Docker Build Script (Development)"
echo "=============================================="
echo ""
echo "👤 User: $CURRENT_USER (UID: $CURRENT_UID, GID: $CURRENT_GID)"
echo ""

for model in "${MODELS[@]}"; do
    MODEL_DIR="$SCRIPT_DIR/$model"

    if [ ! -d "$MODEL_DIR" ]; then
        echo "WARNING: Model directory not found: $MODEL_DIR"
        continue
    fi

    # 이미지 이름: 모델명-develop:MARBLE_아이디
    IMAGE_NAME="${model}-develop"
    IMAGE_TAG="MARBLE_${CURRENT_USER}"
    FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

    echo "----------------------------------------------"
    echo "Building: $model"
    echo "  🏷️  Image: $FULL_IMAGE"
    echo "  👤 UID: $CURRENT_UID, GID: $CURRENT_GID"
    echo "----------------------------------------------"

    # Docker 빌드 (USER_UID, USER_GID 전달)
    echo "  → Building Docker image..."
    cd "$MODEL_DIR"
    docker build \
        --build-arg USER_UID=$CURRENT_UID \
        --build-arg USER_GID=$CURRENT_GID \
        -t "$FULL_IMAGE" .

    echo "  ✓ $model build complete!"
    echo ""
done

echo "=============================================="
echo "All builds complete!"
echo "=============================================="
echo ""
echo "Available images:"
for model in "${MODELS[@]}"; do
    echo "  - ${model}-develop:MARBLE_${CURRENT_USER}"
done
echo ""
echo "Run example:"
echo "  docker run --user $CURRENT_UID:$CURRENT_GID -it ${MODELS[0]}-develop:MARBLE_${CURRENT_USER} bash"
