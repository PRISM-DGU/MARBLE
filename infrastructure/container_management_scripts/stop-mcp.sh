#!/bin/bash

# Get current user information dynamically
CURRENT_USER=$(whoami)
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

# Get script directory and calculate PROJECT_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in infrastructure/container_management_scripts/, so go up 2 levels
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Set USER_ID from current user (same as DynamicMCPConfig behavior)
USER_ID=$CURRENT_USER

# Set MCP container names (same pattern as DynamicMCPConfig)
SEQUENTIAL_NAME="mcp-sequential_${USER_ID}"
DESKTOP_NAME="mcp-desktop-commander_${USER_ID}"
CONTEXT7_NAME="mcp-context7_${USER_ID}"
SERENA_NAME="mcp-serena_${USER_ID}"
DRP_VIS_NAME="drp-vis-mcp_${USER_ID}"

# All MCP containers list
ALL_CONTAINERS="$SEQUENTIAL_NAME $DESKTOP_NAME $CONTEXT7_NAME $SERENA_NAME $DRP_VIS_NAME"

# MCP 환경 정지 스크립트
echo "🛑 Stopping MCP environment for user: $USER_ID"
echo "👤 Current User: $CURRENT_USER (UID: $CURRENT_UID, GID: $CURRENT_GID)"
echo "🏷️ Container naming with USER_ID: $USER_ID"
echo "📋 Containers to stop: $ALL_CONTAINERS"

# MCP 컨테이너들 정지
echo "⏹️  Stopping MCP containers for $USER_ID..."
docker stop $ALL_CONTAINERS 2>/dev/null || echo "ℹ️ Some containers were not running"

# 컨테이너 제거
echo "🗑️  Removing MCP containers for $USER_ID..."
docker rm $ALL_CONTAINERS 2>/dev/null || echo "ℹ️ Some containers were already removed"

# 최종 상태 확인
echo "📋 Final status check..."
remaining_mcp=$(docker ps -a --filter "name=mcp-.*_$USER_ID" --format "{{.Names}}" 2>/dev/null)
remaining_drp=$(docker ps -a --filter "name=drp-vis-mcp_$USER_ID" --format "{{.Names}}" 2>/dev/null)

if [ -n "$remaining_mcp" ] || [ -n "$remaining_drp" ]; then
    echo "⚠️  Some containers for $USER_ID still exist:"
    docker ps -a --filter "name=mcp-.*_$USER_ID" --format "table {{.Names}}\t{{.Status}}"
    docker ps -a --filter "name=drp-vis-mcp_$USER_ID" --format "table {{.Names}}\t{{.Status}}"
else
    echo "✅ All MCP containers for $USER_ID have been removed"
fi

echo "🎉 MCP environment cleanup complete for user: $USER_ID"
