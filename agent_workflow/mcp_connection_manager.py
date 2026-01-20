"""MCP server management."""

import asyncio
import atexit
import json
import os
import sys
from typing import Any, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient
import docker

from agent_workflow.logger import logger
from .utils import DynamicMCPConfig

DEBUG_LOGS = os.getenv("AUTODRP_DEBUG_LOGS", "false").lower() == "true"
COMPILE_MODE = "compile" in " ".join(sys.argv)

_mcp_config = DynamicMCPConfig()
container_names = _mcp_config.ALL_MCP_CONTAINERS


def debug_print(*args, **kwargs):
    """Print only if debug logging is enabled."""
    if DEBUG_LOGS and not COMPILE_MODE:
        print(*args, **kwargs)


def load_mcp_config(config_path: str = "configs/mcp.json") -> Dict[str, Any]:
    """Load MCP configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"[MCP] Failed to load config from {config_path}: {e}")


class MCPManager:
    """MCP server manager with Docker container support."""

    def __init__(self):
        self.clients = {}
        self.tools = {}
        self._initialized = False
        self._servers_initialized = {}
        self.docker_client = None
        self.config = load_mcp_config()

        self.mcp_enabled = os.getenv("ENABLE_MCP", "false").lower() == "true"

        if self.mcp_enabled:
            logger.mcp(f"Configuration loaded: {list(self.config.get('servers', {}).keys())}")
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.error(f"Failed to connect to Docker: {e}")
                raise
        else:
            logger.mcp("MCP disabled - running without Docker containers")

        atexit.register(self.stop_all_servers)

    async def initialize_all_servers(self):
        """Initialize all MCP servers."""
        if self._initialized:
            return self.tools

        if not self.mcp_enabled:
            logger.mcp("MCP disabled - skipping initialization")
            self._initialized = True
            return self.tools

        try:
            await self._wait_for_containers()

            successful_connections = 0
            for container_name in container_names:
                if not container_name:
                    continue

                if container_name in self._servers_initialized:
                    debug_print(f"[MCP] Skipping {container_name} - already initialized")
                    if container_name in self.tools and self.tools[container_name]:
                        successful_connections += 1
                    continue

                tools = await self._connect_container_server(container_name)
                self.tools[container_name] = tools
                self._servers_initialized[container_name] = True

                if tools:
                    successful_connections += 1
                    debug_print(f"[MCP] {container_name}: {len(tools)} tools")
                else:
                    debug_print(f"[MCP] {container_name}: no tools")

            self._initialized = True
            total_tools = sum(len(tools) for tools in self.tools.values())
            logger.mcp(f"MCP Ready: {total_tools} tools")

            return self.tools

        except Exception as e:
            logger.error(f"MCP Initialization failed: {e}")
            self._initialized = False
            raise

    async def _wait_for_containers(self):
        """Wait for MCP containers to be ready."""
        if not self.docker_client or not container_names:
            return

        try:
            running_containers = []
            for name in container_names:
                if self._is_container_running(name):
                    running_containers.append(name)

            if running_containers:
                debug_print(f"[MCP] Found running containers: {running_containers}")
            else:
                raise RuntimeError(f"[MCP] No MCP containers running. Expected: {container_names}")

        except Exception as e:
            raise RuntimeError(f"[MCP] Error checking containers: {e}")

    async def _connect_container_server(self, container_name: str):
        """Connect to a containerized MCP server."""
        try:
            if not self._is_container_running(container_name):
                debug_print(f"[MCP] Container {container_name} is not running")
                return []

            base_name = container_name
            user_id = os.getenv('USER_ID', '').strip()

            if user_id and container_name.endswith(f'_{user_id}'):
                base_name = container_name[:-len(f'_{user_id}')]
            elif '_' in container_name:
                base_name = container_name.rsplit('_', 1)[0]

            debug_print(f"[MCP] Container: {container_name}, Base: {base_name}, User: {user_id}")

            server_config = self.config.get("servers", {}).get(base_name, {})
            if not server_config:
                debug_print(f"[MCP] No config for {base_name}")
                debug_print(f"[MCP] Available: {list(self.config.get('servers', {}).keys())}")
                return []

            command = server_config.get("command", "node")
            args = server_config.get("args", ["dist/index.js"])
            docker_args = ["exec", "-i", container_name, command] + args

            env = os.environ.copy()
            env["MCP_QUIET"] = "true"
            env["LOG_LEVEL"] = "ERROR"

            client_config = {
                container_name: {
                    "command": "docker",
                    "args": docker_args,
                    "transport": server_config.get("transport", "stdio"),
                    "env": env
                }
            }

            client = MultiServerMCPClient(client_config)
            self.clients[container_name] = client

            try:
                timeout = self.config.get("settings", {}).get("connection_timeout", 15)
                tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)

                if tools:
                    blocked_tools = server_config.get("blocked_tools", [])
                    if blocked_tools:
                        original_count = len(tools)
                        tools = [tool for tool in tools if tool.name not in blocked_tools]
                        blocked_count = original_count - len(tools)
                        if blocked_count > 0:
                            debug_print(f"[MCP] Blocked {blocked_count} tools from {container_name}")

                tool_names = [tool.name for tool in tools] if tools else []
                debug_print(f"[MCP] Tools: {', '.join(tool_names[:5])}{', ... (+{} more)'.format(len(tool_names)-5) if len(tool_names) > 5 else ''}")
                return tools
            except asyncio.TimeoutError:
                raise RuntimeError(f"[MCP] Connection to {container_name} timed out")
            except Exception as conn_error:
                raise RuntimeError(f"[MCP] Connection error for {container_name}: {conn_error}")

        except Exception as e:
            raise RuntimeError(f"[MCP] Failed to connect to {container_name}: {e}")

    def _is_container_running(self, container_name: str) -> bool:
        """Check if a Docker container is running."""
        if not self.docker_client:
            raise RuntimeError(f"[MCP] Docker client not available")

        try:
            container = self.docker_client.containers.get(container_name)
            return container.status == "running"
        except docker.errors.NotFound:
            raise RuntimeError(f"[MCP] Container {container_name} not found")
        except Exception as e:
            raise RuntimeError(f"[MCP] Error checking container {container_name}: {e}")

    def get_tools_from_server(self, server_name: str):
        """Get tools from a specific MCP server."""
        debug_print(f"[MCP] get_tools_from_server: {server_name}")
        debug_print(f"[MCP] Available: {list(self.tools.keys())}")

        if not self._initialized:
            debug_print(f"[MCP] Not initialized, returning empty for {server_name}")
            return []

        server_tools = self.tools.get(server_name, [])
        if server_tools:
            debug_print(f"[MCP] Retrieved {len(server_tools)} tools from {server_name}")
            for i, tool in enumerate(server_tools[:2]):
                debug_print(f"[MCP]   Tool #{i+1}: {getattr(tool, 'name', 'unknown')}")
        else:
            debug_print(f"[MCP] No tools for {server_name}")

        return server_tools

    def stop_all_servers(self):
        """Stop all server connections."""
        self.clients.clear()
        self.tools.clear()
        self._initialized = False
        debug_print("[MCP] Cleared all connections")
