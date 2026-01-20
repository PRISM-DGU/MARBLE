"""Configuration for MARBLE multi-agent system."""

import os
from dotenv import load_dotenv
load_dotenv()
from agent_workflow.utils import DynamicMCPConfig, get_project_root

# =============================================================================
# 1. LLM MODEL CONFIGURATION
# =============================================================================

MODEL_PROVIDER = "openai"
MODEL_NAME = "gpt-5-mini"

# GPT-4 family parameters
GPT4_PARAMS = {
    "temperature": 0.2,
    "seed": 42,
    "max_tokens": 8192,
    "top_p": 0.1,
    "streaming": True,
}

# GPT-5 family parameters
GPT5_PARAMS = {
    "reasoning_effort": "low",
    "max_completion_tokens": 8192,
    "model_kwargs": {"stream": False},
}

# Claude family parameters
CLAUDE_PARAMS = {
    "temperature": 0.2,
    "max_tokens": 8192,
    "streaming": True,
}

# Auto-select parameters based on model
if "gpt-5" in MODEL_NAME or "gpt-5-mini" in MODEL_NAME or "gpt-5.1-codex-mini" in MODEL_NAME:
    MODEL_PARAMS = GPT5_PARAMS
elif MODEL_PROVIDER == "anthropic" or "claude" in MODEL_NAME.lower():
    MODEL_PARAMS = CLAUDE_PARAMS
else:
    MODEL_PARAMS = GPT4_PARAMS

# =============================================================================
# 2. LANGGRAPH CONFIGURATION
# =============================================================================

# Can be overridden via LANGGRAPH_RECURSION_LIMIT env var
_default_recursion_limit = 5000
LANGGRAPH_CONFIG = {
    "recursion_limit": int(os.getenv("LANGGRAPH_RECURSION_LIMIT", str(_default_recursion_limit))),
    "max_execution_time": 3600,
    "node_timeout": 1800,
}

# =============================================================================
# 3. MCP SERVER CONFIGURATION
# =============================================================================

# Set ENABLE_MCP=true in .env to enable MCP
ENABLE_MCP = os.getenv("ENABLE_MCP", "false").lower() == "true"

if ENABLE_MCP:
    _mcp_config = DynamicMCPConfig()
    SEQUENTIAL_MCP = _mcp_config.SEQUENTIAL_MCP
    DESKTOP_MCP = _mcp_config.DESKTOP_MCP
    CONTEXT7_MCP = _mcp_config.CONTEXT7_MCP
    SERENA_MCP = _mcp_config.SERENA_MCP
    DRP_VIS_MCP = _mcp_config.DRP_VIS_MCP
    ALL_MCP_CONTAINERS = _mcp_config.ALL_MCP_CONTAINERS
else:
    SEQUENTIAL_MCP = None
    DESKTOP_MCP = None
    CONTEXT7_MCP = None
    SERENA_MCP = None
    DRP_VIS_MCP = None
    ALL_MCP_CONTAINERS = []

# =============================================================================
# 4. NODE ACTIVATION CONFIGURATION
# =============================================================================

ACTIVE_NODES = {
    "entry_router": True,
    "analysis_subgraph": True,
    "build_debate_subgraph": True,
    "build_development_subgraph": True,
    "docker_execution_subgraph": True,
    "init_iteration": True,
    "inject_memory_context": True,
    "save_to_memory": True,
    "check_continue": True,
}

# =============================================================================
# 5. MCP TOOL ASSIGNMENTS
# =============================================================================

if ENABLE_MCP:
    NODE_MCP_MAPPING = {
        "entry_router": [],
        "analysis_router": [],
        "visualizer": [DRP_VIS_MCP, SERENA_MCP],
        "agenda_generator": [SERENA_MCP],
        "synthesis_specialist": [SERENA_MCP],
        "representative_researcher": [SERENA_MCP],
        "debate_moderator": [],
        "debate_reporter": [SERENA_MCP],
        "model_researcher": [SERENA_MCP],
    }
else:
    NODE_MCP_MAPPING = {}

# =============================================================================
# 6. GRAPH ROUTING CONFIGURATION
# =============================================================================

MAIN_GRAPH_EDGES = {
    "entry_router": {
        "type": "conditional",
        "function": "route_from_entry_router",
        "routes": {
            "analysis_subgraph": "analysis_subgraph",
            "build_debate_subgraph": "build_debate_subgraph",
            "build_development_subgraph": "build_development_subgraph",
            "init_iteration": "init_iteration",
            "END": "END"
        }
    },
    "analysis_subgraph": "END",
    "build_debate_subgraph": "build_development_subgraph",
    "build_development_subgraph": "docker_execution_subgraph",
    "docker_execution_subgraph": "END",
}

# =============================================================================
# 7. WORKFLOW CONFIGURATION
# =============================================================================

WORKFLOW_CONFIG = {
    "max_iterations": 3,
    "improvement_threshold": 0.05,
}
