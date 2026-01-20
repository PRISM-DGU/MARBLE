"""Simple entry router function with flag-based routing for MARBLE multi-agent system."""

import os
import shutil
from pathlib import Path
from typing import Dict
from langchain_core.messages import AIMessage

from agent_workflow.state import MARBLEState
from agent_workflow.logger import logger


def setup_model_directories(model_name: str) -> bool:
    """Setup model agent directories by copying from original model directory."""
    # Source from models directory, target to experiments directory
    models_base_path = Path.cwd() / "models"
    experiments_base_path = Path.cwd() / "experiments"
    source_dir = models_base_path / model_name
    target_dir = experiments_base_path / f"{model_name}"
    
    try:
        # Ensure experiments directory exists
        experiments_base_path.mkdir(exist_ok=True)
        
        # Check if agent directory already exists
        if target_dir.exists():
            pass  # Directory already exists
            return True
        
        # Check if source directory exists
        if not source_dir.exists():
            pass  # Source directory not found
            return False
        
        # Copy entire directory
        # Copy source directory to target
        shutil.copytree(source_dir, target_dir)
        
        # Create required subdirectories
        logs_dir = target_dir / "logs"
        reports_dir = target_dir / "reports"
        
        logs_dir.mkdir(exist_ok=True)
        reports_dir.mkdir(exist_ok=True)
        
        # Created experiment directory with subdirectories
        return True
        
    except Exception as e:
        pass  # Error setting up directory
        return False


def simple_entry_router(state: MARBLEState) -> Dict:
    """Simple 2-way routing function based on task parameter.

    New simplified routing:
    - task == train → base_model_generator
    - task == develop → debate_subgraph
    """
    # Get user message
    if not state.get("messages"):
        return {
            "next_node": "END",
            "router_decision": "ERROR",
            "router_reasoning": "No messages found in state",
            "messages": [AIMessage(content="❌ No input message found")]
        }

    last_message = state["messages"][-1]
    if isinstance(last_message.content, list):
        # Handle multimodal/list content by joining text parts
        user_input = " ".join(
            str(part) for part in last_message.content 
            if isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
        ).lower()
    else:
        user_input = str(last_message.content).lower()

    # Extract task parameter
    task_mode = None
    if "--task" in user_input:
        # Parse --task value (e.g., --task train, --task develop)
        parts = user_input.split("--task")
        if len(parts) > 1:
            task_part = parts[1].strip().split()[0]
            task_mode = task_part.lower()

    # Validate task
    if task_mode not in ["build", "continue", "visualization", "mermaid", "html"]:
        error_msg = """Invalid or missing --task parameter.

Available tasks:
  --task build         Build new component (paper debate -> code -> docker test)
  --task continue      Resume from specific iteration
  --task visualization Run visualization analysis
  --task html          Generate HTML report

Flags:
  --model <name>       Target model (deeptta, deepdr, stagate, deepst, dlm-dti, hyperattentiondti)
  --iter <N>           Iteration count (default: 1)
  --stage <stage>      Resume stage: debate | development | docker | auto (continue only)
  --patience <N>       Reward patience (default: 10)
  --weight <float>     Reward weight 0-1 (default: 0.1)

Examples:
  --task build --model deeptta
  --task build --model deeptta --iter 3
  --task continue --model deeptta --iter 2
  --task continue --model deeptta --iter 2 --stage development
  --task visualization
  --task html
"""
        return {
            "next_node": "END",
            "router_decision": "ERROR",
            "router_reasoning": f"Invalid task: {task_mode}",
            "messages": [AIMessage(content=error_msg)]
        }

    # Extract model parameter
    target_model = None

    # Extract iteration count
    iteration_count = 1  # default
    if "--iter" in user_input:
        parts = user_input.split("--iter")
        if len(parts) > 1:
            iter_part = parts[1].strip().split()[0]
            try:
                iteration_count = int(iter_part)
                if iteration_count < 1:
                    iteration_count = 1
                logger.info(f"[ENTRY_ROUTER] Iteration count: {iteration_count}")
            except ValueError:
                iteration_count = 1
                logger.warning(f"[ENTRY_ROUTER] Invalid --iter value, using default: 1")

    # Extract stage for continue mode (debate, development, docker, auto)
    continue_stage = "debate"  # default: start from debate
    if "--stage" in user_input:
        parts = user_input.split("--stage")
        if len(parts) > 1:
            stage_part = parts[1].strip().split()[0].lower()
            valid_stages = ["debate", "development", "docker", "auto"]
            if stage_part in valid_stages:
                continue_stage = stage_part
                logger.info(f"[ENTRY_ROUTER] Continue stage: {continue_stage}")
            else:
                logger.warning(f"[ENTRY_ROUTER] Invalid --stage '{stage_part}', using default: debate")

    # Extract reward patience (reward block size)
    reward_patience = 10  # default
    if "--patience" in user_input:
        parts = user_input.split("--patience")
        if len(parts) > 1:
            patience_part = parts[1].strip().split()[0]
            try:
                reward_patience = int(patience_part)
                if reward_patience < 1:
                    reward_patience = 10
                logger.info(f"[ENTRY_ROUTER] Reward patience: {reward_patience}")
            except ValueError:
                reward_patience = 10
                logger.warning(f"[ENTRY_ROUTER] Invalid --patience value, using default: 10")

    # Extract reward weight
    reward_weight = 0.1  # default
    if "--weight" in user_input:
        parts = user_input.split("--weight")
        if len(parts) > 1:
            weight_part = parts[1].strip().split()[0]
            try:
                reward_weight = float(weight_part)
                if reward_weight < 0 or reward_weight > 1:
                    reward_weight = 0.1
                logger.info(f"[ENTRY_ROUTER] Reward weight: {reward_weight}")
            except ValueError:
                reward_weight = 0.1
                logger.warning(f"[ENTRY_ROUTER] Invalid --weight value, using default: 0.1")

    # Extract --model flag
    if "--model" in user_input:
        parts = user_input.split("--model")
        if len(parts) > 1:
            model_part = parts[1].strip().split()[0]
            target_model = model_part.lower()

    if not target_model:
        target_model = "unknown"

    # Multi-way routing logic
    is_continue_mode = False
    if task_mode == "build":
        # Route to build debate subgraph (paper-based component building)
        route = "build_debate_subgraph"
        reasoning = f"Task=build → Starting paper-based debate for new component"
    elif task_mode == "continue":
        # Continue mode: resume from a specific iteration
        route = "build_debate_subgraph"
        is_continue_mode = True
        reasoning = f"Task=continue → Resuming from iteration {iteration_count} (preserving existing build folders)"
        logger.info(f"[ENTRY_ROUTER] Continue mode: starting from iteration {iteration_count}")
    else:  # visualization / analysis / mermaid / html
        route = "analysis_subgraph"
        reasoning = f"Task={task_mode} → Run analysis subgraph"

    # Build result
    result = {
        "next_node": route,
        "router_decision": route,
        "router_reasoning": reasoning,
        "task_mode": task_mode,
        "target_model": target_model,
        "iteration_count": iteration_count,
        "total_iterations": iteration_count,  # iteration 워크플로우용
        "is_continue_mode": is_continue_mode,
        "reward_patience": reward_patience,
        "reward_weight": reward_weight,
        "messages": [AIMessage(content=f"✅ Routing to {route}\n🎯 Task: {task_mode} | Model: {target_model} | Iter: {iteration_count} | Patience: {reward_patience} | Weight: {reward_weight}" + (" (continue)" if is_continue_mode else ""))]
    }

    # Continue mode: set current_iteration and stage to start from
    if is_continue_mode:
        result["current_iteration"] = iteration_count
        result["continue_stage"] = continue_stage
        logger.info(f"[ENTRY_ROUTER] Continue mode: iteration={iteration_count}, stage={continue_stage}")

    return result


# Routing function for LangGraph conditional edges
def route_from_entry_router(state: MARBLEState) -> str:
    """Route from entry router based on multi-way routing decision.

    Valid routes:
    - init_iteration: build mode (iteration 지원)
    - analysis_subgraph: visualization/analysis mode
    - END: error case
    """
    next_node = state.get("next_node", "")

    # Validate route exists
    valid_routes = [
        "build_debate_subgraph",       # Build mode (paper-based) - iteration 미사용 시
        "build_development_subgraph",  # Build development (code implementation)
        "analysis_subgraph",           # Visualization/analysis mode
        "init_iteration",              # Build mode with iteration
        "END"
    ]

    # Build task는 init_iteration으로 라우팅 (iteration 지원)
    if next_node == "build_debate_subgraph":
        total_iterations = state.get("total_iterations", 1)
        if total_iterations > 1:
            logger.info(f"[ENTRY_ROUTER] Build task → init_iteration ({total_iterations}회 반복)")
            return "init_iteration"
        # iteration 1회면 기존 플로우 사용 가능하지만, 일관성을 위해 init_iteration 사용
        logger.info("[ENTRY_ROUTER] Build task → init_iteration (1회)")
        return "init_iteration"

    if next_node in valid_routes:
        return next_node

    # Invalid route, end graph
    return "END"
