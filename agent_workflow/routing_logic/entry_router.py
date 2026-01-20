"""Entry router for MARBLE multi-agent system."""

import shutil
from pathlib import Path
from typing import Dict
from langchain_core.messages import AIMessage

from agent_workflow.state import MARBLEState
from agent_workflow.logger import logger


def setup_model_directories(model_name: str) -> bool:
    """Setup model agent directories by copying from original model directory."""
    models_base_path = Path.cwd() / "models"
    experiments_base_path = Path.cwd() / "experiments"
    source_dir = models_base_path / model_name
    target_dir = experiments_base_path / f"{model_name}"

    try:
        experiments_base_path.mkdir(exist_ok=True)

        if target_dir.exists():
            return True

        if not source_dir.exists():
            return False

        shutil.copytree(source_dir, target_dir)

        logs_dir = target_dir / "logs"
        reports_dir = target_dir / "reports"
        logs_dir.mkdir(exist_ok=True)
        reports_dir.mkdir(exist_ok=True)

        return True

    except Exception:
        return False


def simple_entry_router(state: MARBLEState) -> Dict:
    """Simple routing function based on task parameter.

    Routes:
    - build/continue → init_iteration (via build_debate_subgraph)
    - visualization/mermaid/html → analysis_subgraph
    """
    if not state.get("messages"):
        return {
            "next_node": "END",
            "router_decision": "ERROR",
            "router_reasoning": "No messages found in state",
            "messages": [AIMessage(content="No input message found")]
        }

    last_message = state["messages"][-1]
    if isinstance(last_message.content, list):
        user_input = " ".join(
            str(part) for part in last_message.content
            if isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
        ).lower()
    else:
        user_input = str(last_message.content).lower()

    # Parse --task
    task_mode = None
    if "--task" in user_input:
        parts = user_input.split("--task")
        if len(parts) > 1:
            task_part = parts[1].strip().split()[0]
            task_mode = task_part.lower()

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

    # Parse --model
    target_model = None
    if "--model" in user_input:
        parts = user_input.split("--model")
        if len(parts) > 1:
            model_part = parts[1].strip().split()[0]
            target_model = model_part.lower()

    if not target_model:
        target_model = "unknown"

    # Parse --iter
    iteration_count = 1
    if "--iter" in user_input:
        parts = user_input.split("--iter")
        if len(parts) > 1:
            iter_part = parts[1].strip().split()[0]
            try:
                iteration_count = max(1, int(iter_part))
                logger.info(f"[ENTRY_ROUTER] Iteration count: {iteration_count}")
            except ValueError:
                iteration_count = 1
                logger.warning("[ENTRY_ROUTER] Invalid --iter value, using default: 1")

    # Parse --stage (for continue mode)
    continue_stage = "debate"
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

    # Parse --patience
    reward_patience = 10
    if "--patience" in user_input:
        parts = user_input.split("--patience")
        if len(parts) > 1:
            patience_part = parts[1].strip().split()[0]
            try:
                reward_patience = max(1, int(patience_part))
                logger.info(f"[ENTRY_ROUTER] Reward patience: {reward_patience}")
            except ValueError:
                reward_patience = 10
                logger.warning("[ENTRY_ROUTER] Invalid --patience value, using default: 10")

    # Parse --weight
    reward_weight = 0.1
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
                logger.warning("[ENTRY_ROUTER] Invalid --weight value, using default: 0.1")

    # Routing logic
    is_continue_mode = False
    if task_mode == "build":
        route = "build_debate_subgraph"
        reasoning = "Task=build -> Starting paper-based debate for new component"
    elif task_mode == "continue":
        route = "build_debate_subgraph"
        is_continue_mode = True
        reasoning = f"Task=continue -> Resuming from iteration {iteration_count}"
        logger.info(f"[ENTRY_ROUTER] Continue mode: starting from iteration {iteration_count}")
    else:
        route = "analysis_subgraph"
        reasoning = f"Task={task_mode} -> Run analysis subgraph"

    result = {
        "next_node": route,
        "router_decision": route,
        "router_reasoning": reasoning,
        "task_mode": task_mode,
        "target_model": target_model,
        "iteration_count": iteration_count,
        "total_iterations": iteration_count,
        "is_continue_mode": is_continue_mode,
        "reward_patience": reward_patience,
        "reward_weight": reward_weight,
        "messages": [AIMessage(content=f"Routing to {route}\nTask: {task_mode} | Model: {target_model} | Iter: {iteration_count} | Patience: {reward_patience} | Weight: {reward_weight}" + (" (continue)" if is_continue_mode else ""))]
    }

    if is_continue_mode:
        result["current_iteration"] = iteration_count
        result["continue_stage"] = continue_stage
        logger.info(f"[ENTRY_ROUTER] Continue mode: iteration={iteration_count}, stage={continue_stage}")

    return result


def route_from_entry_router(state: MARBLEState) -> str:
    """Route from entry router based on routing decision.

    Valid routes:
    - init_iteration: build/continue mode
    - analysis_subgraph: visualization/analysis mode
    - END: error case
    """
    next_node = state.get("next_node", "")

    valid_routes = [
        "build_debate_subgraph",
        "build_development_subgraph",
        "analysis_subgraph",
        "init_iteration",
        "END"
    ]

    # Build task routes to init_iteration
    if next_node == "build_debate_subgraph":
        total_iterations = state.get("total_iterations", 1)
        logger.info(f"[ENTRY_ROUTER] Build task -> init_iteration ({total_iterations} iterations)")
        return "init_iteration"

    if next_node in valid_routes:
        return next_node

    return "END"
