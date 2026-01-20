"""Main graph builder for MARBLE multi-agent system."""

import asyncio
import atexit
import os
import signal
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from configs.config import ACTIVE_NODES, LANGGRAPH_CONFIG, MAIN_GRAPH_EDGES, WORKFLOW_CONFIG
from agent_workflow.logger import logger
from agent_workflow.mcp_connection_manager import MCPManager
from agent_workflow.evolving_memory import EvolvingMemory
from agent_workflow.state import MARBLEState
from agent_workflow.routing_logic.entry_router import route_from_entry_router, simple_entry_router
from agent_workflow.workflow_subgraphs.analysis_workflow import get_analysis_subgraph
from agent_workflow.workflow_subgraphs.build_debate_workflow import get_build_debate_subgraph
from agent_workflow.workflow_subgraphs.build_development_workflow import get_build_development_subgraph
from agent_workflow.workflow_subgraphs.docker_execution import get_docker_execution_subgraph
from agent_workflow.iteration_nodes import (
    init_iteration_node,
    inject_memory_context_node,
    save_to_memory_node,
    check_continue_node,
    route_after_save_to_memory,
    route_from_inject_memory,
)
from agent_workflow.utils import GlobalStateManager, get_project_root

load_dotenv()

_GRAPH_BUILD_COUNT = 0
_SYSTEM_INITIALIZED = False

mcp_manager = MCPManager()

try:
    from langgraph.errors import GraphRecursionError
except Exception:
    GraphRecursionError = None


def _extract_flag_value(text: str, flag: str) -> Optional[str]:
    if not text or flag not in text:
        return None
    parts = text.split(flag, 1)
    if len(parts) < 2:
        return None
    value = parts[1].strip().split()[0] if parts[1].strip() else None
    return value


def _get_last_message_text(input_state: Any) -> Optional[str]:
    if not isinstance(input_state, dict):
        return None
    messages = input_state.get("messages")
    if not messages:
        return None
    last_message = messages[-1]
    if isinstance(last_message, dict):
        content = last_message.get("content")
    else:
        content = getattr(last_message, "content", last_message)
    if isinstance(content, list):
        return " ".join(
            str(part) for part in content
            if isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
        )
    return str(content)


def _is_recursion_error(exc: Exception) -> bool:
    if GraphRecursionError and isinstance(exc, GraphRecursionError):
        return True
    if isinstance(exc, RecursionError):
        return True
    message = str(exc).lower()
    return "graphrecursionerror" in message or "recursion_limit" in message


def _get_iteration_progress(workspace: Path, iteration: int) -> dict:
    """Get iteration progress based on file existence."""
    build_path = workspace / f"build_{iteration}"

    debate_done = (build_path / "build_debate_outputs" / "implementation_proposal.md").exists()
    development_done = bool(list((build_path / "src").glob("*.py"))) if (build_path / "src").exists() else False
    docker_done = (build_path / "docker_result.json").exists() or (build_path / "build_debate_outputs" / "execution_result.json").exists()

    return {
        "iteration": iteration,
        "debate_done": debate_done,
        "development_done": development_done,
        "docker_done": docker_done,
    }


def _find_resume_point(workspace: Path, target_model: str) -> tuple[int, str]:
    """Find most recent iteration and stage to resume from."""
    _ = target_model
    mb = EvolvingMemory(workspace_path=str(workspace))
    session_info = mb.get_session_info()
    completed = session_info.get("completed_iterations", 0)

    next_iter = completed + 1
    progress = _get_iteration_progress(workspace, next_iter)

    if progress["development_done"] and not progress["docker_done"]:
        return (next_iter, "docker")
    elif progress["debate_done"] and not progress["development_done"]:
        return (next_iter, "development")
    else:
        return (next_iter, "debate")


def _has_made_progress(workspace: Path, prev_state: dict, current_state: dict) -> bool:
    """Check if progress was made since last attempt."""
    if not prev_state:
        return True

    if current_state.get("completed", 0) > prev_state.get("completed", 0):
        return True

    curr_iter = current_state.get("next_iter", 1)
    prev_progress = prev_state.get("progress", {})
    curr_progress = _get_iteration_progress(workspace, curr_iter)

    if not prev_progress.get("debate_done") and curr_progress.get("debate_done"):
        return True
    if not prev_progress.get("development_done") and curr_progress.get("development_done"):
        return True
    if not prev_progress.get("docker_done") and curr_progress.get("docker_done"):
        return True

    return False


def _get_auto_continue_plan(input_state: Any) -> Optional[Dict[str, Any]]:
    """Get auto-continue plan for recursion error recovery."""
    message_text = _get_last_message_text(input_state)
    if not message_text:
        return None

    lower_text = message_text.lower()
    task_mode = _extract_flag_value(lower_text, "--task")
    if task_mode not in ("build", "continue"):
        return None

    fallback_model = _extract_flag_value(lower_text, "--model")
    workspace = Path(get_project_root()) / "experiments"
    mb = EvolvingMemory(workspace_path=str(workspace))
    session_info = mb.get_session_info()
    planned = session_info.get("planned_iterations")
    completed = session_info.get("completed_iterations", 0)
    target_model = session_info.get("target_model") or fallback_model or "unknown"

    reward_settings = mb.get_reward_settings()
    reward_patience = reward_settings.get("patience", 10)
    reward_weight = reward_settings.get("weight", 0.1)

    next_iter, resume_stage = _find_resume_point(workspace, target_model)
    progress = _get_iteration_progress(workspace, next_iter)

    if planned and planned > 0 and next_iter > planned:
        logger.warning(f"[AUTO-CONTINUE] planned={planned}, completed={completed}. Nothing to resume.")
        return None

    continue_message = (
        f"Auto continue after recursion error --task continue --model {target_model} "
        f"--iter {next_iter} --stage {resume_stage} --patience {reward_patience} --weight {reward_weight}"
    )
    resume_state = dict(input_state) if isinstance(input_state, dict) else {}
    resume_state["messages"] = [HumanMessage(content=continue_message)]

    return {
        "resume_state": resume_state,
        "next_iter": next_iter,
        "planned": planned,
        "completed": completed,
        "target_model": target_model,
        "resume_stage": resume_stage,
        "progress": progress,
        "reward_patience": reward_patience,
        "reward_weight": reward_weight,
    }


async def _ainvoke_with_auto_continue(
    original_ainvoke,
    input_state: Any,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Any:
    """Async invoke with auto-continue on recursion error."""
    try:
        return await original_ainvoke(input_state, config=config, **kwargs)
    except Exception as exc:
        if not _is_recursion_error(exc):
            raise

    plan = _get_auto_continue_plan(input_state)
    if not plan:
        raise exc

    max_retries = int(os.getenv("AUTO_CONTINUE_MAX_RETRIES", "100"))
    retry_delay = int(os.getenv("AUTO_CONTINUE_RETRY_DELAY", "5"))
    workspace = Path(get_project_root()) / "experiments"
    prev_state = None
    no_progress_count = 0
    max_no_progress = 3
    last_exc = exc

    for attempt in range(1, max_retries + 1):
        current_state = {
            "completed": plan["completed"],
            "next_iter": plan["next_iter"],
            "progress": plan.get("progress", {}),
        }

        if not _has_made_progress(workspace, prev_state, current_state):
            no_progress_count += 1
            logger.warning(f"[AUTO-CONTINUE] No progress ({no_progress_count}/{max_no_progress})")
            if no_progress_count >= max_no_progress:
                logger.error("[AUTO-CONTINUE] Max no-progress retries reached")
                break
        else:
            no_progress_count = 0

        prev_state = current_state

        logger.warning(
            f"[AUTO-CONTINUE] Recursion error. Attempt {attempt}/{max_retries} "
            f"iter {plan['next_iter']} stage={plan.get('resume_stage', 'debate')} "
            f"(planned={plan['planned']}, completed={plan['completed']})"
        )

        if attempt > 1 and retry_delay > 0:
            logger.info(f"[AUTO-CONTINUE] Waiting {retry_delay}s...")
            await asyncio.sleep(retry_delay)

        try:
            return await original_ainvoke(plan["resume_state"], config=config, **kwargs)
        except Exception as retry_exc:
            if not _is_recursion_error(retry_exc):
                raise
            last_exc = retry_exc
            plan = _get_auto_continue_plan(input_state) or plan

    raise last_exc


def _invoke_with_auto_continue(
    original_invoke,
    input_state: Any,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Any:
    """Sync invoke with auto-continue on recursion error."""
    import time

    try:
        return original_invoke(input_state, config=config, **kwargs)
    except Exception as exc:
        if not _is_recursion_error(exc):
            raise

    plan = _get_auto_continue_plan(input_state)
    if not plan:
        raise exc

    max_retries = int(os.getenv("AUTO_CONTINUE_MAX_RETRIES", "100"))
    retry_delay = int(os.getenv("AUTO_CONTINUE_RETRY_DELAY", "5"))
    workspace = Path(get_project_root()) / "experiments"
    prev_state = None
    no_progress_count = 0
    max_no_progress = 3
    last_exc = exc

    for attempt in range(1, max_retries + 1):
        current_state = {
            "completed": plan["completed"],
            "next_iter": plan["next_iter"],
            "progress": plan.get("progress", {}),
        }

        if not _has_made_progress(workspace, prev_state, current_state):
            no_progress_count += 1
            logger.warning(f"[AUTO-CONTINUE] No progress ({no_progress_count}/{max_no_progress})")
            if no_progress_count >= max_no_progress:
                logger.error("[AUTO-CONTINUE] Max no-progress retries reached")
                break
        else:
            no_progress_count = 0

        prev_state = current_state

        logger.warning(
            f"[AUTO-CONTINUE] Recursion error. Attempt {attempt}/{max_retries} "
            f"iter {plan['next_iter']} stage={plan.get('resume_stage', 'debate')} "
            f"(planned={plan['planned']}, completed={plan['completed']})"
        )

        if attempt > 1 and retry_delay > 0:
            logger.info(f"[AUTO-CONTINUE] Waiting {retry_delay}s...")
            time.sleep(retry_delay)

        try:
            return original_invoke(plan["resume_state"], config=config, **kwargs)
        except Exception as retry_exc:
            if not _is_recursion_error(retry_exc):
                raise
            last_exc = retry_exc
            plan = _get_auto_continue_plan(input_state) or plan

    raise last_exc


def _ensure_recursion_limit(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure config has recursion_limit set."""
    from configs.config import LANGGRAPH_CONFIG

    if config is None:
        config = {}
    else:
        config = dict(config)

    if "recursion_limit" not in config:
        config["recursion_limit"] = LANGGRAPH_CONFIG["recursion_limit"]
        logger.debug(f"[CONFIG] recursion_limit set: {config['recursion_limit']}")

    return config


def _wrap_graph_with_auto_continue(graph):
    """Wrap graph methods with auto-continue functionality."""
    original_ainvoke = graph.ainvoke
    original_invoke = graph.invoke
    original_astream = getattr(graph, "astream", None)
    original_stream = getattr(graph, "stream", None)
    original_astream_events = getattr(graph, "astream_events", None)
    original_stream_events = getattr(graph, "stream_events", None)

    async def ainvoke(input_state: Any, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        config = _ensure_recursion_limit(config)
        return await _ainvoke_with_auto_continue(original_ainvoke, input_state, config, **kwargs)

    def invoke(input_state: Any, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        config = _ensure_recursion_limit(config)
        return _invoke_with_auto_continue(original_invoke, input_state, config, **kwargs)

    async def astream(input_state: Any, config: Optional[Dict[str, Any]] = None, **kwargs):
        config = _ensure_recursion_limit(config)
        max_retries = int(os.getenv("AUTO_CONTINUE_MAX_RETRIES", "100"))
        attempt = 0
        while True:
            try:
                async for chunk in original_astream(input_state, config=config, **kwargs):
                    yield chunk
                return
            except Exception as exc:
                if not _is_recursion_error(exc):
                    raise
                plan = _get_auto_continue_plan(input_state)
                attempt += 1
                if not plan or attempt > max_retries:
                    raise
                logger.warning(
                    f"[AUTO-CONTINUE] Recursion error during stream. "
                    f"Attempt {attempt}/{max_retries} iter {plan['next_iter']}"
                )
                input_state = plan["resume_state"]

    def stream(input_state: Any, config: Optional[Dict[str, Any]] = None, **kwargs):
        config = _ensure_recursion_limit(config)
        max_retries = int(os.getenv("AUTO_CONTINUE_MAX_RETRIES", "100"))
        attempt = 0
        while True:
            try:
                for chunk in original_stream(input_state, config=config, **kwargs):
                    yield chunk
                return
            except Exception as exc:
                if not _is_recursion_error(exc):
                    raise
                plan = _get_auto_continue_plan(input_state)
                attempt += 1
                if not plan or attempt > max_retries:
                    raise
                logger.warning(
                    f"[AUTO-CONTINUE] Recursion error during stream. "
                    f"Attempt {attempt}/{max_retries} iter {plan['next_iter']}"
                )
                input_state = plan["resume_state"]

    async def astream_events(input_state: Any, config: Optional[Dict[str, Any]] = None, **kwargs):
        config = _ensure_recursion_limit(config)
        max_retries = int(os.getenv("AUTO_CONTINUE_MAX_RETRIES", "100"))
        attempt = 0
        while True:
            try:
                async for chunk in original_astream_events(input_state, config=config, **kwargs):
                    yield chunk
                return
            except Exception as exc:
                if not _is_recursion_error(exc):
                    raise
                plan = _get_auto_continue_plan(input_state)
                attempt += 1
                if not plan or attempt > max_retries:
                    raise
                logger.warning(
                    f"[AUTO-CONTINUE] Recursion error during event stream. "
                    f"Attempt {attempt}/{max_retries} iter {plan['next_iter']}"
                )
                input_state = plan["resume_state"]

    def stream_events(input_state: Any, config: Optional[Dict[str, Any]] = None, **kwargs):
        config = _ensure_recursion_limit(config)
        max_retries = int(os.getenv("AUTO_CONTINUE_MAX_RETRIES", "100"))
        attempt = 0
        while True:
            try:
                for chunk in original_stream_events(input_state, config=config, **kwargs):
                    yield chunk
                return
            except Exception as exc:
                if not _is_recursion_error(exc):
                    raise
                plan = _get_auto_continue_plan(input_state)
                attempt += 1
                if not plan or attempt > max_retries:
                    raise
                logger.warning(
                    f"[AUTO-CONTINUE] Recursion error during event stream. "
                    f"Attempt {attempt}/{max_retries} iter {plan['next_iter']}"
                )
                input_state = plan["resume_state"]

    graph.ainvoke = ainvoke
    graph.invoke = invoke
    if original_astream:
        graph.astream = astream
    if original_stream:
        graph.stream = stream
    if original_astream_events:
        graph.astream_events = astream_events
    if original_stream_events:
        graph.stream_events = stream_events
    return graph


def build_main_graph() -> StateGraph:
    """Build main graph with 2-way routing."""
    logger.debug("Building main graph...")

    try:
        if not mcp_manager._initialized:
            asyncio.run(initialize_system())
    except Exception as e:
        logger.warning(f"[MainGraph] MCP init skipped: {e}")

    builder = StateGraph(MARBLEState)

    if ACTIVE_NODES.get("entry_router", False):
        builder.add_node("entry_router", simple_entry_router)
        logger.debug("[Main] Entry Router added")

    if ACTIVE_NODES.get("analysis_subgraph", False):
        builder.add_node("analysis_subgraph", get_analysis_subgraph())
        logger.debug("[Main] Analysis Subgraph added")

    if ACTIVE_NODES.get("build_debate_subgraph", False):
        build_debate_subgraph, _ = get_build_debate_subgraph()
        builder.add_node("build_debate_subgraph", build_debate_subgraph)
        logger.debug("[Main] Build Debate Subgraph added")

    if ACTIVE_NODES.get("build_development_subgraph", False):
        build_development_subgraph, _ = get_build_development_subgraph()
        builder.add_node("build_development_subgraph", build_development_subgraph)
        logger.debug("[Main] Build Development Subgraph added")

    if ACTIVE_NODES.get("docker_execution_subgraph", False):
        docker_execution_subgraph = get_docker_execution_subgraph()
        builder.add_node("docker_execution_subgraph", docker_execution_subgraph)
        logger.debug("[Main] Docker Execution Subgraph added")

    builder.add_node("init_iteration", init_iteration_node)
    builder.add_node("inject_memory_context", inject_memory_context_node)
    builder.add_node("save_to_memory", save_to_memory_node)
    builder.add_node("check_continue", check_continue_node)
    logger.debug("[Main] Iteration nodes added")

    if ACTIVE_NODES.get("entry_router", False):
        builder.set_entry_point("entry_router")
        logger.debug("[Main] Entry point: entry_router")
    else:
        logger.warning("No entry point configured")

    for source_node, target_config in MAIN_GRAPH_EDGES.items():
        if not ACTIVE_NODES.get(source_node, False):
            continue

        if isinstance(target_config, str):
            if target_config == "END":
                builder.add_edge(source_node, END)
                logger.debug(f"  Edge: {source_node} -> END")
            elif ACTIVE_NODES.get(target_config, False):
                builder.add_edge(source_node, target_config)
                logger.debug(f"  Edge: {source_node} -> {target_config}")

        elif isinstance(target_config, dict) and target_config.get("type") == "conditional":
            function_name = target_config["function"]
            routes = target_config["routes"]

            filtered_routes = {}
            for route_key, route_value in routes.items():
                if route_value == "END":
                    filtered_routes[route_key] = END
                elif ACTIVE_NODES.get(route_value, False):
                    filtered_routes[route_key] = route_value
            if function_name == "route_from_entry_router" and filtered_routes:
                builder.add_conditional_edges(source_node, route_from_entry_router, filtered_routes)
                logger.debug(f"  Conditional: {source_node} -> {list(filtered_routes.keys())}")

    if ACTIVE_NODES.get("build_debate_subgraph", False):
        builder.add_edge("init_iteration", "inject_memory_context")

        builder.add_conditional_edges(
            "inject_memory_context",
            route_from_inject_memory,
            {
                "debate": "build_debate_subgraph",
                "development": "build_development_subgraph",
                "docker": "docker_execution_subgraph",
            }
        )

        builder.add_edge("docker_execution_subgraph", "save_to_memory")
        builder.add_edge("save_to_memory", "check_continue")

        builder.add_conditional_edges(
            "check_continue",
            route_after_save_to_memory,
            {
                "continue_iteration": "inject_memory_context",
                "end": END,
            }
        )
        logger.debug("[Main] Iteration flow edges added")

    from configs.config import LANGGRAPH_CONFIG
    recursion_limit = LANGGRAPH_CONFIG.get("recursion_limit", 5000)
    compiled_graph = builder.compile().with_config({"recursion_limit": recursion_limit})
    logger.debug(f"[Main] Compiled with recursion_limit={recursion_limit}")

    global _GRAPH_BUILD_COUNT
    _GRAPH_BUILD_COUNT += 1

    node_count = len(ACTIVE_NODES)
    edge_count = len(MAIN_GRAPH_EDGES)

    if _GRAPH_BUILD_COUNT == 1:
        logger.info(f"Graph: {node_count} agents, {edge_count} edges")
    else:
        logger.debug(f"Graph: {node_count} agents, {edge_count} edges (rebuild #{_GRAPH_BUILD_COUNT})")

    return compiled_graph


async def initialize_system():
    """Initialize MCP system."""
    global _SYSTEM_INITIALIZED

    try:
        logger.debug("[SYSTEM] Initializing MARBLE...")
        await mcp_manager.initialize_all_servers()

        if mcp_manager.mcp_enabled:
            total_tools = sum(len(tools) for tools in mcp_manager.tools.values())
            server_count = len([s for s, tools in mcp_manager.tools.items() if tools])

            if not _SYSTEM_INITIALIZED:
                logger.info(f"MCP: {server_count} servers, {total_tools} tools")
                _SYSTEM_INITIALIZED = True
            else:
                logger.debug(f"MCP: {server_count} servers, {total_tools} tools [cached]")

            try:
                from external.open_deep_research.src.open_deep_research.utils_patch import patch_odr_utils, set_autodrp_tool_cache

                patch_odr_utils()

                arxiv_container = None
                if mcp_manager.docker_client:
                    try:
                        containers = mcp_manager.docker_client.containers.list()
                        for container in containers:
                            if "arxiv-semantic" in container.name:
                                arxiv_container = container.name
                                logger.debug(f"Found ArXiv container: {arxiv_container}")
                                break
                    except Exception as e:
                        logger.debug(f"Could not check ArXiv container: {e}")

                set_autodrp_tool_cache(mcp_manager.tools, arxiv_container)
                logger.debug("[SYSTEM] ODR bridge applied")

            except (ImportError, Exception):
                pass
        else:
            logger.info("MCP disabled - basic LLM mode")

        GlobalStateManager.initialize()
        GlobalStateManager.set_mcp_manager(mcp_manager)
        logger.debug("[SYSTEM] MCP Manager registered")
        logger.debug("[SYSTEM] MARBLE initialization complete")
        return True

    except Exception as e:
        logger.error(f"[SYSTEM] Initialization failed: {e}")
        raise


async def create_app():
    """Create LangGraph application with MCP initialization."""
    try:
        logger.debug("[GRAPH] Initializing MCP...")
        await initialize_system()
        await asyncio.sleep(0.5)

        graph = _wrap_graph_with_auto_continue(build_main_graph())

        global _GRAPH_BUILD_COUNT
        if _GRAPH_BUILD_COUNT <= 1:
            logger.info("MARBLE Ready")
        else:
            logger.debug(f"MARBLE Ready (rebuild #{_GRAPH_BUILD_COUNT})")

        return graph

    except Exception as e:
        logger.error(f"[GRAPH] Failed to create app: {e}")
        raise


logger.info("[LANGGRAPH] Initializing graph...")
try:
    asyncio.run(initialize_system())
except Exception as e:
    logger.warning(f"[LANGGRAPH] MCP init skipped: {e}")

app = _wrap_graph_with_auto_continue(build_main_graph())
logger.info("[LANGGRAPH] Graph initialized")


def signal_handler(signum, frame):
    """Signal handler for graceful shutdown."""
    logger.info(f"[SYSTEM] Signal {signum} received, shutting down...")
    cleanup_resources()
    os._exit(0)


def cleanup_resources():
    """Clean up system resources."""
    try:
        if mcp_manager:
            mcp_manager.stop_all_servers()
        logger.info("[CLEANUP] Resources cleaned")
    except Exception as e:
        logger.warning(f"[CLEANUP] Error: {e}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_resources)

    async def main():
        graph = await app()

        initial_state = {
            "messages": [],
            "max_iterations": WORKFLOW_CONFIG["max_iterations"],
            "improvement_threshold": WORKFLOW_CONFIG["improvement_threshold"],
            "processing_logs": [],
            "turn_count": 0,
            "max_turns": 50,
            "development_iteration_count": 0,
            "debate_complete": False,
            "final_report_generated": False,
        }

        logger.info("Starting workflow...")
        result = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": LANGGRAPH_CONFIG["recursion_limit"]}
        )

        logger.info("[RESULT] Workflow completed")
        logger.info(f"  Final iteration: {result.get('iteration_count', 0)}")
        logger.info(f"  Improvement: {result.get('overall_improvement', 0)}%")
        logger.info(f"  Best model: {result.get('best_refined_model', 'Unknown')}")

    asyncio.run(main())
