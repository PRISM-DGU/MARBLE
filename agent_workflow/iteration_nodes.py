"""Iteration workflow nodes."""

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_workflow.logger import logger
from agent_workflow.state import MARBLEState
from agent_workflow.evolving_memory import EvolvingMemory, IterationAnalyzerAgent
from agent_workflow.utils import get_project_root


def get_build_path(iteration: int) -> Path:
    """Get build folder path for iteration."""
    return Path(get_project_root()) / "experiments" / f"build_{iteration}"


def _parse_implementation_proposal(proposal_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Parse component and description from implementation_proposal.md."""
    if not proposal_path.exists():
        logger.warning(f"[ITERATION] Proposal file not found: {proposal_path}")
        return None, None

    try:
        content = proposal_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[ITERATION] Failed to read proposal: {e}")
        return None, None

    component = None
    description = None

    component_patterns = [
        r"Component to Modify[:\s]*\*?\*?([a-zA-Z_]+)",
        r"Target Component[:\s]*\*?\*?([a-zA-Z_]+)",
        r"modify the[:\s]*\*?\*?([a-zA-Z_]+)",
    ]
    for pattern in component_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            component = match.group(1).lower().strip("*").strip()
            break

    description_parts = []

    decision_match = re.search(
        r"Decision Summary[^\n]*\n+(.*?)(?=\n##|\n---|\Z)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if decision_match:
        decision_text = decision_match.group(1).strip()
        lines = [l.strip().lstrip("-*").strip() for l in decision_text.split("\n") if l.strip()]
        if lines:
            description_parts.append(lines[0][:200])

    arch_match = re.search(
        r"(?:New Architecture|Architecture|Proposed Architecture)[^\n]*\n+(.*?)(?=\n##|\n---|\Z)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if arch_match:
        arch_text = arch_match.group(1).strip()
        lines = [l.strip().lstrip("-*").strip() for l in arch_text.split("\n") if l.strip()]
        if lines and lines[0] not in description_parts:
            description_parts.append(lines[0][:100])

    config_match = re.search(
        r"Config Changes[^\n]*\n+(.*?)(?=\n##|\n---|\Z)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if config_match:
        config_text = config_match.group(1).strip()
        param_matches = re.findall(r"\*?\*?Parameter\*?\*?[:\s]*([^\n]+)", config_text, re.IGNORECASE)
        if param_matches:
            description_parts.append(f"Config: {param_matches[0][:50]}")

    if description_parts:
        description = " | ".join(description_parts)
    else:
        clean_content = re.sub(r"[#\*\-]+", "", content[:500]).strip()
        if clean_content:
            description = clean_content[:150] + "..."

    logger.info(f"[ITERATION] Proposal parsed: component={component}, description={description[:50] if description else None}...")
    return component, description


def _extract_proposal_sections(proposal_path: Path) -> Optional[str]:
    """Extract key sections from implementation_proposal.md.

    Sections: Decision Summary, Architecture Overview, Config Changes
    """
    if not proposal_path.exists():
        logger.warning(f"[ITERATION] Proposal file not found: {proposal_path}")
        return None

    try:
        content = proposal_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[ITERATION] Failed to read proposal: {e}")
        return None

    sections = []

    decision_match = re.search(
        r"(##\s*1\.?\s*Decision Summary.*?)(?=##\s*2\.|##\s*3\.|$)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if decision_match:
        sections.append(decision_match.group(1).strip())

    arch_match = re.search(
        r"(##\s*2\.?\s*Architecture Overview.*?)(?=##\s*3\.|##\s*4\.|$)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if arch_match:
        sections.append(arch_match.group(1).strip())

    config_match = re.search(
        r"(##\s*6\.?\s*Config Changes.*?)(?=##\s*7\.|##\s*8\.|$)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if config_match:
        sections.append(config_match.group(1).strip())

    if sections:
        result = "\n\n".join(sections)
        logger.info(f"[ITERATION] Proposal sections extracted: {len(result)} chars")
        return result

    return None


def _extract_weakness_summary(weakness_path: Path) -> Optional[str]:
    """Extract content from weakness_of_target_model.md."""
    if not weakness_path.exists():
        logger.warning(f"[ITERATION] Weakness file not found: {weakness_path}")
        return None

    try:
        content = weakness_path.read_text(encoding="utf-8")
        logger.info(f"[ITERATION] Weakness file extracted: {len(content)} chars")
        return content.strip()
    except Exception as e:
        logger.warning(f"[ITERATION] Failed to read weakness file: {e}")
        return None


def _extract_paper_titles(other_papers_path: Path) -> List[str]:
    """Extract paper titles from other_papers.json."""
    if not other_papers_path.exists():
        logger.warning(f"[ITERATION] other_papers.json not found: {other_papers_path}")
        return []

    try:
        data = json.loads(other_papers_path.read_text(encoding="utf-8"))
        titles = [p["title"] for p in data.get("selected_papers", []) if p.get("title")]
        logger.info(f"[ITERATION] Paper titles extracted: {len(titles)}")
        return titles
    except Exception as e:
        logger.warning(f"[ITERATION] Failed to parse other_papers.json: {e}")
        return []


def init_iteration_node(state: MARBLEState) -> Dict[str, Any]:
    """Initialize iteration node.

    Normal mode (--task build):
    - Clear EvolvingMemory
    - Set current_iteration = 1
    - Create build_1 ~ build_N folders

    Continue mode (--task continue):
    - Keep EvolvingMemory
    - Set current_iteration from --iter N
    - Keep existing build folders
    """
    total_iterations = state.get("total_iterations", 1)
    target_model = state.get("target_model", "unknown")
    is_continue_mode = state.get("is_continue_mode", False)
    start_iteration = state.get("current_iteration", 1) if is_continue_mode else 1
    reward_patience = state.get("reward_patience", 10)
    reward_weight = state.get("reward_weight", 0.1)

    workspace = Path(get_project_root()) / "experiments"
    mb = EvolvingMemory(workspace_path=str(workspace))

    if is_continue_mode:
        session_info = mb.get_session_info()
        planned = session_info.get("planned_iterations")
        saved_model = session_info.get("target_model")
        completed = session_info.get("completed_iterations", 0)

        if planned is None:
            logger.warning("[ITERATION] Continue mode: no session info in EvolvingMemory")
            planned = start_iteration

        actual_total = planned
        logger.info(f"[ITERATION] Continue mode: resuming from iteration {start_iteration}")
        logger.info(f"[ITERATION] Plan: {planned}, completed: {completed}, model: {saved_model or target_model}")

        existing_reward = mb.get_reward_settings()
        if reward_patience != 10 or reward_weight != 0.1:
            mb.set_reward_settings(patience=reward_patience, weight=reward_weight)
            logger.info(f"[ITERATION] Continue: reward updated (patience={reward_patience}, weight={reward_weight})")
        else:
            reward_patience = existing_reward.get("patience", 10)
            reward_weight = existing_reward.get("weight", 0.1)
            logger.info(f"[ITERATION] Continue: using existing reward (patience={reward_patience}, weight={reward_weight})")

        build_path = get_build_path(start_iteration)
        build_path.mkdir(parents=True, exist_ok=True)
        (build_path / "build_debate_outputs").mkdir(exist_ok=True)
        (build_path / "src").mkdir(exist_ok=True)
        logger.info(f"[ITERATION] build_{start_iteration} ready")

        return {
            "current_iteration": start_iteration,
            "total_iterations": actual_total,
            "iteration_context": str(mb.memory_file) if start_iteration > 1 else "",
            "reward_patience": reward_patience,
            "reward_weight": reward_weight,
            "processing_logs": [f"[ITERATION] Continue: iteration {start_iteration}/{actual_total}"],
        }
    else:
        logger.info(f"[ITERATION] Init: {total_iterations} iterations planned (model: {target_model})")

        mb.clear()
        mb.init_baseline(target_model)
        mb.set_session_info(planned_iterations=total_iterations, target_model=target_model)
        mb.set_reward_settings(patience=reward_patience, weight=reward_weight)

        for i in range(0, 100):
            old_build = get_build_path(i)
            if old_build.exists():
                shutil.rmtree(str(old_build))
                logger.info(f"[ITERATION] Deleted old folder: {old_build}")
            elif i > 0:
                break

        build_0_path = get_build_path(0)
        source_path = Path(get_project_root()) / "docker_images" / target_model
        if source_path.exists():
            shutil.copytree(
                str(source_path),
                str(build_0_path),
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
            )
            logger.info(f"[ITERATION] build_0 created (baseline): {source_path} -> {build_0_path}")
        else:
            logger.warning(f"[ITERATION] docker_images/{target_model} not found - skipping build_0")

        for i in range(1, total_iterations + 1):
            build_path = get_build_path(i)
            build_path.mkdir(parents=True, exist_ok=True)
            (build_path / "build_debate_outputs").mkdir(exist_ok=True)
            (build_path / "src").mkdir(exist_ok=True)
            logger.info(f"[ITERATION] build_{i} created")

        return {
            "current_iteration": 1,
            "total_iterations": total_iterations,
            "iteration_context": "",
            "reward_patience": reward_patience,
            "reward_weight": reward_weight,
            "processing_logs": [f"[ITERATION] Init complete: {total_iterations} iterations, build_1~{total_iterations} created"],
        }


def inject_memory_context_node(state: MARBLEState) -> Dict[str, Any]:
    """Inject previous iteration context from EvolvingMemory."""
    current_iteration = state.get("current_iteration", 1)

    workspace = Path(get_project_root()) / "experiments"
    mb = EvolvingMemory(workspace_path=str(workspace))

    logs = [f"[ITERATION] Iteration {current_iteration} starting"]

    if current_iteration > 1:
        context = str(mb.memory_file)
        logger.info(f"[ITERATION] Memory file: {context}")

        session_info = mb.get_session_info()
        completed = session_info.get("completed_iterations", 0)
        planned = session_info.get("planned_iterations", 0)

        logs.append(f"[MEMORY] Progress: {completed}/{planned} iterations")

        if mb.data.best_iteration is not None and mb.data.best_performance:
            best_perf = mb.data.best_performance
            perf_str = []
            if best_perf.rmse is not None:
                perf_str.append(f"RMSE={best_perf.rmse:.4f}")
            if best_perf.pearson is not None:
                perf_str.append(f"Pearson={best_perf.pearson:.4f}")
            if best_perf.ari is not None:
                perf_str.append(f"ARI={best_perf.ari:.4f}")
            if best_perf.auprc is not None:
                perf_str.append(f"AUPRC={best_perf.auprc:.4f}")
            if perf_str:
                logs.append(f"[MEMORY] Best: iter {mb.data.best_iteration} ({', '.join(perf_str)})")
                logger.info(f"[ITERATION] Best: iter {mb.data.best_iteration} ({', '.join(perf_str)})")

        last_record = mb.get_last_iteration()
        if last_record:
            if last_record.analysis.improved is True:
                result_str = "improved"
            elif last_record.analysis.improved is False:
                result_str = "degraded"
            else:
                result_str = "baseline"
            logs.append(f"[MEMORY] Last iter {last_record.iteration}: {result_str}")
            if last_record.analysis.reason:
                logs.append(f"[MEMORY] Reason: {last_record.analysis.reason[:100]}...")
            logger.info(f"[ITERATION] Last iter {last_record.iteration}: {result_str}")

        if mb.data.key_lessons:
            recent_lessons = mb.data.key_lessons[-3:]
            logs.append(f"[MEMORY] Key lessons ({len(mb.data.key_lessons)} total, recent 3):")
            for i, lesson in enumerate(recent_lessons, 1):
                logs.append(f"  {i}. {lesson[:80]}...")
                logger.info(f"[ITERATION] Lesson {i}: {lesson[:50]}...")

        if mb.data.failed_approaches:
            logs.append(f"[MEMORY] Failed approaches to avoid: {len(mb.data.failed_approaches)}")
            for approach in mb.data.failed_approaches[-2:]:
                logs.append(f"  - {approach[:60]}...")

        logger.info(f"[ITERATION] Memory context loaded: {completed} iterations")
    else:
        context = ""
        logger.info("[ITERATION] First iteration - no context")

    return {
        "iteration_context": context,
        "processing_logs": logs,
    }


async def save_to_memory_node(state: MARBLEState) -> Dict[str, Any]:
    """Save current iteration results to EvolvingMemory."""
    current_iteration = state.get("current_iteration", 1)
    target_model = state.get("target_model", "unknown")

    current_metrics = state.get("iteration_metrics", {})
    docker_success = state.get("docker_test_success", False)
    docker_output = state.get("docker_test_output", "")

    target_component = state.get("target_component", "unknown")
    build_path = get_build_path(current_iteration)
    proposal_path = build_path / "build_debate_outputs" / "implementation_proposal.md"
    weakness_path = build_path / "build_debate_outputs" / "weakness_of_target_model.md"
    other_papers_path = build_path / "build_debate_outputs" / "other_papers.json"
    parsed_component, parsed_description = _parse_implementation_proposal(proposal_path)

    papers_used = _extract_paper_titles(other_papers_path)

    implementation = _extract_proposal_sections(proposal_path)
    weakness = _extract_weakness_summary(weakness_path)

    changes = {
        "component": parsed_component or target_component or "unknown",
        "description": parsed_description or f"Iteration {current_iteration} changes",
        "implementation": implementation,
        "weakness": weakness,
    }

    logger.info(f"[ITERATION] Saving results: iteration {current_iteration}")
    logger.info(f"[ITERATION] Metrics: {current_metrics}")
    logger.info(f"[ITERATION] Docker success: {docker_success}")

    workspace = Path(get_project_root()) / "experiments"
    mb = EvolvingMemory(workspace_path=str(workspace))

    best_metrics = None
    if mb.data.best_performance:
        perf_dict = mb.data.best_performance.model_dump()
        best_metrics = {k: v for k, v in perf_dict.items() if v is not None}
        logger.info(f"[ITERATION] Delta calculation base: best iteration {mb.data.best_iteration}")

    try:
        analyzer = IterationAnalyzerAgent()
        primary_metrics = None
        if target_model:
            model_metric = EvolvingMemory.MODEL_PRIMARY_METRIC.get(target_model.lower())
            if model_metric:
                primary_metrics = [model_metric]
        analysis = analyzer.analyze(
            current_iteration=current_iteration,
            current_metrics=current_metrics,
            current_changes=changes,
            prev_metrics=best_metrics,
            docker_output=docker_output,
            primary_metrics=primary_metrics,
        )
        analysis_dict = analysis.model_dump()
    except Exception as e:
        logger.warning(f"[ITERATION] Analysis failed: {e}")
        analysis_dict = {
            "improved": None,
            "delta": None,
            "reason": f"Analysis failed: {e}",
            "lessons": [],
        }

    if not docker_success and analysis_dict.get("improved") is None:
        analysis_dict["improved"] = False
        analysis_dict["reason"] = f"Docker test failed after max retries. {analysis_dict.get('reason', '')}"
        logger.warning("[ITERATION] Docker failed -> improved = False")

    artifacts = {
        "debate_outputs_path": f"experiments/build_{current_iteration}/build_debate_outputs",
        "src_path": f"experiments/build_{current_iteration}/src",
    }

    mb.add_iteration(
        iteration=current_iteration,
        performance=current_metrics,
        changes=changes,
        analysis=analysis_dict,
        artifacts=artifacts,
        weights=None,
        papers_used=papers_used,
    )

    mb.update_paper_rewards(
        papers_used=papers_used,
        iteration=current_iteration,
        improved=analysis_dict.get("improved")
    )

    improved_str = "improved" if analysis_dict.get("improved") else (
        "degraded" if analysis_dict.get("improved") is False else "baseline"
    )

    return {
        "processing_logs": [
            f"[ITERATION] Iteration {current_iteration} saved",
            f"[ITERATION] Result: {improved_str}",
        ],
    }


def check_continue_node(state: MARBLEState) -> Dict[str, Any]:
    """Check if next iteration should continue."""
    current_iteration = state.get("current_iteration", 1)
    total_iterations = state.get("total_iterations", 1)

    if current_iteration < total_iterations:
        next_iter = current_iteration + 1
        logger.info(f"[ITERATION] Next: {next_iter}/{total_iterations}")

        return {
            "current_iteration": next_iter,
            "docker_test_success": False,
            "docker_test_error": None,
            "docker_test_round": 0,
            "docker_test_output": None,
            "iteration_metrics": {},
            "continue_stage": None,
            "processing_logs": [f"[ITERATION] Iteration {next_iter}/{total_iterations} ready"],
        }
    else:
        logger.info(f"[ITERATION] All iterations complete: {total_iterations}")
        return {
            "current_iteration": total_iterations + 1,
            "processing_logs": [f"[ITERATION] All {total_iterations} iterations complete"],
        }


def route_after_save_to_memory(state: MARBLEState) -> str:
    """Route after save_to_memory."""
    current_iteration = state.get("current_iteration", 1)
    total_iterations = state.get("total_iterations", 1)

    if current_iteration <= total_iterations:
        return "continue_iteration"
    return "end"


def _auto_detect_stage(iteration: int) -> tuple[str, str]:
    """Auto-detect start stage based on existing files."""
    build_path = get_build_path(iteration)

    proposal_path = build_path / "build_debate_outputs" / "implementation_proposal.md"
    if proposal_path.exists():
        return ("development", "implementation_proposal.md found -> start from development")

    if iteration > 1:
        prev_build_path = get_build_path(iteration - 1)
        prev_proposal = prev_build_path / "build_debate_outputs" / "implementation_proposal.md"
        if prev_build_path.exists() and prev_proposal.exists():
            return ("debate", f"build_{iteration - 1} exists -> start from debate (with iteration_critic)")

    return ("debate", "No files found -> start from debate")


def _validate_stage_requirements(stage: str, iteration: int) -> tuple[bool, str, str]:
    """Validate required files for stage."""
    build_path = get_build_path(iteration)

    if stage == "debate" and iteration > 1:
        prev_build_path = get_build_path(iteration - 1)
        if not prev_build_path.exists():
            return (False, "debate", f"build_{iteration - 1} folder not found")
        prev_proposal = prev_build_path / "build_debate_outputs" / "implementation_proposal.md"
        prev_weakness = prev_build_path / "build_debate_outputs" / "weakness_of_target_model.md"
        if not prev_proposal.exists() and not prev_weakness.exists():
            return (False, "debate", f"build_{iteration - 1} has no proposal/weakness files")
        return (True, stage, f"build_{iteration - 1} verified")

    if stage == "development":
        proposal_path = build_path / "build_debate_outputs" / "implementation_proposal.md"
        if not proposal_path.exists():
            return (False, "debate", "implementation_proposal.md not found -> start from debate")
        return (True, stage, "implementation_proposal.md verified")

    elif stage == "docker":
        src_path = build_path / "src"
        if not src_path.exists():
            return (False, "debate", "src/ folder not found -> start from debate")
        py_files = list(src_path.glob("*.py"))
        if not py_files:
            proposal_path = build_path / "build_debate_outputs" / "implementation_proposal.md"
            if proposal_path.exists():
                return (False, "development", "No .py files in src/ -> start from development")
            else:
                return (False, "debate", "No .py files and no proposal -> start from debate")
        return (True, stage, f"{len(py_files)} .py files in src/ verified")

    return (True, "debate", "debate stage always available")


def route_from_inject_memory(state: MARBLEState) -> str:
    """Route based on stage after inject_memory_context.

    Routes:
    - "debate" -> build_debate_subgraph
    - "development" -> build_development_subgraph
    - "docker" -> docker_execution_subgraph
    """
    is_continue_mode = state.get("is_continue_mode", False)
    continue_stage = state.get("continue_stage", "debate")
    current_iteration = state.get("current_iteration", 1)

    if not is_continue_mode:
        logger.info("[ITERATION] Normal mode -> debate_subgraph")
        return "debate"

    if continue_stage is None:
        logger.info("[ITERATION] Continue mode (next iteration) -> debate_subgraph")
        return "debate"

    if continue_stage == "auto":
        detected_stage, reason = _auto_detect_stage(current_iteration)
        logger.info(f"[ITERATION] Auto detection: {reason}")
        logger.info(f"[ITERATION] Continue mode -> {detected_stage}_subgraph (iter {current_iteration})")
        return detected_stage

    if continue_stage in ("development", "docker"):
        valid, fallback_stage, message = _validate_stage_requirements(continue_stage, current_iteration)
        if not valid:
            logger.warning(f"[ITERATION] Stage validation failed: {message}")
            logger.info(f"[ITERATION] Fallback -> {fallback_stage}_subgraph (iter {current_iteration})")
            return fallback_stage
        else:
            logger.info(f"[ITERATION] Stage validated: {message}")
        logger.info(f"[ITERATION] Continue mode -> {continue_stage}_subgraph (iter {current_iteration})")
        return continue_stage

    logger.info(f"[ITERATION] Continue mode -> debate_subgraph (iter {current_iteration})")
    return "debate"
