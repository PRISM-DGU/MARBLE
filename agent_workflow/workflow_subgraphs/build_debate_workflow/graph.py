import asyncio
import json
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage

from configs.config import LANGGRAPH_CONFIG
import yaml

from agent_workflow.logger import logger
from agent_workflow.state import MARBLEState
from agent_workflow.utils import get_project_root
from agent_workflow.evolving_memory import EvolvingMemory

from .agents import (
    ModelResearcherAgent,
    ArticleResearcherAgent,
    CriticAgent,
    ProposalAgent,
    PMCResearcherAgent,
    OpenReviewResearcherAgent,
    PaperAggregatorAgent,
)
from .agents.iteration_critic_agent import IterationCriticAgent
from .agents.embedding_scorer import cleanup_embedder

# Module-level cache
_BUILD_DEBATE_SUBGRAPH = None


# ==============================================================================
# PATH HELPERS - Use PROJECT_ROOT from environment
# ==============================================================================

# Model-specific path and workflow configuration
# This configuration is designed to be extensible for various bioinformatics models.
# Each model should define all required fields for dynamic prompt generation.
MODEL_WORKFLOW_CONFIG = {
    "stagate": {
        # === Path Settings ===
        "source": "docker_images/stagate",
        "workspace": "experiments/build",  # Unified workspace for all models
        "dataset": "datasets/stagate",

        # === Model Info ===
        "target_paper": "stagate.pdf",
        "model_name": "STAGATE",
        "domain": "spatial transcriptomics clustering",
        "domain_description": "analyzing spatial gene expression patterns and identifying tissue domains",

        # === Component Settings ===
        "components": ["encoder", "decoder"],
        "component_templates": {
            "encoder": "components/encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
    },

    "deeptta": {
        # === Path Settings ===
        "source": "docker_images/deeptta",
        "workspace": "experiments/build",
        "dataset": "datasets/shared",

        # === Model Info ===
        "target_paper": "deeptta.pdf",
        "model_name": "DeepTTA",
        "domain": "drug response prediction",
        "domain_description": "predicting drug sensitivity/response using molecular and cell line features",

        # === Component Settings ===
        "components": ["drug_encoder", "cell_encoder", "decoder"],
        "component_templates": {
            "drug_encoder": "components/drug_encoder_other.py",
            "cell_encoder": "components/cell_encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
    },

    "deepdr": {
        # === Path Settings ===
        "source": "docker_images/deepdr",
        "workspace": "experiments/build",
        "dataset": "datasets/shared",

        # === Model Info ===
        "target_paper": "deepdr.pdf",
        "model_name": "DeepDR",
        "domain": "drug response prediction",
        "domain_description": "predicting drug sensitivity/response using graph neural networks with MPG drug encoding",

        # === Component Settings ===
        "components": ["drug_encoder", "cell_encoder", "decoder"],
        "component_templates": {
            "drug_encoder": "components/drug_encoder_other.py",
            "cell_encoder": "components/cell_encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
    },

    "deepst": {
        # === Path Settings ===
        "source": "docker_images/deepst",
        "workspace": "experiments/build",
        "dataset": "datasets/deepst",

        # === Model Info ===
        "target_paper": "deepst.pdf",
        "model_name": "DeepST",
        "domain": "spatial transcriptomics clustering",
        "domain_description": "analyzing spatial gene expression patterns and identifying tissue domains using variational graph autoencoders",

        # === Component Settings ===
        "components": ["encoder", "decoder"],
        "component_templates": {
            "encoder": "components/encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
    },

    "hyperattentiondti": {
        # === Path Settings ===
        "source": "docker_images/hyperattentiondti",
        "workspace": "experiments/build",
        "dataset": "datasets/hyperattentiondti",

        # === Model Info ===
        "target_paper": "hyperattentiondti.pdf",
        "model_name": "HyperAttentionDTI",
        "domain": "drug repurposing",
        "domain_description": "predicting drug-target interactions for drug repurposing using hypergraph attention networks",

        # === Component Settings ===
        "components": ["drug_encoder", "protein_encoder", "decoder"],
        "component_templates": {
            "drug_encoder": "components/drug_encoder_other.py",
            "protein_encoder": "components/protein_encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
    },

    "dlm-dti": {
        # === Path Settings ===
        "source": "docker_images/dlm-dti",
        "workspace": "experiments/build",
        "dataset": "datasets/dlm-dti",

        # === Model Info ===
        "target_paper": "dlm-dti.pdf",
        "model_name": "DLM-DTI",
        "domain": "drug target interaction",
        "domain_description": "predicting drug-target interactions using dual language model encoders for drugs and proteins",

        # === Component Settings ===
        "components": ["drug_encoder", "protein_encoder", "decoder"],
        "component_templates": {
            "drug_encoder": "components/drug_encoder_other.py",
            "protein_encoder": "components/protein_encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
    },
}

def get_components_display(model: str = None) -> str:
    """Get component display string for prompts (e.g., 'encoder / decoder / overall')."""
    config = _get_model_config(model)
    components = config.get("components", ["encoder", "decoder"])
    return " / ".join(components) + " / overall"


def get_prompt_context(model: str = None) -> dict:
    """Get all prompt-related context for a model.

    Returns a dict with all variables needed for dynamic prompt generation:
    - model_name: Display name (e.g., "STAGATE", "DeepTTA")
    - domain: Short domain name (e.g., "spatial transcriptomics clustering")
    - domain_description: Longer description for context
    - components_display: For component selection (e.g., "encoder / decoder / overall")
    - components_list: List of components
    """
    config = _get_model_config(model)

    return {
        "model_name": config.get("model_name", model or "Unknown"),
        "domain": config.get("domain", "machine learning"),
        "domain_description": config.get("domain_description", ""),
        "components_display": get_components_display(model),
        "components_list": config.get("components", []),
    }

# Default fallback (uses deeptta config)
DEFAULT_MODEL = "deeptta"

# Backward compatibility alias
MODEL_PATH_CONFIG = MODEL_WORKFLOW_CONFIG


def _get_model_config(model: str = None) -> dict:
    """Get full workflow configuration for a model."""
    if model and model in MODEL_WORKFLOW_CONFIG:
        return MODEL_WORKFLOW_CONFIG[model]
    return MODEL_WORKFLOW_CONFIG[DEFAULT_MODEL]


def _get_source_path(model: str = None) -> Path:
    """Get source directory path from docker_images based on model."""
    project_root = get_project_root()
    config = _get_model_config(model)
    return Path(project_root) / config["source"]


def _get_workspace_path(model: str = None, iteration: int = 1) -> Path:
    """Get build workspace path in experiments directory based on model and iteration.

    Args:
        model: Model name (e.g., "stagate")
        iteration: Current iteration number (1-indexed)

    Returns:
        Path to experiments/build_{iteration}/
    """
    project_root = get_project_root()
    return Path(project_root) / "experiments" / f"build_{iteration}"


def _get_dataset_path(model: str = None) -> Path:
    """Get dataset path based on model."""
    project_root = get_project_root()
    config = _get_model_config(model)
    return Path(project_root) / config["dataset"]


def _get_debate_outputs_path(model: str = None, iteration: int = 1) -> Path:
    """Get build debate outputs path."""
    return _get_workspace_path(model, iteration) / "build_debate_outputs"


def _ensure_workspace(model: str = None, iteration: int = 1) -> Path:
    """Ensure build debate outputs workspace exists."""
    workspace = _get_debate_outputs_path(model, iteration)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _get_papers_dir(model: str = None, iteration: int = 1) -> Path:
    """Directory for downloaded papers."""
    return _get_debate_outputs_path(model, iteration) / "papers"


def _get_selected_papers_path(model: str = None, iteration: int = 1) -> Path:
    """Path to selected paper metadata JSON."""
    return _get_debate_outputs_path(model, iteration) / "other_papers.json"


def _load_selected_papers(model: str = None, iteration: int = 1) -> list:
    """Load selected papers from JSON file."""
    metadata_path = _get_selected_papers_path(model, iteration)
    if not metadata_path.exists():
        logger.error(f"[ARTICLE_RESEARCHER] Metadata file not found: {metadata_path}")
        return []

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data.get("selected_papers", [])
    except Exception as e:
        logger.error(f"[ARTICLE_RESEARCHER] Failed to parse metadata: {e}")
        return []


# ==============================================================================
# WORKSPACE SETUP HELPERS (Best iteration 기반)
# ==============================================================================

def _get_best_iteration() -> int:
    """memory.json에서 best_iteration 조회. 없으면 0 (baseline) 반환."""
    workspace = Path(get_project_root()) / "experiments"
    mb = EvolvingMemory(workspace_path=str(workspace))
    best_iter = mb.data.best_iteration
    return best_iter if best_iter is not None else 0


def _parse_other_components(config_path: Path) -> Dict[str, str]:
    """config.yaml에서 _other로 변경된 컴포넌트 목록 추출.

    Args:
        config_path: config.yaml 파일 경로

    Returns:
        {component_type: component_file_name} 딕셔너리
        예: {"encoder": "encoder_other", "decoder": "decoder_stagate_gat"}
    """
    if not config_path.exists():
        return {}

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_config = config.get("model", {})

        result = {}
        for key in model_config:
            if isinstance(model_config[key], dict) and "type" in model_config[key]:
                component_type = key  # encoder, decoder 등
                component_file = model_config[key]["type"]  # encoder_other, decoder_stagate_gat 등
                result[component_type] = component_file

        return result
    except Exception as e:
        logger.warning(f"[WORKSPACE] config.yaml 파싱 실패: {e}")
        return {}


def _apply_best_changes_to_workspace(
    target_path: Path,
    best_path: Path,
    target_model: str
) -> List[str]:
    """best iteration의 변경사항을 target 워크스페이스에 적용.

    1. best의 config.yaml 파싱 → _other로 변경된 컴포넌트 확인
    2. best의 *_other.py → target의 *_other.py로 복사 (덮어쓰기)
       - 이렇게 하면 target의 *_other.py에 best의 구현이 들어가고,
       - code expert가 이 파일을 수정하여 새로운 개선을 추가할 수 있음
    3. best의 src/ → target의 src/로 복사
    4. best의 config.yaml → target으로 복사 (encoder.type: encoder_other 유지)

    Args:
        target_path: 새로 생성할 build_t 경로 (이미 build_0에서 복사됨)
        best_path: best iteration의 build 경로
        target_model: 타겟 모델 이름

    Returns:
        적용된 변경사항 로그 리스트
    """
    logs = []

    # 1. best의 config.yaml에서 _other로 변경된 컴포넌트 확인
    best_config_path = best_path / "config.yaml"
    best_components = _parse_other_components(best_config_path)

    for component_type, component_file in best_components.items():
        # _other로 변경된 컴포넌트만 처리
        if "_other" not in component_file:
            continue

        # 2. best의 *_other.py → target의 *_other.py로 복사 (같은 이름)
        best_other_file = best_path / "components" / f"{component_file}.py"
        if not best_other_file.exists():
            logger.warning(f"[WORKSPACE] best의 {component_file}.py 없음")
            continue

        # target의 같은 이름 파일에 덮어쓰기 (build_0의 빈 템플릿 → best의 구현)
        target_other_file = target_path / "components" / f"{component_file}.py"
        shutil.copy2(str(best_other_file), str(target_other_file))
        logs.append(f"[WORKSPACE] {component_type}: {best_other_file.name} → {target_other_file.name}")
        logger.info(f"[WORKSPACE] {component_type} 적용: {best_other_file.name} → {target_other_file.name}")

    # 3. best의 src/ → target의 src/로 복사
    best_src = best_path / "src"
    target_src = target_path / "src"
    if best_src.exists():
        # target_src가 있으면 삭제 후 복사
        if target_src.exists():
            shutil.rmtree(str(target_src))
        shutil.copytree(str(best_src), str(target_src))
        src_files = list(best_src.glob("*.py"))
        logs.append(f"[WORKSPACE] src/ 복사: {len(src_files)}개 파일")
        logger.info(f"[WORKSPACE] src/ 복사: {best_src} → {target_src}")

    # 4. best의 config.yaml을 target에 복사 (best 설정 유지)
    if best_config_path.exists():
        target_config_path = target_path / "config.yaml"
        shutil.copy2(str(best_config_path), str(target_config_path))
        logs.append(f"[WORKSPACE] config.yaml 복사 (best 설정)")
        logger.info(f"[WORKSPACE] config.yaml 복사: {best_config_path} → {target_config_path}")

    return logs


# ==============================================================================
# WORKSPACE SETUP NODE
# ==============================================================================

def setup_build_workspace_node(state: MARBLEState) -> Dict[str, Any]:
    """Setup build workspace by copying source directory to experiments.

    Similar to workspace_creator_agent.py pattern in development_workflow.
    Each iteration has its own workspace: experiments/build_{iteration}/

    Source path logic (변경됨):
    - iter 1: build_0 → build_1 (baseline 템플릿 복사)
    - iter 2+: build_0 기반 + best iteration의 변경사항 적용
    """
    target_model = state.get("target_model")
    current_iteration = state.get("current_iteration", 1)
    target_path = _get_workspace_path(target_model, current_iteration)
    dataset_path = _get_dataset_path(target_model)
    project_root = Path(get_project_root())

    # 항상 build_0 (baseline 템플릿)을 기반으로 함
    baseline_path = project_root / "experiments" / "build_0"

    # build_0이 없으면 docker_images에서 fallback (init_iteration_node가 호출되지 않은 경우)
    if not baseline_path.exists():
        baseline_path = _get_source_path(target_model)
        logger.warning(f"[BUILD_WORKSPACE] build_0 없음, docker_images 사용: {baseline_path}")

    logger.info(f"[BUILD_WORKSPACE] Model: {target_model}, Iteration: {current_iteration}")
    logger.info(f"[BUILD_WORKSPACE] Baseline: {baseline_path}")
    logger.info(f"[BUILD_WORKSPACE] Target: {target_path}")
    logger.info(f"[BUILD_WORKSPACE] Dataset: {dataset_path}")

    # Validate baseline exists
    if not baseline_path.exists():
        logger.error(f"[BUILD_WORKSPACE] Baseline not found: {baseline_path}")
        return {
            "build_workspace_created": False,
            "processing_logs": [f"[BUILD_WORKSPACE] ERROR: Baseline not found: {baseline_path}"]
        }

    # Remove existing workspace if exists (overwrite) - 이미 init_iteration_node에서 처리됨
    if target_path.exists():
        shutil.rmtree(str(target_path))
        logger.info(f"[BUILD_WORKSPACE] Removed existing workspace: {target_path}")

    # Copy baseline to target
    try:
        ignore_patterns = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "build_debate_outputs")
        shutil.copytree(str(baseline_path), str(target_path), ignore=ignore_patterns)
        logger.info(f"[BUILD_WORKSPACE] Created workspace from baseline: {target_path}")

        processing_logs = [
            f"[BUILD_WORKSPACE] Iteration {current_iteration}: baseline → {target_path}",
            f"[BUILD_WORKSPACE] Dataset: {dataset_path}",
        ]

        # iter 2+: best iteration의 변경사항 적용
        if current_iteration > 1:
            best_iter = _get_best_iteration()
            logger.info(f"[BUILD_WORKSPACE] Best iteration: {best_iter}")

            if best_iter > 0:
                # best가 baseline(0)이 아닌 경우에만 변경사항 적용
                best_path = project_root / "experiments" / f"build_{best_iter}"
                if best_path.exists():
                    change_logs = _apply_best_changes_to_workspace(
                        target_path=target_path,
                        best_path=best_path,
                        target_model=target_model
                    )
                    processing_logs.extend(change_logs)
                    processing_logs.append(f"[BUILD_WORKSPACE] Best iteration {best_iter}의 변경사항 적용 완료")
                else:
                    logger.warning(f"[BUILD_WORKSPACE] Best build 폴더 없음: {best_path}")
            else:
                processing_logs.append(f"[BUILD_WORKSPACE] Best = baseline, 변경사항 없음")

        # Ensure debate outputs directory exists
        _ensure_workspace(target_model, current_iteration)

        processing_logs.append(f"[BUILD_WORKSPACE] Ready for build workflow")

        return {
            "build_workspace": str(target_path),
            "build_workspace_created": True,
            "dataset_path": str(dataset_path),
            "processing_logs": processing_logs
        }
    except Exception as e:
        logger.error(f"❌ [BUILD_WORKSPACE] Copy failed: {e}")
        return {
            "build_workspace_created": False,
            "processing_logs": [f"[BUILD_WORKSPACE] ERROR: {str(e)}"]
        }


# ==============================================================================
# NODE FUNCTIONS
# ==============================================================================

async def paper_reader_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Read target model paper and generate summary using ModelResearcherAgent."""
    target_model = state.get("target_model")
    current_iteration = state.get("current_iteration", 1)
    model_config = _get_model_config(target_model)
    build_workspace = _get_workspace_path(target_model, current_iteration)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    # Model-specific PDF from config
    pdf_name = model_config.get("target_paper", f"{target_model}.pdf")
    model_name = model_config.get("model_name", target_model or "DeepTTA")
    domain = model_config.get("domain", "drug response prediction")
    components = model_config.get("components", ["encoder", "decoder"])
    pdf_path = str(build_workspace / pdf_name)
    output_path = str(debate_outputs / f"{target_model or 'deeptta'}_summary.md")

    logger.info("=" * 60)
    logger.info(f"[STEP 1/10] 📄 MODEL_RESEARCHER - Reading target model ({model_name})")
    logger.info(f"  PDF: {pdf_path}")

    agent = ModelResearcherAgent(
        pdf_path=pdf_path,
        output_path=output_path,
        model_config=model_config,  # Pass model config for dynamic prompts
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    # Build dynamic component focus list
    component_focus = "\n".join([f"{i+1}. {comp.replace('_', ' ').title()}" for i, comp in enumerate(components)])

    prompt = f"""Read the {model_name} paper and create a comprehensive summary.

PDF Path: {pdf_path}
Output Path: {output_path}

Focus on:
{component_focus}
{len(components)+1}. Key innovations and any limitations

IMPORTANT: You MUST use the write_file tool to save the summary to: {output_path}
Do NOT just output the summary as text - you MUST call write_file to save it.

Use read_pdf tool to read the paper, then write_file to save the summary.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    # Check if file was actually written, if not extract from response and save
    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[PAPER_READER] Agent did not write file, extracting from response...")

        messages = result.get("messages", [])
        summary_content = None

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                if f"# {model_name}" in content or "## Overview" in content or components[0] in content.lower():
                    summary_content = content
                    break

        if summary_content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(summary_content, encoding='utf-8')
            logger.info(f"[PAPER_READER] Manually saved summary to {output_path}")
        else:
            logger.error("[PAPER_READER] Could not extract summary content from agent response")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[PAPER_READER] Summary saved to {output_path}"],
    }


async def weakness_analysis_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Analyze target model weaknesses using CriticAgent (replaces model_problem_node)."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    model_name = model_config.get("model_name", target_model or "DeepTTA")
    domain = model_config.get("domain", "machine learning")

    model_summary = str(debate_outputs / f"{target_model or 'deeptta'}_summary.md")
    output_path = str(debate_outputs / "weakness_of_target_model.md")

    # Config file path for analysis
    workspace_path = _get_workspace_path(target_model, current_iteration)
    config_path = str(workspace_path / "config.yaml")

    logger.info("=" * 60)
    logger.info(f"[STEP 2/10] 🔍 CRITIC - Analyzing {model_name} weaknesses (including config)")

    # Get iteration context from state
    iteration_context = state.get("iteration_context", "")

    agent = CriticAgent(
        mode="analyze_weakness",
        target_summary_path=model_summary,
        output_path=output_path,
        model_config=model_config,  # Pass model config for dynamic prompts
        iteration_context=iteration_context,  # Pass iteration context for prompt injection
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Analyze the target model ({model_name}) and identify its weaknesses.

Domain: {domain}
Target Model Summary: {model_summary}
Config File: {config_path}
Output Path: {output_path}

IMPORTANT:
- You MUST read both the model summary AND the config file.
- Analyze the ENTIRE model holistically - architecture, training, and all components.
- Identify weaknesses across ALL aspects: encoder, decoder, loss function, optimization, etc.
- For config: ONLY analyze "# Can touch" sections (encoder, decoder, training).
- Do NOT suggest changes to "# Never touch" sections (data, clustering, evaluation, output, logging).
- You MUST use write_file to save weakness_of_target_model.md.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[WEAKNESS_ANALYSIS] Agent did not write file, extracting from response...")
        messages = result.get("messages", [])
        content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Weakness of Target Model" in msg.content or "## Weakness" in msg.content:
                    content = msg.content
                    break
        if content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            logger.info(f"[WEAKNESS_ANALYSIS] Manually saved to {output_path}")
        else:
            logger.error("[WEAKNESS_ANALYSIS] Could not extract weakness content")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[WEAKNESS_ANALYSIS] Saved to {output_path}"],
    }


async def iteration_critic_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Analyze best iteration results and generate new weakness analysis (for iter 2+).

    This node handles:
    1. Weight adjustment logic based on performance
    2. Memory.json updates (consecutive_failures, used_papers, current_weights)
    3. Setting skip_paper_search and should_terminate flags
    4. Generating new weakness_of_target_model.md for current iteration

    Weight Adjustment Logic:
    - Performance improved: Reset consecutive_failures, do new paper search
    - Performance dropped (fail < 3): Skip paper search, use next papers from aggregated_results
    - Performance dropped (fail >= 3): Adjust weights (domain -0.1, arch -0.1, novelty +0.2), reset used_papers
    - Novelty >= 0.8: Terminate (no more relevant papers)

    NOTE: best iteration의 build 폴더를 분석 대상으로 함 (t-1이 아님)
    """
    import json

    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    prev_iteration = current_iteration - 1  # memory.json 분석용 (PHASE 1)

    # Paths
    project_root = get_project_root()
    memory_json_path = str(Path(project_root) / "experiments" / "evolving_memory" / "memory.json")
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    output_path = str(debate_outputs / "weakness_of_target_model.md")

    model_name = model_config.get("model_name", target_model or "Target Model")

    # best iteration 조회 (코드 분석용, PHASE 2)
    best_iteration = _get_best_iteration()
    if best_iteration == 0:
        # best가 baseline인 경우, build_0을 분석 대상으로 사용
        best_build_path = str(Path(project_root) / "experiments" / "build_0")
    else:
        best_build_path = str(Path(project_root) / "experiments" / f"build_{best_iteration}")

    logger.info("=" * 60)
    logger.info(f"[ITERATION_CRITIC] Analyzing for iteration {current_iteration}")
    logger.info(f"  Previous iteration (for metrics): {prev_iteration}")
    logger.info(f"  Best iteration (for code analysis): {best_iteration}")
    logger.info(f"  Memory JSON: {memory_json_path}")
    logger.info(f"  Best Build: {best_build_path}")
    logger.info(f"  Output: {output_path}")

    # =========================================================================
    # PHASE 1: Read memory.json and determine beta adjustment
    # =========================================================================
    should_terminate = False

    try:
        with open(memory_json_path, "r", encoding="utf-8") as f:
            memory_data = json.load(f)

        # Get previous iteration's analysis result
        iterations = memory_data.get("iterations", [])
        if prev_iteration <= len(iterations):
            prev_iter_data = iterations[prev_iteration - 1]  # 0-indexed
            prev_improved_raw = prev_iter_data.get("analysis", {}).get("improved")
            # None을 명시적으로 False로 처리 (baseline 비교 불가 → 실패로 간주)
            # 단, init_baseline에서 baseline을 초기 best로 설정하므로 iter1도 True/False를 가짐
            prev_improved = prev_improved_raw if prev_improved_raw is not None else False
            prev_papers_used = prev_iter_data.get("papers_used", [])
        else:
            prev_improved = False
            prev_papers_used = []

        # Get current memory state
        consecutive_failures = memory_data.get("consecutive_failures", 0)
        used_papers = memory_data.get("used_papers", [])
        # New weight format: w_d, w_a are auto-calculated, only beta is adjustable
        default_weights = {"w_d": 0.9, "w_a": 0.1, "beta": 1.0}
        saved_weights = memory_data.get("current_weights", {})
        # Handle both old and new format, and null values
        if saved_weights and "beta" in saved_weights and saved_weights.get("beta") is not None:
            current_weights = saved_weights
        else:
            current_weights = default_weights.copy()

        logger.info(f"[ITERATION_CRITIC] Previous iteration improved: {prev_improved}")
        logger.info(f"[ITERATION_CRITIC] Consecutive failures: {consecutive_failures}")
        logger.info(f"[ITERATION_CRITIC] Current beta: {current_weights.get('beta', 1.0)}")

        # NOTE: Online paper search removed. Always use EmbeddingScorer with local PDFs.
        # The skip_paper_search flag is no longer used for routing.

        # Add previous papers to used_papers for deduplication
        for paper_title in prev_papers_used:
            if paper_title not in used_papers:
                used_papers.append(paper_title)

        if prev_improved:
            logger.info("[ITERATION_CRITIC] Performance IMPROVED")
        else:
            logger.info(f"[ITERATION_CRITIC] Performance DROPPED - consecutive failures: {consecutive_failures}")

        # NOTE: Beta adjustment is entirely LLM-controlled via paper_aggregator's adjust_beta() tool.
        # The LLM can check consecutive_failures and decide whether to adjust beta.

        logger.info(f"[ITERATION_CRITIC] Used papers count: {len(used_papers)}")

        # Update memory.json
        memory_data["consecutive_failures"] = consecutive_failures
        memory_data["used_papers"] = used_papers
        memory_data["current_weights"] = current_weights

        with open(memory_json_path, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[ITERATION_CRITIC] Memory.json updated")

    except Exception as e:
        logger.error(f"[ITERATION_CRITIC] Error processing memory.json: {e}")
        should_terminate = False

    # =========================================================================
    # PHASE 2: Run agent to generate weakness analysis
    # =========================================================================
    agent = IterationCriticAgent(
        current_iteration=current_iteration,
        best_iteration=best_iteration,
        memory_json_path=memory_json_path,
        best_build_path=best_build_path,
        output_path=output_path,
        model_config=model_config,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Analyze best iteration {best_iteration} results and generate weakness analysis for iteration {current_iteration}.

You are the Iteration Critic for {model_name}. Your task is to:
1. Read memory.json to understand performance changes (baseline vs best iteration {best_iteration})
2. Read the best iteration's weakness_of_target_model.md and implementation_proposal.md
3. Read ALL code in components/ and src/ folders (especially *_other.py files that contain modifications)
4. Read config.yaml to see which component types were applied
5. Analyze what worked and what didn't
6. Generate a NEW weakness_of_target_model.md for iteration {current_iteration}

Memory JSON Path: {memory_json_path}
Best Build Path: {best_build_path}
Output Path: {output_path}

IMPORTANT:
- Compare performance between iteration 0 (baseline) and best iteration {best_iteration}
- Focus on *_other.py files - they contain the actual code changes
- Check config.yaml to see which encoder/decoder types were used
- Be specific about WHY the best approach succeeded
- Provide actionable guidance for the next iteration to improve upon the best
- You MUST use write_file to save the weakness analysis
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    # Fallback: extract from response if file wasn't written
    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[ITERATION_CRITIC] Agent did not write file, extracting from response...")
        messages = result.get("messages", [])
        content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Weakness Analysis" in msg.content or "## Previous Iteration" in msg.content:
                    content = msg.content
                    break
        if content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            logger.info(f"[ITERATION_CRITIC] Manually saved to {output_path}")
        else:
            logger.error("[ITERATION_CRITIC] Could not extract weakness content")

    return {
        "messages": result["messages"],
        "should_terminate": should_terminate,
        "processing_logs": [
            f"[ITERATION_CRITIC] Weakness analysis for iter {current_iteration} saved to {output_path}",
            f"[ITERATION_CRITIC] should_terminate={should_terminate}"
        ],
    }


async def article_researcher_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Search and download papers based on weakness_of_target_model.md, summary.md, and code analysis."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    papers_dir = _get_papers_dir(target_model, current_iteration)
    repos_dir = debate_outputs / "repos"

    # Get model-specific settings
    model_name = model_config.get("model_name", target_model or "DeepTTA")
    domain = model_config.get("domain", "machine learning")

    # Clean up previous papers and repos to avoid stale files
    if papers_dir.exists():
        shutil.rmtree(str(papers_dir))
        logger.info(f"[ARTICLE_RESEARCHER] Cleaned up previous papers: {papers_dir}")
    if repos_dir.exists():
        shutil.rmtree(str(repos_dir))
        logger.info(f"[ARTICLE_RESEARCHER] Cleaned up previous repos: {repos_dir}")
    papers_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Key input files from previous steps
    summary_path = str(debate_outputs / f"{target_model or 'deeptta'}_summary.md")
    weakness_path = str(debate_outputs / "weakness_of_target_model.md")
    output_path = str(_get_selected_papers_path(target_model, current_iteration))

    # Get build workspace path for model code analysis
    build_workspace = _get_workspace_path(target_model, current_iteration)

    logger.info("=" * 60)
    logger.info(f"[STEP 3/10] 🔎 ARTICLE_RESEARCHER - Searching papers via PMC API for {model_name} ({domain})")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"  Weakness: {weakness_path}")
    logger.info(f"  Code: {build_workspace}")

    # Create ArticleResearcherAgent with PMC-based search
    # LLM will generate search keywords dynamically based on model analysis
    agent = ArticleResearcherAgent(
        mode="search_papers",
        workspace_path=str(build_workspace),
        output_path=output_path,
        papers_dir=str(papers_dir),
        start_year=2022,
        keywords=None,  # LLM will generate keywords based on analysis
        model_config=model_config,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    # Prompt for PMC-based paper search - LLM generates keywords from 3 sources
    prompt = f"""Find the top 2 papers for improving {model_name} in the domain of {domain}.

## Your Workflow

### Step 1: Read Analysis Documents
You MUST read ALL THREE sources to understand the model:

1. **Model Summary** (target model paper analysis):
   Path: {summary_path}

2. **Weakness Analysis** (identified areas for improvement):
   Path: {weakness_path}

3. **Model Code** (actual implementation - DO NOT read config.yaml):
   Path: {build_workspace}
   Use `analyze_model_structure` tool to analyze the code structure.

### Step 2: Generate 3 Search Keywords
Based on your analysis of ALL THREE sources above, generate exactly 3 search keywords.
Keywords should target papers that can address the identified weaknesses.
Each keyword should be 2-3 words. 

### Step 3: Search PMC
Use `search_and_filter_papers` with your keywords to collect candidate papers.
- Maximum 10 search rounds
- Stop early if you reach 50+ candidates with GitHub

### Step 4: Score Each Paper (YOUR JUDGMENT)
For each paper, YOU must assign scores:
- relevance_score (0.0 ~ 1.0): How applicable is this paper to {domain} and the identified weaknesses?
- novelty_score (0.0 ~ 1.0): How innovative/recent is the approach?

Final Score = 0.8 × relevance_score + 0.2 × novelty_score (FIXED WEIGHTS)

### Step 5: Check GitHub (REQUIRED)
Only papers WITH GitHub repository can be selected!

### Step 6: Download and Save
Download PDFs for the top 2 papers (by final_score, must have GitHub).
Use `write_file` to save results to: {output_path}

## Output Format
Save a JSON file with this structure:
{{
    "selected_papers": [
        {{
            "pmid": "...",
            "title": "...",
            "abstract": "...",
            "doi": "...",
            "pdf_path": "...",
            "github_urls": ["..."],
            "year": 2024,
            "source": "pmc",
            "relevance_score": 0.85,
            "novelty_score": 0.72,
            "final_score": 0.82,
            "llm_reasoning": "This paper addresses weakness X by..."
        }}
    ],
    "search_metadata": {{
        "relevance_weight": 0.8,
        "novelty_weight": 0.2,
        "total_searched": 20,
        "keywords_used": ["..."]
    }}
}}
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    # Verify output file was created
    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[ARTICLE_RESEARCHER] Metadata file missing, attempting extraction...")
        messages = result.get("messages", [])
        extracted = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                text = msg.content
                # Try to extract from ```json code block first
                import re as regex
                json_block_match = regex.search(r'```json\s*([\s\S]*?)\s*```', text)
                if json_block_match:
                    extracted = json_block_match.group(1).strip()
                    logger.info("[ARTICLE_RESEARCHER] Extracted JSON from code block")
                    break
                # Fallback: find raw JSON object
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    extracted = text[start:end + 1]
                    break
        if extracted:
            try:
                # Try to fix common JSON issues before parsing
                def fix_json(text):
                    import re as regex
                    # Remove trailing commas before } or ]
                    text = regex.sub(r',\s*}', '}', text)
                    text = regex.sub(r',\s*]', ']', text)
                    # Fix single quotes to double quotes (careful with apostrophes)
                    # Only replace quotes that look like JSON keys/values
                    text = regex.sub(r"(?<=[{,\[])\s*'([^']+)'\s*:", r'"\1":', text)
                    text = regex.sub(r":\s*'([^']*)'", r': "\1"', text)
                    return text

                fixed_json = fix_json(extracted)
                json.loads(fixed_json)  # Validate
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(fixed_json, encoding="utf-8")
                logger.info(f"[ARTICLE_RESEARCHER] Manually saved metadata to {output_path}")
            except json.JSONDecodeError as e:
                logger.error(f"[ARTICLE_RESEARCHER] Failed to parse extracted JSON: {e}")
                logger.info("[ARTICLE_RESEARCHER] Will use fallback papers")
            except Exception as e:
                logger.error(f"[ARTICLE_RESEARCHER] Unexpected error: {e}")
        else:
            logger.error("[ARTICLE_RESEARCHER] Could not extract metadata JSON from response")

    # RETRY: If JSON still doesn't exist, ask agent to output JSON again (max 2 retries)
    max_retries = 2
    for retry_attempt in range(max_retries):
        if output_file.exists():
            break

        logger.warning(f"[ARTICLE_RESEARCHER] Retry {retry_attempt + 1}/{max_retries} - Requesting JSON output...")

        retry_prompt = f"""You FAILED to save the JSON file. You MUST use `write_file` tool NOW.

CRITICAL INSTRUCTION:
1. Use the `write_file` tool to save JSON to: {output_path}
2. The JSON must have this structure:

{{
  "selected_papers": [
    {{
      "pmid": "...",
      "title": "...",
      "abstract": "...",
      "doi": "...",
      "pdf_path": "...",
      "github_urls": ["..."],
      "year": 2024,
      "relevance_score": 0.85,
      "final_score": 0.82
    }}
  ]
}}

If you found no papers, save:
{{
  "selected_papers": []
}}

ACTION REQUIRED: Call `write_file` tool with path="{output_path}" and the JSON content.
Also output the JSON in ```json``` block as backup."""

        retry_result = await agent.compiled_agent.ainvoke({
            **state,
            "messages": result.get("messages", []) + [HumanMessage(content=retry_prompt)]
        })

        # Try to extract JSON from retry response
        for msg in reversed(retry_result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                text = msg.content
                import re as regex
                json_block_match = regex.search(r'```json\s*([\s\S]*?)\s*```', text)
                if json_block_match:
                    extracted = json_block_match.group(1).strip()
                else:
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        extracted = text[start:end + 1]
                    else:
                        extracted = None

                if extracted:
                    try:
                        json.loads(extracted)  # Validate
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        output_file.write_text(extracted, encoding="utf-8")
                        logger.info(f"[ARTICLE_RESEARCHER] Retry {retry_attempt + 1} succeeded - saved to {output_path}")
                    except json.JSONDecodeError:
                        logger.error(f"[ARTICLE_RESEARCHER] Retry {retry_attempt + 1} - Invalid JSON")
                break

        # Update result for next retry if needed
        result = retry_result

    # FALLBACK: If JSON still doesn't exist, try to use candidates backup
    if not output_file.exists():
        candidates_backup_path = papers_dir / "_candidates_backup.json"
        if candidates_backup_path.exists():
            logger.info("[ARTICLE_RESEARCHER] Using candidates backup for fallback scoring...")
            try:
                backup_data = json.loads(candidates_backup_path.read_text(encoding="utf-8"))
                candidates = backup_data.get("candidates", [])

                if candidates:
                    # Sort by GitHub availability (papers with more GitHub URLs first)
                    # Then download top 2 with working PDFs
                    import requests
                    import re as regex

                    downloaded_papers = []
                    for candidate in candidates[:10]:  # Check top 10
                        if len(downloaded_papers) >= 2:
                            break

                        title = candidate.get("title", "")
                        pmid = candidate.get("pmid", "")
                        doi = candidate.get("doi", "")
                        github_urls = candidate.get("github_urls", [])

                        if not github_urls:
                            continue

                        # Try to download PDF
                        pdf_path = None

                        # Try Europe PMC
                        if pmid:
                            try:
                                pmc_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
                                resp = requests.get(pmc_api, timeout=10)
                                if resp.status_code == 200:
                                    records = resp.json().get("records", [])
                                    if records and records[0].get("pmcid"):
                                        pmcid = records[0]["pmcid"]
                                        pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                                        resp = requests.get(pdf_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                                        if resp.status_code == 200 and resp.content[:4] == b'%PDF':
                                            safe_title = regex.sub(r'[^\w\s\-]', '', title)
                                            safe_title = regex.sub(r'\s+', '_', safe_title.strip())[:80]
                                            filepath = papers_dir / f"{safe_title}.pdf"
                                            with open(filepath, 'wb') as f:
                                                f.write(resp.content)
                                            pdf_path = str(filepath)
                            except Exception:
                                pass

                        # Try Semantic Scholar
                        if not pdf_path and doi:
                            try:
                                ss_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
                                resp = requests.get(ss_url, timeout=10)
                                if resp.status_code == 200:
                                    oa_pdf = resp.json().get("openAccessPdf", {})
                                    if oa_pdf and oa_pdf.get("url"):
                                        pdf_url = oa_pdf["url"]
                                        resp = requests.get(pdf_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                                        if resp.status_code == 200 and resp.content[:4] == b'%PDF':
                                            safe_title = regex.sub(r'[^\w\s\-]', '', title)
                                            safe_title = regex.sub(r'\s+', '_', safe_title.strip())[:80]
                                            filepath = papers_dir / f"{safe_title}.pdf"
                                            with open(filepath, 'wb') as f:
                                                f.write(resp.content)
                                            pdf_path = str(filepath)
                            except Exception:
                                pass

                        if pdf_path:
                            paper = {**candidate, "pdf_path": pdf_path}
                            downloaded_papers.append(paper)
                            logger.info(f"[ARTICLE_RESEARCHER] Fallback downloaded: {title[:50]}...")

                    if downloaded_papers:
                        output_data = {
                            "selected_papers": downloaded_papers,
                            "search_metadata": backup_data.get("stats", {})
                        }
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, indent=4, ensure_ascii=False)
                        logger.info(f"[ARTICLE_RESEARCHER] ✅ Fallback created JSON with {len(downloaded_papers)} papers")

            except Exception as e:
                logger.error(f"[ARTICLE_RESEARCHER] Fallback from backup failed: {e}")

    # Fallback: Download PDFs if pdf_path is empty
    if output_file.exists():
        try:
            import requests
            import re as regex
            data = json.loads(output_file.read_text(encoding="utf-8"))
            papers = data.get("selected_papers", [])
            updated = False

            for paper in papers:
                if not paper.get("pdf_path"):
                    logger.info(f"[ARTICLE_RESEARCHER] Fallback download for: {paper.get('title', 'Unknown')[:50]}")
                    doi = paper.get("doi", "")
                    pmid = paper.get("pmid", "")
                    title = paper.get("title", "")
                    pdf_url = None

                    # Try Europe PMC first (NCBI PMC는 403 차단됨)
                    if pmid:
                        try:
                            pmc_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
                            resp = requests.get(pmc_api, timeout=10)
                            if resp.status_code == 200:
                                records = resp.json().get("records", [])
                                if records and records[0].get("pmcid"):
                                    pmcid = records[0]["pmcid"]
                                    # Europe PMC 사용 (더 안정적)
                                    pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                        except Exception:
                            pass

                    # Try Semantic Scholar (Unpaywall은 422 에러)
                    if doi and not pdf_url:
                        try:
                            ss_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
                            resp = requests.get(ss_url, timeout=10)
                            if resp.status_code == 200:
                                oa_pdf = resp.json().get("openAccessPdf", {})
                                if oa_pdf and oa_pdf.get("url"):
                                    pdf_url = oa_pdf["url"]
                        except Exception:
                            pass

                    # Download if URL found
                    if pdf_url:
                        try:
                            resp = requests.get(pdf_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                            if resp.status_code == 200 and len(resp.content) > 1000:
                                if resp.content[:4] == b'%PDF' or 'application/pdf' in resp.headers.get('Content-Type', ''):
                                    safe_title = regex.sub(r'[^\w\s\-]', '', title)
                                    safe_title = regex.sub(r'\s+', '_', safe_title.strip())[:80]
                                    filepath = os.path.join(str(papers_dir), f"{safe_title}.pdf")
                                    with open(filepath, 'wb') as f:
                                        f.write(resp.content)
                                    paper["pdf_path"] = filepath
                                    updated = True
                                    logger.info(f"[ARTICLE_RESEARCHER] Fallback download success: {filepath}")
                        except Exception as e:
                            logger.warning(f"[ARTICLE_RESEARCHER] Fallback download failed: {e}")

            if updated:
                output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info("[ARTICLE_RESEARCHER] Updated JSON with downloaded PDF paths")

        except Exception as e:
            logger.error(f"[ARTICLE_RESEARCHER] Fallback download error: {e}")

    # ========================================
    # AUTOMATIC RETRY: Check if we have enough valid papers (PDF + Git)
    # ========================================
    retry_needed = False
    valid_count = 0

    if output_file.exists():
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
            papers = data.get("selected_papers", [])
            # Count papers with valid PDF + cloned_repo_path
            for paper in papers:
                pdf_path = paper.get("pdf_path", "")
                repo_path = paper.get("cloned_repo_path", "")
                if pdf_path and Path(pdf_path).exists() and repo_path and Path(repo_path).exists():
                    valid_count += 1

            if valid_count < 2:
                retry_needed = True
                logger.warning(f"[ARTICLE_RESEARCHER] ⚠️ Only {valid_count} valid paper(s) (PDF+Git), need at least 2!")
        except Exception as e:
            logger.error(f"[ARTICLE_RESEARCHER] Error checking valid papers: {e}")
            retry_needed = True
    else:
        retry_needed = True
        logger.warning("[ARTICLE_RESEARCHER] ⚠️ No output file found!")

    # Retry with broader keywords if needed (max 1 retry)
    if retry_needed and not state.get("_article_researcher_retried", False):
        logger.info("=" * 60)
        logger.info("[ARTICLE_RESEARCHER] 🔄 AUTOMATIC RETRY - Searching with BROADER keywords")
        logger.info("=" * 60)

        # Mark that we've retried to prevent infinite loop
        state["_article_researcher_retried"] = True

        # Clean up papers and repos directories for fresh search
        if papers_dir.exists():
            shutil.rmtree(str(papers_dir))
        papers_dir.mkdir(parents=True, exist_ok=True)
        if repos_dir.exists():
            shutil.rmtree(str(repos_dir))
        repos_dir.mkdir(parents=True, exist_ok=True)

        # Create new agent instance for retry
        retry_agent = ArticleResearcherAgent(
            mode="search_papers",
            workspace_path=str(build_workspace),
            output_path=output_path,
            papers_dir=str(papers_dir),
            start_year=2021,  # Expand time range on retry
            model_config=model_config,
            checkpointer=checkpointer
        )
        retry_agent.initialize_agent()

        retry_prompt = f"""⚠️ RETRY MODE: Previous search failed to find enough papers!

We need at least 2 papers with downloadable PDFs + GitHub for {model_name} ({domain}).

## CRITICAL INSTRUCTIONS FOR RETRY

1. Use BROADER, SIMPLER keywords (2-4 words only)
   - Instead of "drug response prediction graph neural network attention mechanism",
     try "drug prediction neural network" or "drug response deep learning"

2. Focus on WELL-KNOWN papers that are likely to have PDFs available

3. You have 5 NEW search rounds - use them wisely!

4. MANDATORY: Download PDFs and save results to: {output_path}

Workspace: {build_workspace}

Start searching NOW with simpler keywords!
"""

        retry_result = await retry_agent.compiled_agent.ainvoke({
            **state,
            "messages": [HumanMessage(content=retry_prompt)]
        })

        # Check retry result
        if output_file.exists():
            try:
                data = json.loads(output_file.read_text(encoding="utf-8"))
                papers = data.get("selected_papers", [])
                valid_count = sum(
                    1 for p in papers
                    if p.get("pdf_path") and Path(p["pdf_path"]).exists()
                    and p.get("cloned_repo_path") and Path(p["cloned_repo_path"]).exists()
                )
                logger.info(f"[ARTICLE_RESEARCHER] ✅ Retry completed - {valid_count} valid paper(s) (PDF+Git)")
            except Exception:
                pass

        result = retry_result

    return {
        "messages": result["messages"],
        "processing_logs": [f"[ARTICLE_RESEARCHER] PMC search completed - Metadata saved to {output_path}"],
    }


# ==============================================================================
# PARALLEL PAPER SEARCH NODES (PMC + OpenReview)
# ==============================================================================

async def pmc_researcher_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Search PMC/PubMed with 6 LLM-generated keywords (3 domain + 3 weakness)."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    papers_dir = _get_papers_dir(target_model, current_iteration) / "pmc"

    # Clean up previous PMC results
    if papers_dir.exists():
        shutil.rmtree(str(papers_dir))
    papers_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(papers_dir / "pmc_candidates.json")
    build_workspace = _get_workspace_path(target_model, current_iteration)

    logger.info("=" * 60)
    logger.info(f"[PMC_RESEARCHER] 🔎 Searching PMC (LLM generates all keywords)")
    logger.info("=" * 60)

    agent = PMCResearcherAgent(
        workspace_path=str(build_workspace),
        output_path=output_path,
        papers_dir=str(papers_dir),
        start_year=2022,
        model_config=model_config,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    model_name = model_config.get('model_name', 'the model')
    domain = model_config.get('domain', 'machine learning')
    domain_description = model_config.get('domain_description', '')
    prompt = f"""Search PMC for papers to improve {model_name}.

## Domain Context
- Domain: {domain}
- Description: {domain_description}

## Your Task:
1. Call analyze_model_structure to read code, summary, and weakness files
2. Generate 6 keywords:
   - 3 domain keywords: Based on the domain context above
   - 3 weakness keywords: Based on model weakness analysis
3. Call search_and_filter_papers with all 6 keywords
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    logger.info(f"[PMC_RESEARCHER] ✅ Complete - Results saved to {output_path}")

    return {
        "messages": result["messages"],
        "pmc_results_path": output_path,
        "processing_logs": [f"[PMC_RESEARCHER] Completed - saved to {output_path}"],
    }


async def openreview_researcher_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Search OpenReview (ICLR, NeurIPS, ICML) with 6 LLM-generated keywords (3 domain + 3 weakness)."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    papers_dir = _get_papers_dir(target_model, current_iteration) / "openreview"

    # Clean up previous OpenReview results
    if papers_dir.exists():
        shutil.rmtree(str(papers_dir))
    papers_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(papers_dir / "openreview_candidates.json")
    build_workspace = _get_workspace_path(target_model, current_iteration)

    logger.info("=" * 60)
    logger.info(f"[OPENREVIEW_RESEARCHER] 🔎 Searching OpenReview (LLM generates all keywords)")
    logger.info("=" * 60)

    agent = OpenReviewResearcherAgent(
        workspace_path=str(build_workspace),
        output_path=output_path,
        papers_dir=str(papers_dir),
        model_config=model_config,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    model_name = model_config.get('model_name', 'the model')
    domain = model_config.get('domain', 'machine learning')
    domain_description = model_config.get('domain_description', '')
    prompt = f"""Search OpenReview for ML papers to improve {model_name}.

## Domain Context
- Domain: {domain}
- Description: {domain_description}

## Your Task:
1. Read {model_name.lower()}_summary.md and weakness_of_target_model.md
2. Generate 6 keywords:
   - 3 domain keywords: Based on the domain context above
   - 3 weakness keywords: Based on model weakness analysis (use ML architecture terms)
3. Call search_openreview_papers with all 6 keywords
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    logger.info(f"[OPENREVIEW_RESEARCHER] ✅ Complete - Results saved to {output_path}")

    return {
        "messages": result["messages"],
        "openreview_results_path": output_path,
        "processing_logs": [f"[OPENREVIEW_RESEARCHER] Completed - saved to {output_path}"],
    }


async def paper_aggregator_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Score papers from local PDFs using EmbeddingScorer, select Top-5, clone repos."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    total_iterations = state.get("total_iterations", 1)
    reward_patience = state.get("reward_patience", 10)
    reward_weight = state.get("reward_weight", 0.1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    papers_dir = _get_papers_dir(target_model, current_iteration)

    # Output paths
    output_path = str(papers_dir / "aggregated_results.json")
    repos_dir = str(papers_dir / "repos")

    # Read used_papers from memory.json
    project_root = get_project_root()
    memory_json_path = str(Path(project_root) / "experiments" / "evolving_memory" / "memory.json")
    used_papers = []

    try:
        if Path(memory_json_path).exists():
            with open(memory_json_path, "r", encoding="utf-8") as f:
                memory_data = json.load(f)
            # Get used_papers for deduplication
            used_papers = memory_data.get("used_papers", []) or []
            if used_papers:
                logger.info(f"[PAPER_AGGREGATOR] Excluding {len(used_papers)} previously used papers")
    except Exception as e:
        logger.warning(f"[PAPER_AGGREGATOR] Could not read memory.json: {e}")

    logger.info("=" * 60)
    logger.info(f"[PAPER_AGGREGATOR] 📊 Scoring papers using EmbeddingScorer")
    logger.info(f"  Target model: {target_model}")
    logger.info(f"  Iteration: {current_iteration}/{total_iterations}")
    logger.info(f"  Build dir: {debate_outputs}")
    logger.info(f"  Repos dir: {repos_dir}")
    logger.info("=" * 60)

    # Use new EmbeddingScorer-based agent
    # NOTE: build_dir should be build_N, not build_N/build_debate_outputs
    build_workspace = _get_workspace_path(target_model, current_iteration)
    agent = PaperAggregatorAgent(
        target_model=target_model,
        current_iteration=current_iteration,
        total_iterations=total_iterations,
        build_dir=str(build_workspace),  # build_N, not build_N/build_debate_outputs
        output_path=output_path,
        repos_dir=repos_dir,
        beta=1.0,  # Default, can be adjusted by agent
        model_config=model_config,
        used_papers=used_papers,
        checkpointer=checkpointer,
        reward_patience=reward_patience,
        reward_weight=reward_weight,
    )
    agent.initialize_agent()

    # Use the agent's internal prompt (get_prompt()) which has the stratified workflow
    # The prompt is already defined in PaperAggregatorAgent.get_prompt()
    # We just need a simple trigger message
    prompt = """Execute the stratified paper selection workflow.

Follow the 4 steps defined in your system prompt:
1. score_papers_stratified()
2. generate_stratified_summary()
3. Read all 3 summary files, then select_stratified_papers()
4. clone_selected_repos()

Start now with step 1.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    # Create compatible output for downstream nodes
    # Read stratified scoring results and convert to other_papers.json format
    stratified_path = build_workspace / "embeddings" / "stratified_scoring.json"
    selected_papers_path = _get_selected_papers_path(target_model, current_iteration)

    if stratified_path.exists():
        try:
            with open(stratified_path, 'r', encoding='utf-8') as f:
                agg_data = json.load(f)

            # Get final selected papers (up to 5)
            final_selected = agg_data.get("final_selected", [])[:5]

            # Convert to other_papers.json format
            output_data = {
                "selected_papers": final_selected,
                "search_metadata": {
                    "sources": ["stratified_scoring"],
                    "total_papers": agg_data.get("metadata", {}).get("total_papers", 0),
                    "weight_configs": agg_data.get("metadata", {}).get("weight_configs", [])
                }
            }

            selected_papers_path.parent.mkdir(parents=True, exist_ok=True)
            with open(selected_papers_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[PAPER_AGGREGATOR] ✅ Saved {len(final_selected)} papers to {selected_papers_path}")

            # Update last_paper_search_iteration in memory.json
            # This tracks which iteration has aggregated_results.json for paper_selector
            try:
                if Path(memory_json_path).exists():
                    with open(memory_json_path, "r", encoding="utf-8") as f:
                        memory_data = json.load(f)
                    memory_data["last_paper_search_iteration"] = current_iteration
                    with open(memory_json_path, "w", encoding="utf-8") as f:
                        json.dump(memory_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"[PAPER_AGGREGATOR] Updated last_paper_search_iteration = {current_iteration}")
            except Exception as mem_e:
                logger.warning(f"[PAPER_AGGREGATOR] Could not update last_paper_search_iteration: {mem_e}")

        except Exception as e:
            logger.error(f"[PAPER_AGGREGATOR] Error saving selected papers: {e}")
    else:
        # Fallback: try legacy output_path (aggregated_results.json)
        legacy_path = Path(output_path)
        if legacy_path.exists():
            logger.warning(f"[PAPER_AGGREGATOR] stratified_scoring.json not found, trying legacy {legacy_path}")
            try:
                with open(legacy_path, 'r', encoding='utf-8') as f:
                    agg_data = json.load(f)
                final_selected = agg_data.get("final_selected", agg_data.get("top_20", []))[:5]
                output_data = {
                    "selected_papers": final_selected,
                    "search_metadata": {
                        "sources": ["legacy"],
                        "total_papers": agg_data.get("metadata", {}).get("total_papers", 0),
                    }
                }
                selected_papers_path.parent.mkdir(parents=True, exist_ok=True)
                with open(selected_papers_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                logger.info(f"[PAPER_AGGREGATOR] ✅ (Legacy) Saved {len(final_selected)} papers")
            except Exception as e:
                logger.error(f"[PAPER_AGGREGATOR] Legacy fallback failed: {e}")
        else:
            logger.error(f"[PAPER_AGGREGATOR] No scoring results found at {stratified_path} or {legacy_path}")

    # GPU 메모리 해제 - embedding scoring 완료 후 FlagEmbedding 모델 정리
    cleanup_embedder()

    return {
        "messages": result["messages"],
        "processing_logs": [f"[PAPER_AGGREGATOR] Aggregation complete - saved to {stratified_path}"],
    }


async def paper_selector_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Select next papers from last_paper_search_iteration's aggregated_results when paper search is skipped.

    This node is used when:
    - Previous iteration failed (performance dropped)
    - consecutive_failures < 3
    - skip_paper_search = True

    It reads the aggregated_results.json from last_paper_search_iteration and selects
    the next 5 unused papers (not in memory.json's used_papers list).

    Fallback: If aggregated_results.json doesn't exist, sets fallback_to_paper_search=True
    to trigger new paper search via routing logic.
    """
    target_model = state.get("target_model")
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    papers_dir = _get_papers_dir(target_model, current_iteration)

    project_root = get_project_root()
    memory_json_path = str(Path(project_root) / "experiments" / "evolving_memory" / "memory.json")

    logger.info("=" * 60)
    logger.info(f"[PAPER_SELECTOR] 📄 Selecting next papers (skip_paper_search mode)")
    logger.info("=" * 60)

    try:
        # Read memory.json to get last_paper_search_iteration and used_papers
        with open(memory_json_path, "r", encoding="utf-8") as f:
            memory_data = json.load(f)

        # Use last_paper_search_iteration instead of best_iteration
        # This tracks where aggregated_results.json actually exists
        source_iteration = memory_data.get("last_paper_search_iteration")
        if source_iteration is None:
            # Fallback: no paper search done yet, trigger new search
            logger.warning("[PAPER_SELECTOR] No last_paper_search_iteration found - triggering fallback")
            return {
                "fallback_to_paper_search": True,
                "processing_logs": ["[PAPER_SELECTOR] No previous paper search found - fallback to new search"],
            }

        used_papers = set(memory_data.get("used_papers") or [])

        logger.info(f"[PAPER_SELECTOR] Source iteration (last paper search): {source_iteration}")
        logger.info(f"[PAPER_SELECTOR] Used papers: {len(used_papers)}")

        # Read source iteration's aggregated_results.json
        source_aggregated_path = Path(project_root) / "experiments" / f"build_{source_iteration}" / "build_debate_outputs" / "papers" / "aggregated_results.json"

        if not source_aggregated_path.exists():
            # Fallback: file doesn't exist, trigger new paper search
            logger.warning(f"[PAPER_SELECTOR] Aggregated results not found: {source_aggregated_path}")
            logger.warning("[PAPER_SELECTOR] Triggering fallback to new paper search")
            return {
                "fallback_to_paper_search": True,
                "processing_logs": [f"[PAPER_SELECTOR] {source_aggregated_path} not found - fallback to new search"],
            }

        with open(source_aggregated_path, "r", encoding="utf-8") as f:
            agg_data = json.load(f)

        # Get top_20 papers
        top_papers = agg_data.get("top_20", agg_data.get("top_10", []))

        # Select next 5 unused papers
        selected_papers = []
        for paper in top_papers:
            title = paper.get("title", "")
            if title not in used_papers:
                selected_papers.append(paper)
                if len(selected_papers) >= 5:
                    break

        if len(selected_papers) < 5:
            logger.warning(f"[PAPER_SELECTOR] Only {len(selected_papers)} unused papers available")

        # Setup repos directory
        repos_dir = str(papers_dir / "repos")
        os.makedirs(repos_dir, exist_ok=True)

        # Try to clone repos for selected papers
        valid_papers = _select_papers_with_fallback(selected_papers, repos_dir, count=5)

        # If not enough valid papers from selection, try more from top_papers
        if len(valid_papers) < 5:
            remaining = [p for p in top_papers if p.get("title", "") not in used_papers and p not in selected_papers]
            more_papers = _select_papers_with_fallback(remaining, repos_dir, count=5 - len(valid_papers))
            valid_papers.extend(more_papers)

        # Save to other_papers.json (same format as paper_aggregator)
        selected_papers_path = _get_selected_papers_path(target_model, current_iteration)
        output_data = {
            "selected_papers": valid_papers,
            "search_metadata": {
                "sources": ["pmc", "openreview"],
                "total_papers": len(top_papers),
                "weights": agg_data.get("metadata", {}).get("weights", {}),
                "selection_mode": "paper_selector",
                "source_iteration": source_iteration  # Changed from best_iteration
            }
        }

        selected_papers_path.parent.mkdir(parents=True, exist_ok=True)
        with open(selected_papers_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[PAPER_SELECTOR] ✅ Selected {len(valid_papers)} papers")
        for i, p in enumerate(valid_papers, 1):
            logger.info(f"  {i}. {p.get('title', 'Unknown')[:60]}...")

        return {
            "fallback_to_paper_search": False,  # Explicitly mark success
            "processing_logs": [
                f"[PAPER_SELECTOR] Selected {len(valid_papers)} papers from build_{source_iteration}",
                f"[PAPER_SELECTOR] Saved to {selected_papers_path}"
            ],
        }

    except Exception as e:
        logger.error(f"[PAPER_SELECTOR] Error: {e}")
        return {
            "processing_logs": [f"[PAPER_SELECTOR] ERROR: {e}"],
        }


def _try_clone_single(github_url: str, repos_dir: str) -> Optional[str]:
    """Try to clone a single GitHub repo. Returns path or None."""
    import subprocess
    import shutil

    if not github_url:
        return None

    # Normalize URL
    if not github_url.startswith('http'):
        github_url = f'https://{github_url}'
    github_url = github_url.rstrip('/').rstrip('.git')

    os.makedirs(repos_dir, exist_ok=True)
    repo_name = github_url.split('/')[-1]
    clone_path = os.path.join(repos_dir, repo_name)

    # Check if already cloned
    if os.path.exists(clone_path):
        git_dir = os.path.join(clone_path, ".git")
        has_files = any(f for f in os.listdir(clone_path) if f != ".git") if os.path.isdir(clone_path) else False
        if os.path.isdir(git_dir) and has_files:
            logger.info(f"[CLONE] Already exists: {clone_path}")
            return clone_path
        else:
            shutil.rmtree(clone_path, ignore_errors=True)

    # Clone (disable interactive prompts)
    clone_url = f"{github_url}.git"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"  # Disable git credential prompts
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, clone_path],
            capture_output=True, text=True, timeout=120, env=env
        )
        if result.returncode != 0:
            logger.warning(f"[CLONE] Failed: {github_url} - {result.stderr[:100]}")
            return None

        logger.info(f"[CLONE] ✅ Success: {clone_path}")
        return clone_path
    except Exception as e:
        logger.warning(f"[CLONE] Exception: {e}")
        return None


def _select_papers_with_fallback(selected_papers: List[Dict], repos_dir: str, count: int = 2) -> List[Dict]:
    """Select papers ensuring both PDF and GitHub clone work. Try next rank if either fails."""
    valid_papers = []

    logger.info(f"[SELECT] Finding {count} papers with valid PDF + GitHub from {len(selected_papers)} candidates")

    for rank, paper in enumerate(selected_papers, 1):
        if len(valid_papers) >= count:
            break

        pdf_path = paper.get("pdf_path", "")
        github_urls = paper.get("github_urls", [])
        title = paper.get("title", "Unknown")[:50]

        # Check PDF exists
        if not pdf_path or not Path(pdf_path).exists():
            logger.info(f"[SELECT] Rank {rank}: ❌ PDF missing - {title}...")
            continue

        # Try clone GitHub (only if exactly 1 URL - multiple URLs likely means data/comparison repos)
        cloned_path = None
        if len(github_urls) == 1:
            cloned_path = _try_clone_single(github_urls[0], repos_dir)
        elif len(github_urls) > 1:
            logger.info(f"[SELECT] Rank {rank}: ⏭️ Skipping - {len(github_urls)} GitHub URLs (ambiguous) - {title}...")
            continue

        if not cloned_path:
            logger.info(f"[SELECT] Rank {rank}: ❌ GitHub clone failed - {title}...")
            continue

        # Both PDF and GitHub valid!
        paper_with_repo = {**paper, "cloned_repo_path": cloned_path}
        valid_papers.append(paper_with_repo)
        logger.info(f"[SELECT] Rank {rank}: ✅ Valid ({len(valid_papers)}/{count}) - {title}...")

    if len(valid_papers) < count:
        logger.warning(f"[SELECT] ⚠️ Only found {len(valid_papers)} valid papers (target: {count})")

    return valid_papers


async def other_paper_reader_1_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Read first selected paper and generate summary."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    repos_dir = str(debate_outputs / "repos")

    # Load papers from other_papers.json (already has pdf_path + cloned_repo_path)
    selected = _load_selected_papers(target_model, current_iteration)

    if len(selected) < 1:
        logger.error("[OTHER_PAPER_READER_1] No selected papers found")
        return {
            "processing_logs": ["[OTHER_PAPER_READER_1] ERROR: No selected papers"]
        }

    # Use first paper directly (already validated during download)
    paper = selected[0]
    pdf_path = paper.get("pdf_path")
    github_urls = paper.get("github_urls", [])
    paper_title = paper.get("title", "Unknown")
    cloned_repo_path = paper.get("cloned_repo_path")
    output_path = str(debate_outputs / "other_paper_1_summary.md")

    # Save papers for other_paper_reader_2 to use
    state["_valid_papers"] = selected

    logger.info(f"[OTHER_PAPER_READER_1] Using: {paper_title[:50]}...")
    logger.info(f"[OTHER_PAPER_READER_1] PDF: {pdf_path}")
    logger.info(f"[OTHER_PAPER_READER_1] Repo: {cloned_repo_path}")

    agent = ArticleResearcherAgent(
        mode="paper_summary",
        paper_name="other_paper_1",
        pdf_path=pdf_path,
        output_path=output_path,
        repos_dir=repos_dir,
        model_config=model_config,  # Pass model config for dynamic prompts
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Read the selected paper and write a focused summary WITH GitHub code.

Paper Title: {paper_title}
PDF Path: {pdf_path}
Output Path: {output_path}

## CRITICAL: GitHub Repository Available
GitHub URLs: {github_urls}
Repos Directory: {repos_dir}

You MUST:
1. Use clone_github_repo to clone: {github_urls[0] if github_urls else 'N/A'}
   - target_dir: {repos_dir}
2. Use list_repo_structure to find model files (look for: model.py, encoder.py, layers.py, conv.py)
3. Use read_github_file to understand the code structure and imports
4. List all FILE PATHS related to encoder/model in your summary (NOT the code itself)
5. Trace import dependencies and list those file paths too

DO NOT copy code into summary - only list file paths!

CRITICAL: Use list_repo_structure FIRST to see actual folder structure.
Then use read_github_file to verify the paths and find class names.

Look for main model/encoder classes in these common locations:
- benchmarks/model*.py
- models/model.py
- src/model.py
- networks/model.py

Example output format:
```
File: [ACTUAL VERIFIED PATH]
  - Classes: [CLASS NAME] (line XX)
  - Imports from: [DEPENDENCY PATH]
```

Focus on:
1. Core architecture and design (from paper)
2. What problems it addresses
3. How it could improve {model_config.get("model_name", "the target model")} (component-level)
4. VERIFIED file paths from GitHub (use list_repo_structure!)

IMPORTANT: You MUST use write_file to save the summary.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[OTHER_PAPER_READER_1] Agent did not write file, extracting from response...")
        messages = result.get("messages", [])
        summary_content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Reference Paper" in msg.content or "## Overview" in msg.content or "Architecture" in msg.content:
                    summary_content = msg.content
                    break
        if summary_content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(summary_content, encoding="utf-8")
            logger.info(f"[OTHER_PAPER_READER_1] Manually saved summary to {output_path}")
        else:
            logger.error("[OTHER_PAPER_READER_1] Could not extract summary content from agent response")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[OTHER_PAPER_READER_1] Summary saved to {output_path}"],
        "_valid_papers": selected,  # Pass to other_paper_reader_2
    }


async def other_paper_reader_2_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Read second selected paper and generate summary."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    repos_dir = str(debate_outputs / "repos")

    # Use papers from other_paper_reader_1 or reload from JSON
    valid_papers = state.get("_valid_papers", [])

    if len(valid_papers) < 2:
        # Fallback: reload from other_papers.json
        valid_papers = _load_selected_papers(target_model, current_iteration)

    if len(valid_papers) < 2:
        logger.error("[OTHER_PAPER_READER_2] No second valid paper found")
        return {
            "processing_logs": ["[OTHER_PAPER_READER_2] ERROR: No second valid paper"]
        }

    # Use second valid paper
    paper = valid_papers[1]
    pdf_path = paper.get("pdf_path")
    github_urls = paper.get("github_urls", [])
    paper_title = paper.get("title", "Unknown")
    cloned_repo_path = paper.get("cloned_repo_path")
    output_path = str(debate_outputs / "other_paper_2_summary.md")

    logger.info(f"[OTHER_PAPER_READER_2] Using: {paper_title[:50]}...")
    logger.info(f"[OTHER_PAPER_READER_2] PDF: {pdf_path}")
    logger.info(f"[OTHER_PAPER_READER_2] Repo: {cloned_repo_path}")

    agent = ArticleResearcherAgent(
        mode="paper_summary",
        paper_name="other_paper_2",
        pdf_path=pdf_path,
        output_path=output_path,
        repos_dir=repos_dir,
        model_config=model_config,  # Pass model config for dynamic prompts
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Read the selected paper and write a focused summary WITH GitHub code.

Paper Title: {paper_title}
PDF Path: {pdf_path}
Output Path: {output_path}

## CRITICAL: GitHub Repository Available
GitHub URLs: {github_urls}
Repos Directory: {repos_dir}

You MUST:
1. Use clone_github_repo to clone: {github_urls[0] if github_urls else 'N/A'}
   - target_dir: {repos_dir}
2. Use list_repo_structure to find model files (look for: model.py, encoder.py, layers.py, conv.py)
3. Use read_github_file to understand the code structure and imports
4. List all FILE PATHS related to encoder/model in your summary (NOT the code itself)
5. Trace import dependencies and list those file paths too

DO NOT copy code into summary - only list file paths!

CRITICAL: Use list_repo_structure FIRST to see actual folder structure.
Then use read_github_file to verify the paths and find class names.

Look for main model/encoder classes in these common locations:
- benchmarks/model*.py
- models/model.py
- src/model.py
- networks/model.py

Example output format:
```
File: [ACTUAL VERIFIED PATH]
  - Classes: [CLASS NAME] (line XX)
  - Imports from: [DEPENDENCY PATH]
```

Focus on:
1. Core architecture and design (from paper)
2. What problems it addresses
3. How it could improve {model_config.get("model_name", "the target model")} (component-level)
4. VERIFIED file paths from GitHub (use list_repo_structure!)

IMPORTANT: You MUST use write_file to save the summary.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[OTHER_PAPER_READER_2] Agent did not write file, extracting from response...")
        messages = result.get("messages", [])
        summary_content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Reference Paper" in msg.content or "## Overview" in msg.content or "Architecture" in msg.content:
                    summary_content = msg.content
                    break
        if summary_content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(summary_content, encoding="utf-8")
            logger.info(f"[OTHER_PAPER_READER_2] Manually saved summary to {output_path}")
        else:
            logger.error("[OTHER_PAPER_READER_2] Could not extract summary content from agent response")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[OTHER_PAPER_READER_2] Summary saved to {output_path}"],
    }


async def other_paper_reader_3_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Read third selected paper and generate summary."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    repos_dir = str(debate_outputs / "repos")

    valid_papers = _load_selected_papers(target_model, current_iteration)

    if len(valid_papers) < 3:
        logger.warning("[OTHER_PAPER_READER_3] No third paper found, skipping")
        return {
            "processing_logs": ["[OTHER_PAPER_READER_3] No third paper - skipped"]
        }

    paper = valid_papers[2]
    pdf_path = paper.get("pdf_path")
    github_urls = paper.get("github_urls", [])
    paper_title = paper.get("title", "Unknown")
    cloned_repo_path = paper.get("cloned_repo_path")
    output_path = str(debate_outputs / "other_paper_3_summary.md")

    logger.info(f"[OTHER_PAPER_READER_3] Using: {paper_title[:50]}...")

    agent = ArticleResearcherAgent(
        mode="paper_summary",
        paper_name="other_paper_3",
        pdf_path=pdf_path,
        output_path=output_path,
        repos_dir=repos_dir,
        model_config=model_config,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Read the selected paper and write a focused summary WITH GitHub code.

Paper Title: {paper_title}
PDF Path: {pdf_path}
Output Path: {output_path}

## CRITICAL: GitHub Repository Available
GitHub URLs: {github_urls}
Repos Directory: {repos_dir}

You MUST:
1. Use clone_github_repo to clone: {github_urls[0] if github_urls else 'N/A'}
2. Use list_repo_structure to find model files
3. Use read_github_file to understand the code structure
4. List all FILE PATHS related to encoder/model in your summary

IMPORTANT: You MUST use write_file to save the summary.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    output_file = Path(output_path)
    if not output_file.exists():
        messages = result.get("messages", [])
        summary_content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Reference Paper" in msg.content or "## Overview" in msg.content:
                    summary_content = msg.content
                    break
        if summary_content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(summary_content, encoding="utf-8")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[OTHER_PAPER_READER_3] Summary saved to {output_path}"],
    }


async def other_paper_reader_4_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Read fourth selected paper and generate summary."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    repos_dir = str(debate_outputs / "repos")

    valid_papers = _load_selected_papers(target_model, current_iteration)

    if len(valid_papers) < 4:
        logger.warning("[OTHER_PAPER_READER_4] No fourth paper found, skipping")
        return {
            "processing_logs": ["[OTHER_PAPER_READER_4] No fourth paper - skipped"]
        }

    paper = valid_papers[3]
    pdf_path = paper.get("pdf_path")
    github_urls = paper.get("github_urls", [])
    paper_title = paper.get("title", "Unknown")
    output_path = str(debate_outputs / "other_paper_4_summary.md")

    logger.info(f"[OTHER_PAPER_READER_4] Using: {paper_title[:50]}...")

    agent = ArticleResearcherAgent(
        mode="paper_summary",
        paper_name="other_paper_4",
        pdf_path=pdf_path,
        output_path=output_path,
        repos_dir=repos_dir,
        model_config=model_config,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Read the selected paper and write a focused summary WITH GitHub code.

Paper Title: {paper_title}
PDF Path: {pdf_path}
Output Path: {output_path}

## CRITICAL: GitHub Repository Available
GitHub URLs: {github_urls}
Repos Directory: {repos_dir}

You MUST:
1. Use clone_github_repo to clone: {github_urls[0] if github_urls else 'N/A'}
2. Use list_repo_structure to find model files
3. Use read_github_file to understand the code structure
4. List all FILE PATHS related to encoder/model in your summary

IMPORTANT: You MUST use write_file to save the summary.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    output_file = Path(output_path)
    if not output_file.exists():
        messages = result.get("messages", [])
        summary_content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Reference Paper" in msg.content or "## Overview" in msg.content:
                    summary_content = msg.content
                    break
        if summary_content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(summary_content, encoding="utf-8")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[OTHER_PAPER_READER_4] Summary saved to {output_path}"],
    }


async def other_paper_reader_5_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Read fifth selected paper and generate summary."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    repos_dir = str(debate_outputs / "repos")

    valid_papers = _load_selected_papers(target_model, current_iteration)

    if len(valid_papers) < 5:
        logger.warning("[OTHER_PAPER_READER_5] No fifth paper found, skipping")
        return {
            "processing_logs": ["[OTHER_PAPER_READER_5] No fifth paper - skipped"]
        }

    paper = valid_papers[4]
    pdf_path = paper.get("pdf_path")
    github_urls = paper.get("github_urls", [])
    paper_title = paper.get("title", "Unknown")
    output_path = str(debate_outputs / "other_paper_5_summary.md")

    logger.info(f"[OTHER_PAPER_READER_5] Using: {paper_title[:50]}...")

    agent = ArticleResearcherAgent(
        mode="paper_summary",
        paper_name="other_paper_5",
        pdf_path=pdf_path,
        output_path=output_path,
        repos_dir=repos_dir,
        model_config=model_config,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Read the selected paper and write a focused summary WITH GitHub code.

Paper Title: {paper_title}
PDF Path: {pdf_path}
Output Path: {output_path}

## CRITICAL: GitHub Repository Available
GitHub URLs: {github_urls}
Repos Directory: {repos_dir}

You MUST:
1. Use clone_github_repo to clone: {github_urls[0] if github_urls else 'N/A'}
2. Use list_repo_structure to find model files
3. Use read_github_file to understand the code structure
4. List all FILE PATHS related to encoder/model in your summary

IMPORTANT: You MUST use write_file to save the summary.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    output_file = Path(output_path)
    if not output_file.exists():
        messages = result.get("messages", [])
        summary_content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Reference Paper" in msg.content or "## Overview" in msg.content:
                    summary_content = msg.content
                    break
        if summary_content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(summary_content, encoding="utf-8")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[OTHER_PAPER_READER_5] Summary saved to {output_path}"],
    }


async def proposal_generation_other1_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Generate improvement proposal based on paper 1 and weakness analysis."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    selected = _load_selected_papers(target_model, current_iteration)

    if len(selected) < 1:
        logger.error("[PROPOSAL_OTHER1] No selected papers found")
        return {
            "processing_logs": ["[PROPOSAL_OTHER1] ERROR: No selected papers"]
        }

    paper = selected[0]
    paper_title = paper.get("title", "Unknown")
    weakness_path = str(debate_outputs / "weakness_of_target_model.md")
    paper_summary_path = str(debate_outputs / "other_paper_1_summary.md")
    target_summary_path = str(debate_outputs / f"{target_model or 'deeptta'}_summary.md")
    output_path = str(debate_outputs / "proposal_other1.md")

    logger.info("=" * 60)
    logger.info(f"[STEP 6/12] 📝 PROPOSAL_OTHER1 - Generating proposal from {paper_title[:50]}...")

    agent = ArticleResearcherAgent(
        mode="generate_proposal",
        paper_name=paper_title,
        weakness_path=weakness_path,
        paper_summary_path=paper_summary_path,
        target_summary_path=target_summary_path,
        output_path=output_path,
        model_config=model_config,  # Pass model config for dynamic prompts
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Generate an improvement proposal based on the reference paper.

Paper Title: {paper_title}
Weakness Analysis: {weakness_path}
Reference Paper Summary: {paper_summary_path}
Target Model Summary: {target_summary_path}
Output Path: {output_path}

IMPORTANT: You MUST use write_file to save the proposal.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[PROPOSAL_OTHER1] Agent did not write file, extracting from response...")
        messages = result.get("messages", [])
        content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Proposal Based on" in msg.content or "## Weaknesses Addressed" in msg.content:
                    content = msg.content
                    break
        if content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            logger.info(f"[PROPOSAL_OTHER1] Manually saved proposal to {output_path}")
        else:
            logger.error("[PROPOSAL_OTHER1] Could not extract proposal content")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[PROPOSAL_OTHER1] Proposal saved to {output_path}"],
    }


async def proposal_generation_other2_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Generate improvement proposal based on paper 2 and weakness analysis."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    selected = _load_selected_papers(target_model, current_iteration)

    if len(selected) < 2:
        logger.error("[PROPOSAL_OTHER2] No second paper found")
        return {
            "processing_logs": ["[PROPOSAL_OTHER2] ERROR: No second paper"]
        }

    paper = selected[1]
    paper_title = paper.get("title", "Unknown")
    weakness_path = str(debate_outputs / "weakness_of_target_model.md")
    paper_summary_path = str(debate_outputs / "other_paper_2_summary.md")
    target_summary_path = str(debate_outputs / f"{target_model or 'deeptta'}_summary.md")
    output_path = str(debate_outputs / "proposal_other2.md")

    logger.info("=" * 60)
    logger.info(f"[STEP 7/12] 📝 PROPOSAL_OTHER2 - Generating proposal from {paper_title[:50]}...")

    agent = ArticleResearcherAgent(
        mode="generate_proposal",
        paper_name=paper_title,
        weakness_path=weakness_path,
        paper_summary_path=paper_summary_path,
        target_summary_path=target_summary_path,
        output_path=output_path,
        model_config=model_config,  # Pass model config for dynamic prompts
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Generate an improvement proposal based on the reference paper.

Paper Title: {paper_title}
Weakness Analysis: {weakness_path}
Reference Paper Summary: {paper_summary_path}
Target Model Summary: {target_summary_path}
Output Path: {output_path}

IMPORTANT: You MUST use write_file to save the proposal.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[PROPOSAL_OTHER2] Agent did not write file, extracting from response...")
        messages = result.get("messages", [])
        content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Proposal Based on" in msg.content or "## Weaknesses Addressed" in msg.content:
                    content = msg.content
                    break
        if content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            logger.info(f"[PROPOSAL_OTHER2] Manually saved proposal to {output_path}")
        else:
            logger.error("[PROPOSAL_OTHER2] Could not extract proposal content")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[PROPOSAL_OTHER2] Proposal saved to {output_path}"],
    }


# ==============================================================================
# OPTIMIZING & RANKING NODES (NEW)
# ==============================================================================

async def critique_proposals_initial_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Step 9: Critic provides initial critique of both proposals."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    proposal1_path = str(debate_outputs / "proposal_other1.md")
    proposal2_path = str(debate_outputs / "proposal_other2.md")
    weakness_path = str(debate_outputs / "weakness_of_target_model.md")
    output_path = str(debate_outputs / "critique_of_proposals.md")

    logger.info("=" * 60)
    logger.info("[STEP 9/13] 🔍 CRITIQUE_PROPOSALS_INITIAL - Analyzing both proposals")

    agent = CriticAgent(
        mode="critique_proposals_initial",
        weakness_path=weakness_path,
        proposal_paths=[proposal1_path, proposal2_path],
        model_config=model_config,  # Pass model config for dynamic prompts
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Provide critical analysis of both improvement proposals.

Inputs:
- Weakness Analysis: {weakness_path}
- Proposal 1: {proposal1_path}
- Proposal 2: {proposal2_path}

For EACH proposal, identify:
1. Strengths
2. Weaknesses and potential issues
3. Missing details or unclear points
4. Feasibility concerns

Be thorough and critical. Your feedback will guide the ranking process.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    return {
        "messages": result["messages"],
        "processing_logs": [f"[CRITIQUE_PROPOSALS_INITIAL] Critique saved to {output_path}"],
    }


async def ranking_round1_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Step 10: Model Researcher ranks the two proposals."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    critique_path = str(debate_outputs / "critique_of_proposals.md")
    proposal1_path = str(debate_outputs / "proposal_other1.md")
    proposal2_path = str(debate_outputs / "proposal_other2.md")
    target_summary_path = str(debate_outputs / f"{target_model or 'deeptta'}_summary.md")
    output_path = str(debate_outputs / "ranked_proposals_r1.md")

    logger.info("=" * 60)
    logger.info("[STEP 10/13] 📊 RANKING_ROUND1 - Ranking proposals based on critique")

    agent = ModelResearcherAgent(
        mode="rank_proposals",
        critique_path=critique_path,
        proposal1_path=proposal1_path,
        proposal2_path=proposal2_path,
        target_summary_path=target_summary_path,
        output_path=output_path,
        model_config=model_config,  # Pass model config for dynamic prompts
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Rank the two improvement proposals based on the critic's analysis.

Inputs:
- Critique: {critique_path}
- Proposal 1: {proposal1_path}
- Proposal 2: {proposal2_path}
- Target Model Summary: {target_summary_path}

Evaluate based on:
1. Feasibility (implementation complexity, code availability)
2. Impact (how well it addresses weaknesses)
3. Technical soundness
4. Risk level

Output: Ranked proposals with Rank #1 and Rank #2, with clear justifications.

IMPORTANT: You MUST use write_file to save the ranking to {output_path}
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    # Check if file was created, extract from response if needed
    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[RANKING_ROUND1] Agent did not write file, extracting from response...")
        messages = result.get("messages", [])
        content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Ranked Proposals" in msg.content or "## Evaluation Summary" in msg.content:
                    content = msg.content
                    break
        if content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            logger.info(f"[RANKING_ROUND1] Manually saved ranking to {output_path}")
        else:
            logger.error("[RANKING_ROUND1] Could not extract ranking content")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[RANKING_ROUND1] Ranking saved to {output_path}"],
    }


async def critique_ranked_proposals_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Step 11: Critic reviews the Model Researcher's ranking."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    ranking_r1_path = str(debate_outputs / "ranked_proposals_r1.md")
    proposal1_path = str(debate_outputs / "proposal_other1.md")
    proposal2_path = str(debate_outputs / "proposal_other2.md")
    weakness_path = str(debate_outputs / "weakness_of_target_model.md")
    output_path = str(debate_outputs / "critique_of_ranked_proposals.md")

    logger.info("=" * 60)
    logger.info("[STEP 11/13] 🔍 CRITIQUE_RANKED - Reviewing the ranking decision")

    agent = CriticAgent(
        mode="critique_ranked",
        ranking_r1_path=ranking_r1_path,
        proposal1_path=proposal1_path,
        proposal2_path=proposal2_path,
        weakness_path=weakness_path,
        model_config=model_config,  # Pass model config for dynamic prompts
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Review the Model Researcher's ranking decision and challenge if needed.

Inputs:
- Initial Ranking: {ranking_r1_path}
- Proposal 1: {proposal1_path}
- Proposal 2: {proposal2_path}
- Weakness Analysis: {weakness_path}

Evaluate:
1. Are the evaluation scores accurate?
2. Did the researcher overlook any critical factors?
3. Is the Rank #1 choice truly the best option?

Be critical but constructive. If you disagree with the ranking, explain why.
Highlight concerns that must be addressed before final implementation.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    return {
        "messages": result["messages"],
        "processing_logs": [f"[CRITIQUE_RANKED] Critique saved to {output_path}"],
    }


async def final_ranking_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Step 12: Model Researcher makes final proposal selection."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    critique_ranked_path = str(debate_outputs / "critique_of_ranked_proposals.md")
    ranking_r1_path = str(debate_outputs / "ranked_proposals_r1.md")
    proposal1_path = str(debate_outputs / "proposal_other1.md")
    proposal2_path = str(debate_outputs / "proposal_other2.md")
    output_path = str(debate_outputs / "final_rank1_proposal.md")

    logger.info("=" * 60)
    logger.info("[STEP 12/13] ✅ FINAL_RANKING - Selecting final Rank #1 proposal")

    agent = ModelResearcherAgent(
        mode="final_ranking",
        critique_ranked_path=critique_ranked_path,
        ranking_r1_path=ranking_r1_path,
        proposal1_path=proposal1_path,
        proposal2_path=proposal2_path,
        output_path=output_path,
        model_config=model_config,  # Pass model config for dynamic prompts
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = f"""Make the final decision on which proposal to implement.

Inputs:
- Critic's Feedback on Ranking: {critique_ranked_path}
- Initial Ranking: {ranking_r1_path}
- Proposal 1: {proposal1_path}
- Proposal 2: {proposal2_path}

Review the critic's concerns and decide:
- MAINTAIN your initial Rank #1 (with updated justification), OR
- CHANGE to the other proposal (if critic's arguments are compelling)

Provide a clear, final decision with:
1. Selected proposal (Rank #1)
2. Comprehensive justification
3. Response to all critic's concerns
4. Risk mitigation plan
5. Next steps for implementation

IMPORTANT: You MUST use write_file to save the final decision to {output_path}
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    # Check if file was created, extract from response if needed
    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[FINAL_RANKING] Agent did not write file, extracting from response...")
        messages = result.get("messages", [])
        content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if "# Final Rank #1 Proposal" in msg.content or "## Decision" in msg.content:
                    content = msg.content
                    break
        if content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            logger.info(f"[FINAL_RANKING] Manually saved final ranking to {output_path}")
        else:
            logger.error("[FINAL_RANKING] Could not extract final ranking content")

    return {
        "messages": result["messages"],
        "processing_logs": [f"[FINAL_RANKING] Final Rank #1 proposal saved to {output_path}"],
    }


# ==============================================================================
# LEGACY DEBATE NODES (DEPRECATED - NOT USED IN CURRENT WORKFLOW)
# ==============================================================================
# These nodes have been replaced by the new Optimizing & Ranking workflow:
# - debate_round1_node → critique_proposals_initial_node + ranking_round1_node
# - critic_round2_node → critique_ranked_proposals_node
# - debate_round2_node → final_ranking_node
#
# The DebateExpertAgent class has been removed. If you need to re-enable
# legacy debate functionality, create DebateExpertAgent in agents/ directory.
# ==============================================================================


def _extract_config_changes_section(debate_outputs: Path) -> str:
    """Extract Config Changes section from debate proposal files.

    Searches in order:
    1. proposal_round*_article*.md (most detailed)
    2. response_round*_article*.md

    Returns the Config Changes section as a string, or empty string if not found.
    """
    import glob
    import re

    # Files to search (in priority order)
    search_patterns = [
        "proposal_round*_article*.md",
        "response_round*_article*.md",
    ]

    for pattern in search_patterns:
        files = sorted(glob.glob(str(debate_outputs / pattern)))
        for file_path in files:
            try:
                content = Path(file_path).read_text(encoding='utf-8')

                # Look for Config Changes section with various headers
                # Pattern 1: "### Config Changes" section
                config_match = re.search(
                    r'(###?\s*Config Changes.*?)(?=\n##|\n###\s+[A-Z]|\Z)',
                    content,
                    re.DOTALL | re.IGNORECASE
                )

                if config_match:
                    config_section = config_match.group(1).strip()

                    # Verify it has actual parameter changes
                    if "**Parameter**:" in config_section or "encoder." in config_section or "decoder." in config_section:
                        logger.info(f"[EXTRACT_CONFIG_SECTION] Found Config Changes in {Path(file_path).name}")

                        # Format as Section 6
                        formatted_section = f"""## 6. Config Changes (Auto-extracted from debate)

{config_section.replace('### Config Changes', '').replace('## Config Changes', '').strip()}
"""
                        return formatted_section

            except Exception as e:
                logger.warning(f"[EXTRACT_CONFIG_SECTION] Error reading {file_path}: {e}")
                continue

    return ""


def save_debate_transcript_node(state: MARBLEState) -> Dict[str, Any]:
    """Save the entire debate transcript to a file."""
    target_model = state.get("target_model")
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    transcript_path = debate_outputs / "debate_transcript.md"

    messages = state.get("messages", [])
    transcript_content = "# Build Debate Transcript\n\n"

    for i, msg in enumerate(messages):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        content = msg.content if hasattr(msg, 'content') else str(msg)

        # Truncate very long messages
        if len(content) > 5000:
            content = content[:5000] + "\n... [TRUNCATED]"

        transcript_content += f"## Turn {i + 1} ({role})\n\n{content}\n\n---\n\n"

    transcript_path.write_text(transcript_content, encoding='utf-8')
    logger.info(f"[TRANSCRIPT] Saved debate transcript to {transcript_path}")

    return {
        "processing_logs": [f"[TRANSCRIPT] Saved to {transcript_path}"],
    }


async def proposal_agent_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Step 13: Synthesize final ranked proposal into implementation proposal with code validation."""
    target_model = state.get("target_model")
    current_iteration = state.get("current_iteration", 1)
    model_config = _get_model_config(target_model)
    build_workspace = _get_workspace_path(target_model, current_iteration)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    # Get model-specific settings
    model_name = model_config.get("model_name", target_model or "DeepTTA")
    components = model_config.get("components", ["encoder", "decoder"])
    components_display = " / ".join(components)
    component_templates = model_config.get("component_templates", {})

    # Build template paths from config
    template_paths = {}
    for comp, rel_path in component_templates.items():
        template_paths[comp] = str(build_workspace / rel_path)

    # NEW: Read final ranked proposal instead of debate transcript
    final_proposal_path = str(debate_outputs / "final_rank1_proposal.md")
    paper_summary_1 = str(debate_outputs / "other_paper_1_summary.md")
    paper_summary_2 = str(debate_outputs / "other_paper_2_summary.md")
    repos_dir = str(debate_outputs / "repos")
    output_path = str(debate_outputs / "implementation_proposal.md")

    # Config file path for config change suggestions
    config_path = str(build_workspace / "config.yaml")

    logger.info("=" * 60)
    logger.info(f"[STEP 13/13] 📝 PROPOSAL_AGENT - Creating implementation proposal for {model_name}")

    # Get iteration context for prompt injection
    iteration_context = state.get("iteration_context", "")

    # Phase 1: Proposal Agent creates initial proposal
    proposal_agent = ProposalAgent(
        final_proposal_path=final_proposal_path,
        other_summary_paths=[paper_summary_1, paper_summary_2],
        repos_dir=repos_dir,
        output_path=output_path,
        model_config=model_config,  # Pass model config for dynamic prompts
        iteration_context=iteration_context,  # Pass iteration context for prompt injection
        checkpointer=checkpointer
    )
    proposal_agent.initialize_agent()

    prompt = f"""Create an implementation proposal for {model_name} improvement based on the final Rank #1 proposal.

## Step 0: Read Final Rank #1 Proposal
FIRST, read the final ranking decision:
- Final Rank #1 Proposal: {final_proposal_path}

This file contains:
- The selected proposal (either from Paper 1 or Paper 2)
- Justification for the selection
- Key features to implement
- Risk mitigation plan

## Step 1: Read Summaries
Read the paper summaries to understand which code to use:
- Paper 1 Summary: {paper_summary_1}
- Paper 2 Summary: {paper_summary_2}

Based on the final ranking, focus on the SELECTED paper's summary.

CRITICAL: The summaries contain "GitHub Code Paths" section with:
- Exact file paths to read
- Class names to extract
- Dependency tree

## Step 2: Identify Target Component
From the summaries, find:
- "Recommended Target Component" ({components_display})
- Which paper/repo to use

## Step 3: DIRECTLY EXPLORE the Repositories (CRITICAL!)
REPOS ROOT: {repos_dir}

DO NOT trust summary paths - EXPLORE YOURSELF:

1. First, list what repos exist:
   - list_repo_structure(repo_path="{repos_dir}")

2. Then explore each repo structure:
   - list_repo_structure(repo_path="{repos_dir}/[REPO_NAME]")

3. Find model files (common locations):
   - benchmarks/model*.py
   - models/model.py
   - src/model.py
   - networks/*.py

4. Read the file and find a class that:
   - Inherits from nn.Module
   - Has __init__ and forward methods
   - Is an encoder (look for neural network layers, attention, conv layers)

5. Copy the ENTIRE class code

## Step 4: Copy the REAL Code
After reading the file, copy the ENTIRE class definition.
- Include the full __init__ method
- Include the full forward method
- Include any other methods in the class
- DO NOT use placeholders like "..." or "pass"

## Step 5: Read Config File
Read the current config to understand what hyperparameters can be changed:
- Config File: {config_path}

The config has "# Can touch" and "# Never touch" sections.
ONLY suggest changes to "# Can touch" sections:
- encoder.architecture (hidden_dim, heads, dropout)
- decoder.architecture (hidden_dim, heads, dropout)
- training (epochs, learning_rate, weight_decay, gradient_clipping)

## Step 6: Output Proposal
Output the proposal in markdown format DIRECTLY in your response.
DO NOT use write_file tool - your response will be saved automatically.

Format:
```markdown
# Implementation Proposal for {model_name}

## 1. Decision Summary
- Target Component: [{components_display}]
- Source Model: [paper name from summary]
- Source Class: [exact class name from summary, e.g., EdgePathNN]
- Source File: [exact file path from summary]

## 2. Architecture Overview
[Description based on the paper summary]

## 3. Code to Implement
[PASTE THE ENTIRE CLASS CODE HERE - copied from read_github_file output]

## 4. Dependencies
[List any imports or helper classes needed]

## 5. Integration Notes
[How to integrate with {model_name}]

## 6. Config Changes (IMPORTANT!)
Based on the new architecture, suggest hyperparameter changes.
ONLY for "# Can touch" sections. Format:

- **Parameter**: [param.path]
- **Current → Proposed**: [old] → [new]
- **Reason**: [why]

Allowed parameters: encoder.architecture.*, decoder.architecture.*, training.*
Do NOT change "# Never touch" sections.
```

CRITICAL RULES:
1. Only use code from the paths mentioned in the paper summaries
2. Copy the FULL class code, not snippets
3. The class name in proposal MUST match the class name in summary
4. NO placeholders (pass, ..., TODO) - only real code
5. Include Section 6 (Config Changes) with the EXACT format shown above
6. DO NOT use write_file tool - output the markdown DIRECTLY in your response
"""

    result = await proposal_agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    # Phase 2: Validation - Check if proposal uses correct code from summaries
    logger.info("[PROPOSAL] Phase 2: Validating proposal")

    output_file = Path(output_path)
    validation_passed = False
    max_retries = 2

    # First, check if file exists. If not, try to extract from response
    if not output_file.exists():
        logger.warning("[PROPOSAL] File not created, extracting from response...")
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                if "# Implementation Proposal" in content or "## 1. Decision Summary" in content:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    output_file.write_text(content, encoding='utf-8')
                    logger.info("[PROPOSAL] Extracted and saved proposal from response")
                    break

    for retry in range(max_retries + 1):
        if not output_file.exists():
            logger.warning(f"[PROPOSAL] Proposal file still not found after retry {retry + 1}")
            break

        proposal_content = output_file.read_text(encoding='utf-8')

        # Validation: Check for ACTUAL code patterns (not class name matching)
        has_class = "class " in proposal_content
        has_init = "def __init__" in proposal_content
        has_forward = "def forward" in proposal_content
        has_nn_module = "nn.Module" in proposal_content or "torch.nn" in proposal_content

        # Check for placeholders
        has_placeholder_pass = "def forward" in proposal_content and proposal_content.count("pass") > 2
        has_ellipsis = "self." in proposal_content and "..." in proposal_content

        code_quality = has_class and has_init and has_forward and has_nn_module
        no_placeholders = not has_placeholder_pass and not has_ellipsis

        logger.info(f"[PROPOSAL] Validation check:")
        logger.info(f"  - Has class: {has_class}, __init__: {has_init}, forward: {has_forward}")
        logger.info(f"  - Has nn.Module: {has_nn_module}")
        logger.info(f"  - No placeholders: {no_placeholders}")

        if code_quality and no_placeholders:
            logger.info("[PROPOSAL] Validation PASS: Proposal contains valid implementation code")
            validation_passed = True
            break
        else:
            if retry < max_retries:
                logger.warning(f"[PROPOSAL] Validation FAIL (attempt {retry + 1})")

                # Re-run with more specific instructions
                retry_prompt = f"""The proposal is missing proper implementation code.

Required:
- A class that inherits from nn.Module
- Complete __init__ method with layer definitions
- Complete forward method with actual logic (no 'pass' or '...')

Please:
1. Use list_repo_structure to explore: {repos_dir}
2. Find an encoder class (look in benchmarks/, models/, src/)
3. Use read_github_file to read the FULL class code
4. Copy the ENTIRE class into the proposal (not snippets)
5. Save to: {output_path}
"""
                result = await proposal_agent.compiled_agent.ainvoke({
                    **state,
                    "messages": result["messages"] + [HumanMessage(content=retry_prompt)]
                })

    if not validation_passed:
        logger.warning("[PROPOSAL] Validation incomplete, proceeding with current proposal")

    result_state = result

    # Check if file was actually written, if not extract from response and save
    output_file = Path(output_path)
    if not output_file.exists():
        logger.warning("[PROPOSAL] Agent did not write file, extracting from response...")

        # Extract proposal content from the last AI message
        messages = result_state.get("messages", [])
        proposal_content = None

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                # Look for markdown proposal content
                if "# Implementation Proposal" in content or "## 1. Decision Summary" in content:
                    proposal_content = content
                    break
                # Also check if the content looks like a proposal (has key sections)
                elif "Component to Modify" in content and "Architecture" in content:
                    proposal_content = content
                    break

        if proposal_content:
            # Clean up content if it has extra formatting
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(proposal_content, encoding='utf-8')
            logger.info(f"[PROPOSAL] Manually saved proposal to {output_path}")
        else:
            logger.error("[PROPOSAL] Could not extract proposal content from agent response")
            return {
                "messages": result_state["messages"],
                "processing_logs": [f"[PROPOSAL] ERROR: Failed to save proposal to {output_path}"],
            }

    # Phase 4: Ensure Config Changes section exists
    # If implementation_proposal.md doesn't have Config Changes, extract from debate proposals
    logger.info("[PROPOSAL] Phase 4: Checking Config Changes section")
    proposal_content = output_file.read_text(encoding='utf-8')

    has_config_changes = (
        "## 6. Config Changes" in proposal_content or
        "### Config Changes" in proposal_content or
        "**Parameter**:" in proposal_content
    )

    if not has_config_changes:
        logger.warning("[PROPOSAL] Config Changes section missing, extracting from debate proposals...")

        # Try to extract config changes from debate proposal files
        config_section = _extract_config_changes_section(debate_outputs)

        if config_section:
            # Append config changes section to the proposal
            proposal_content += f"\n\n{config_section}"
            output_file.write_text(proposal_content, encoding='utf-8')
            logger.info("[PROPOSAL] Added Config Changes section from debate proposals")
        else:
            logger.warning("[PROPOSAL] Could not find Config Changes in any debate proposal files")

    logger.info("[PROPOSAL] Phase 5: Final proposal complete")
    return {
        "messages": result_state["messages"],
        "processing_logs": [f"[PROPOSAL] Implementation proposal saved to {output_path}"],
    }


# ==============================================================================
# 5-ROUND DEBATE NODES (Fixed Round System)
# ==============================================================================

def _init_transcript(transcript_path: Path, model_name: str) -> None:
    """Initialize debate_transcript.md with header."""
    from datetime import datetime
    header = f"""# Debate Transcript: {model_name} Improvement

## Overview
- Target Model: {model_name}
- Start Time: {datetime.now().isoformat()}
- Status: IN_PROGRESS
- Rounds: 5 (Fixed)

---

"""
    transcript_path.write_text(header, encoding='utf-8')


def _append_to_transcript(transcript_path: Path, content: str) -> None:
    """Append content to debate_transcript.md."""
    with open(transcript_path, 'a', encoding='utf-8') as f:
        f.write(content)


def _read_file_content(file_path: str) -> str:
    """Read file content safely."""
    try:
        return Path(file_path).read_text(encoding='utf-8')
    except Exception as e:
        return f"[Error reading file: {e}]"


def init_debate_node(state: MARBLEState) -> Dict[str, Any]:
    """Initialize debate: create transcript, set initial state."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    transcript_path = debate_outputs / "debate_transcript.md"
    model_name = model_config.get("model_name", target_model)

    _init_transcript(transcript_path, model_name)

    # Add Phase 1 & 2 summaries to transcript
    weakness_path = debate_outputs / "weakness_of_target_model.md"

    # All 5 paper summaries
    summary_paths = [
        debate_outputs / f"other_paper_{i}_summary.md"
        for i in range(1, 6)
    ]

    phase_content = """## Phase 1: Target Model Analysis

"""
    if weakness_path.exists():
        phase_content += f"### Weakness Analysis\n{_read_file_content(str(weakness_path))}\n\n"

    phase_content += """## Phase 2: External Papers (5 Candidates)

"""
    for i, summary_path in enumerate(summary_paths, 1):
        if summary_path.exists():
            phase_content += f"### Paper {i} Summary\n{_read_file_content(str(summary_path))[:2000]}...\n\n"

    phase_content += "---\n\n"
    _append_to_transcript(transcript_path, phase_content)

    logger.info(f"[INIT_DEBATE] Transcript initialized at {transcript_path}")

    return {
        "build_debate_round": 0,
        "build_debate_max_rounds": 5,
        "build_debate_consensus_reached": False,
        "article1_proposal_status": "active",
        "article2_proposal_status": "active",
        "article3_proposal_status": "active",
        "article4_proposal_status": "active",
        "article5_proposal_status": "active",
        "build_debate_transcript_path": str(transcript_path),
        "processing_logs": ["[INIT_DEBATE] Transcript initialized with 5 paper summaries"],
    }


async def debate_round_1_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Round 1: 5 Article experts propose improvements in PARALLEL (1 proposal each)."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    selected = _load_selected_papers(target_model, current_iteration)

    transcript_path = Path(state.get("build_debate_transcript_path"))

    logger.info("=" * 60)
    logger.info(f"[ROUND 1] {len(selected)} Article experts proposing in PARALLEL...")

    # Get file paths
    weakness_path = str(debate_outputs / "weakness_of_target_model.md")
    target_summary_path = str(debate_outputs / f"{target_model or 'deeptta'}_summary.md")

    # Config file path for proposals
    workspace_path = _get_workspace_path(target_model, current_iteration)
    config_path = str(workspace_path / "config.yaml")

    from .prompts.prompt_factory import PromptFactory
    iteration_context = state.get("iteration_context", "")
    prompt_factory = PromptFactory(model_config, iteration_context=iteration_context)

    num_papers = min(len(selected), 5)

    # Helper function for processing a single expert
    async def process_expert(i: int) -> Tuple[int, str, str]:
        """Process a single expert and return (index, paper_name, proposal_content)."""
        paper = selected[i]
        paper_name = paper.get("title", f"Paper {i+1}")[:50]
        summary_path = str(debate_outputs / f"other_paper_{i+1}_summary.md")
        proposal_path = str(debate_outputs / f"proposal_round1_article{i+1}.md")

        logger.info(f"[ROUND 1] Starting Article {i+1}: {paper_name}...")

        agent = ArticleResearcherAgent(
            mode="generate_proposal",
            paper_name=paper_name,
            weakness_path=weakness_path,
            paper_summary_path=summary_path,
            target_summary_path=target_summary_path,
            output_path=proposal_path,
            model_config=model_config,
            iteration_context=iteration_context,
            checkpointer=checkpointer
        )
        agent.initialize_agent()

        prompt = prompt_factory.get_debate_propose_prompt(
            paper_name=paper_name,
            paper_summary_path=summary_path,
            weakness_path=weakness_path,
            target_summary_path=target_summary_path,
            current_round=1,
            config_path=config_path,
        )

        result = await agent.compiled_agent.ainvoke({
            **state,
            "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
        })

        # Extract proposal from agent response
        proposal_content = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                proposal_content = msg.content
                break

        # Save to file
        (debate_outputs / f"proposal_round1_article{i+1}.md").write_text(proposal_content, encoding='utf-8')
        logger.info(f"[ROUND 1] Article {i+1} proposal saved")

        return (i, paper_name, proposal_content)

    # Run all experts in parallel
    tasks = [process_expert(i) for i in range(num_papers)]
    results = await asyncio.gather(*tasks)

    # Collect results and build round content
    round_content = """## Round 1: Initial Proposals (5 Experts - Parallel)

"""
    for idx, paper_name, proposal_content in sorted(results, key=lambda x: x[0]):
        round_content += f"### [{paper_name}] Expert {idx+1} Proposal\n\n{proposal_content}\n\n---\n\n"

    _append_to_transcript(transcript_path, round_content)
    logger.info(f"[ROUND 1] Complete - {num_papers} proposals saved (PARALLEL)")

    return {
        "build_debate_round": 1,
        "processing_logs": [f"[ROUND 1] {num_papers} Article experts proposed in PARALLEL"],
    }


async def debate_round_2_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Round 2: Critic provides feedback on all 5 proposals."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    transcript_path = Path(state.get("build_debate_transcript_path"))

    logger.info("=" * 60)
    logger.info("[ROUND 2] Critic providing feedback on 5 proposals...")

    # Read all 5 proposals
    proposals_combined = ""
    proposal_paths = []
    for i in range(1, 6):
        proposal_path = str(debate_outputs / f"proposal_round1_article{i}.md")
        proposal_content = _read_file_content(proposal_path)
        if "Error reading" not in proposal_content:
            proposals_combined += f"### Proposal {i}:\n{proposal_content}\n\n"
            proposal_paths.append(proposal_path)

    from .prompts.prompt_factory import PromptFactory
    iteration_context = state.get("iteration_context", "")
    prompt_factory = PromptFactory(model_config, iteration_context=iteration_context)

    agent = CriticAgent(
        mode="critique_proposals_initial",
        weakness_path=str(debate_outputs / "weakness_of_target_model.md"),
        proposal_paths=proposal_paths,
        output_path=str(debate_outputs / "critique_round2.md"),
        model_config=model_config,
        iteration_context=iteration_context,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    prompt = prompt_factory.get_debate_critique_round_prompt(
        proposals=proposals_combined,
        current_round=2,
    )

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    critique_content = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            critique_content = msg.content
            break

    round_content = f"""## Round 2: Critic Feedback (5 Proposals)

{critique_content}

---

"""
    _append_to_transcript(transcript_path, round_content)
    (debate_outputs / "critique_round2.md").write_text(critique_content, encoding='utf-8')

    logger.info("[ROUND 2] Complete - critique for 5 proposals saved")

    return {
        "build_debate_round": 2,
        "processing_logs": ["[ROUND 2] Critic provided feedback on 5 proposals"],
    }


async def debate_round_3_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Round 3: 5 Article experts respond and revise in PARALLEL (FULL ADOPTION possible)."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)
    selected = _load_selected_papers(target_model, current_iteration)

    transcript_path = Path(state.get("build_debate_transcript_path"))

    logger.info("=" * 60)
    logger.info("[ROUND 3] 5 Article experts responding in PARALLEL...")

    # Read all 5 proposals and critique
    proposals = []
    for i in range(1, 6):
        proposal = _read_file_content(str(debate_outputs / f"proposal_round1_article{i}.md"))
        proposals.append(proposal if "Error reading" not in proposal else "")
    critique = _read_file_content(str(debate_outputs / "critique_round2.md"))

    # Config file path for proposals
    workspace_path = _get_workspace_path(target_model, current_iteration)
    config_path = str(workspace_path / "config.yaml")

    from .prompts.prompt_factory import PromptFactory
    iteration_context = state.get("iteration_context", "")
    prompt_factory = PromptFactory(model_config, iteration_context=iteration_context)

    # Get article status for all 5
    article_statuses = [
        state.get("article1_proposal_status", "active"),
        state.get("article2_proposal_status", "active"),
        state.get("article3_proposal_status", "active"),
        state.get("article4_proposal_status", "active"),
        state.get("article5_proposal_status", "active"),
    ]

    num_papers = min(len(selected), 5)

    # Helper function for processing a single expert
    async def process_expert(i: int) -> Tuple[int, str, str]:
        """Process a single expert and return (index, response, status)."""
        if article_statuses[i] != "active":
            logger.info(f"[ROUND 3] Article {i+1} already {article_statuses[i]}, skipping")
            return (i, "", article_statuses[i])

        paper_name = selected[i].get("title", f"Paper {i+1}")[:50]

        # Combine other proposals (all except self)
        other_proposals = ""
        for j in range(num_papers):
            if j != i and proposals[j]:
                other_name = selected[j].get("title", f"Paper {j+1}")[:50] if j < len(selected) else f"Paper {j+1}"
                other_proposals += f"### Expert {j+1} ({other_name[:30]}):\n{proposals[j][:1500]}\n\n"

        prompt = prompt_factory.get_debate_respond_and_revise_prompt(
            paper_name=paper_name,
            other_paper_name="Other 4 Experts",
            own_proposal=proposals[i],
            other_proposal=other_proposals,
            critic_feedback=critique,
            current_round=3,
            config_path=config_path,
        )

        agent = ArticleResearcherAgent(
            mode="generate_proposal",
            paper_name=paper_name,
            output_path=str(debate_outputs / f"response_round3_article{i+1}.md"),
            model_config=model_config,
            iteration_context=iteration_context,
            checkpointer=checkpointer
        )
        agent.initialize_agent()

        result = await agent.compiled_agent.ainvoke({
            **state,
            "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
        })

        response = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                response = msg.content
                break

        # Check for FULL_ADOPTION
        status = article_statuses[i]
        if "FULL_ADOPTION" in response or "I WITHDRAW" in response:
            status = "withdrawn"
            logger.info(f"[ROUND 3] {paper_name} Expert: FULL ADOPTION declared")

        # Save response file
        (debate_outputs / f"response_round3_article{i+1}.md").write_text(response, encoding='utf-8')
        logger.info(f"[ROUND 3] Article {i+1} response saved")

        return (i, response, status)

    # Run all experts in parallel
    tasks = [process_expert(i) for i in range(num_papers)]
    results = await asyncio.gather(*tasks)

    # Collect results and build round content
    round_content = """## Round 3: Response & Revision (5 Experts - Parallel)

"""
    for idx, response, status in results:
        article_statuses[idx] = status
        if response:
            paper_name = selected[idx].get("title", f"Paper {idx+1}")[:50]
            round_content += f"### [{paper_name}] Expert {idx+1} Response\n\n{response}\n\n---\n\n"

    _append_to_transcript(transcript_path, round_content)
    logger.info("[ROUND 3] Complete - 5 expert responses saved (PARALLEL)")

    return {
        "build_debate_round": 3,
        "article1_proposal_status": article_statuses[0],
        "article2_proposal_status": article_statuses[1],
        "article3_proposal_status": article_statuses[2] if len(article_statuses) > 2 else "active",
        "article4_proposal_status": article_statuses[3] if len(article_statuses) > 3 else "active",
        "article5_proposal_status": article_statuses[4] if len(article_statuses) > 4 else "active",
        "processing_logs": ["[ROUND 3] 5 Article experts responded in PARALLEL"],
    }


async def debate_round_4_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Round 4: Model researcher ranks 5 proposals + Critic provides final critique."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    transcript_path = Path(state.get("build_debate_transcript_path"))

    logger.info("=" * 60)
    logger.info("[ROUND 4] Ranking 5 proposals and final critique...")

    round_content = """## Round 4: Ranking & Final Critique (5 Proposals)

"""

    critique_r2 = _read_file_content(str(debate_outputs / "critique_round2.md"))

    # Gather all 5 proposals (use round 3 response if available, else round 1 proposal)
    all_proposals = ""
    proposal_paths = []
    for i in range(1, 6):
        proposal = _read_file_content(str(debate_outputs / f"proposal_round1_article{i}.md"))
        response = _read_file_content(str(debate_outputs / f"response_round3_article{i}.md"))

        latest = response if "Error reading" not in response else proposal
        if "Error reading" not in latest:
            all_proposals += f"### Article {i} Latest Proposal:\n{latest[:2000]}\n\n"
            proposal_paths.append(str(debate_outputs / f"proposal_round1_article{i}.md"))

    from .prompts.prompt_factory import PromptFactory
    iteration_context = state.get("iteration_context", "")
    prompt_factory = PromptFactory(model_config, iteration_context=iteration_context)

    # Model Researcher ranking
    ranking_prompt = prompt_factory.get_debate_ranking_prompt(
        all_proposals=all_proposals,
        critic_feedback=critique_r2,
        current_round=4,
    )

    ranking_agent = ModelResearcherAgent(
        mode="rank_proposals",
        critique_path=str(debate_outputs / "critique_round2.md"),
        proposal1_path=proposal_paths[0] if len(proposal_paths) > 0 else "",
        proposal2_path=proposal_paths[1] if len(proposal_paths) > 1 else "",
        target_summary_path=str(debate_outputs / f"{target_model or 'deeptta'}_summary.md"),
        output_path=str(debate_outputs / "ranking_round4.md"),
        model_config=model_config,
        iteration_context=iteration_context,
        checkpointer=checkpointer
    )
    ranking_agent.initialize_agent()

    ranking_result = await ranking_agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=ranking_prompt)]
    })

    ranking_content = ""
    for msg in reversed(ranking_result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            ranking_content = msg.content
            break

    round_content += f"### [Model Researcher] Ranking (5 Proposals)\n\n{ranking_content}\n\n---\n\n"
    (debate_outputs / "ranking_round4.md").write_text(ranking_content, encoding='utf-8')

    # Critic final critique
    critique_prompt = prompt_factory.get_debate_critique_round_prompt(
        proposals=all_proposals,
        current_round=4,
    )

    critique_agent = CriticAgent(
        mode="critique_ranked",
        ranking_r1_path=str(debate_outputs / "ranking_round4.md"),
        proposal1_path=proposal_paths[0] if len(proposal_paths) > 0 else "",
        proposal2_path=proposal_paths[1] if len(proposal_paths) > 1 else "",
        weakness_path=str(debate_outputs / "weakness_of_target_model.md"),
        output_path=str(debate_outputs / "critique_round4.md"),
        model_config=model_config,
        iteration_context=iteration_context,
        checkpointer=checkpointer
    )
    critique_agent.initialize_agent()

    critique_result = await critique_agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=critique_prompt)]
    })

    final_critique = ""
    for msg in reversed(critique_result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            final_critique = msg.content
            break

    round_content += f"### [Critic] Final Critique (5 Proposals)\n\n{final_critique}\n\n---\n\n"
    (debate_outputs / "critique_round4.md").write_text(final_critique, encoding='utf-8')

    _append_to_transcript(transcript_path, round_content)
    logger.info("[ROUND 4] Complete - ranking and critique for 5 proposals saved")

    return {
        "build_debate_round": 4,
        "processing_logs": ["[ROUND 4] Ranking and final critique for 5 proposals complete"],
    }


async def debate_round_5_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Round 5: Model researcher makes final selection from 5 proposals."""
    target_model = state.get("target_model")
    model_config = _get_model_config(target_model)
    current_iteration = state.get("current_iteration", 1)
    debate_outputs = _ensure_workspace(target_model, current_iteration)

    transcript_path = Path(state.get("build_debate_transcript_path"))

    logger.info("=" * 60)
    logger.info("[ROUND 5] Final selection from 5 proposals...")

    # Read all 5 proposals and responses
    ranking = _read_file_content(str(debate_outputs / "ranking_round4.md"))
    final_critique = _read_file_content(str(debate_outputs / "critique_round4.md"))

    # Build summary for all 5 proposals
    all_rounds_summary = "### Round 1 Proposals (5 Experts):\n\n"
    final_proposals = ""
    proposal_paths = []

    for i in range(1, 6):
        proposal = _read_file_content(str(debate_outputs / f"proposal_round1_article{i}.md"))
        response = _read_file_content(str(debate_outputs / f"response_round3_article{i}.md"))

        if "Error reading" not in proposal:
            all_rounds_summary += f"**Article {i}:** {proposal[:500]}...\n\n"
            proposal_paths.append(str(debate_outputs / f"proposal_round1_article{i}.md"))

        latest = response if "Error reading" not in response else proposal
        if "Error reading" not in latest:
            final_proposals += f"### Article {i}:\n{latest[:1500]}\n\n"

    all_rounds_summary += f"\n### Round 3 Responses (5 Experts):\n\n"
    for i in range(1, 6):
        response = _read_file_content(str(debate_outputs / f"response_round3_article{i}.md"))
        if "Error reading" not in response:
            all_rounds_summary += f"**Article {i}:** {response[:500]}...\n\n"

    all_rounds_summary += f"\n### Round 4 Ranking:\n{ranking}\n"

    # Get article statuses for all 5
    article_statuses = [
        state.get(f"article{i}_proposal_status", "active")
        for i in range(1, 6)
    ]
    status_summary = ", ".join([f"Article {i+1}: {s}" for i, s in enumerate(article_statuses)])

    from .prompts.prompt_factory import PromptFactory
    iteration_context = state.get("iteration_context", "")
    prompt_factory = PromptFactory(model_config, iteration_context=iteration_context)

    # Use existing prompt with extended info
    selection_prompt = prompt_factory.get_debate_final_selection_prompt(
        all_rounds_summary=all_rounds_summary,
        final_proposals=final_proposals,
        final_critique=final_critique,
        current_round=5,
        article1_status=article_statuses[0],
        article2_status=f"{article_statuses[1]} | Article3: {article_statuses[2]} | Article4: {article_statuses[3]} | Article5: {article_statuses[4]}",
    )

    agent = ModelResearcherAgent(
        mode="final_ranking",
        critique_ranked_path=str(debate_outputs / "critique_round4.md"),
        ranking_r1_path=str(debate_outputs / "ranking_round4.md"),
        proposal1_path=proposal_paths[0] if len(proposal_paths) > 0 else "",
        proposal2_path=proposal_paths[1] if len(proposal_paths) > 1 else "",
        output_path=str(debate_outputs / "final_selection_round5.md"),
        model_config=model_config,
        iteration_context=iteration_context,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=selection_prompt)]
    })

    selection_content = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            selection_content = msg.content
            break

    # Check for consensus keywords
    consensus_reached = "STRONG_CONSENSUS" in selection_content or "FINAL_DECISION" in selection_content
    consensus_reason = None
    if consensus_reached:
        if "STRONG_CONSENSUS" in selection_content:
            consensus_reason = "STRONG_CONSENSUS"
        elif "FINAL_DECISION" in selection_content:
            consensus_reason = "FINAL_DECISION"

    round_content = f"""## Round 5: Final Selection (from 5 Proposals)

{selection_content}

---

## Conclusion

- Status: COMPLETED
- Consensus Reached: {consensus_reached}
- Consensus Reason: {consensus_reason or 'N/A'}
- Article Statuses: {status_summary}

"""
    _append_to_transcript(transcript_path, round_content)
    (debate_outputs / "final_selection_round5.md").write_text(selection_content, encoding='utf-8')

    # Also save as final_rank1_proposal.md for ProposalAgent compatibility
    (debate_outputs / "final_rank1_proposal.md").write_text(selection_content, encoding='utf-8')

    logger.info(f"[ROUND 5] Complete - Final selection from 5 proposals saved (consensus: {consensus_reached})")

    return {
        "build_debate_round": 5,
        "build_debate_consensus_reached": consensus_reached,
        "build_debate_consensus_reason": consensus_reason,
        "processing_logs": [f"[ROUND 5] Final selection from 5 proposals complete (consensus: {consensus_reached})"],
    }


# ==============================================================================
# ROUTING FUNCTIONS
# ==============================================================================

def route_after_setup_workspace(state: MARBLEState) -> str:
    """Route after setup_workspace based on iteration number.

    - Iteration 1: paper_reader → weakness_analysis (analyze target model from scratch)
    - Iteration 2+: iteration_critic (analyze previous iteration results)
    """
    current_iteration = state.get("current_iteration", 1)

    if current_iteration <= 1:
        logger.info(f"[ROUTING] Iteration {current_iteration}: Going to paper_reader (first iteration)")
        return "paper_reader"
    else:
        logger.info(f"[ROUTING] Iteration {current_iteration}: Going to iteration_critic (analyzing previous iteration)")
        return "iteration_critic"


def route_after_iteration_critic(state: MARBLEState) -> str:
    """Route after iteration_critic - always go to paper_aggregator for embedding scoring.

    NOTE: Online paper search (PMC/OpenReview) has been removed.
    EmbeddingScorer always uses pre-downloaded local PDFs, so skip_paper_search flag
    is no longer relevant. Always route to paper_search_start → paper_aggregator.
    """
    should_terminate = state.get("should_terminate", False)

    if should_terminate:
        logger.warning("[ROUTING] should_terminate=True - Going to END (beta <= 0)")
        return "end"

    # Always use paper_aggregator with EmbeddingScorer (local PDFs)
    logger.info("[ROUTING] Going to paper_search_start → paper_aggregator (EmbeddingScorer)")
    return "paper_search"


def route_after_paper_selector(state: MARBLEState) -> str:
    """Route after paper_selector based on fallback_to_paper_search flag.

    - fallback_to_paper_search=True: Aggregated results not found, do new paper search
    - fallback_to_paper_search=False: Papers selected successfully, continue to readers
    """
    fallback_to_paper_search = state.get("fallback_to_paper_search", False)

    if fallback_to_paper_search:
        logger.warning("[ROUTING] fallback_to_paper_search=True - Going to paper_search_start")
        return "paper_search"
    else:
        logger.info("[ROUTING] Paper selection successful - Going to other_paper_reader_1")
        return "paper_reader"


# ==============================================================================
# GRAPH BUILDER
# ==============================================================================

def create_build_debate_subgraph() -> Tuple[Any, Any]:
    """Create the build debate subgraph with 5-round debate system.

    Complete Workflow:
    Phase 1: Model Understanding (iteration-dependent)
    0. setup_workspace → Copy docker_images to experiments/build (build_0 + best 변경사항)

    For Iteration 1:
    1a. paper_reader (ModelResearcherAgent) → {model}_summary.md
    1b. weakness_analysis (CriticAgent) → weakness_of_target_model.md

    For Iteration 2+:
    1c. iteration_critic → weakness_of_target_model.md (analyzes best iteration results)

    Phase 2: Paper Scoring & Selection (uses EmbeddingScorer with local PDFs)
    2a. paper_search_start → paper_aggregator (EmbeddingScorer로 로컬 PDF 스코어링)
    OR
    2b. paper_selector → paper_reader_start (이전 aggregated_results에서 선택)

    → paper_reader_start (fan-out)
    3a. other_paper_reader_1 ───┐
    3b. other_paper_reader_2 ───┤
    3c. other_paper_reader_3 ───┼→ (parallel) → other_paper_*_summary.md
    3d. other_paper_reader_4 ───┤
    3e. other_paper_reader_5 ───┘

    Phase 3: 5-Round Debate
    5. init_debate → Initialize debate_transcript.md
    6. debate_round_1 → Article experts propose
    7. debate_round_2 → Critic feedback
    8. debate_round_3 → Article experts respond/revise (FULL_ADOPTION possible)
    9. debate_round_4 → Model ranking + Critic critique
    10. debate_round_5 → Final selection

    Phase 4: Implementation Proposal
    11. proposal_agent → implementation_proposal.md → END
    """
    builder = StateGraph(MARBLEState)
    checkpointer = InMemorySaver()

    # Phase 1: Model Understanding (with iteration-dependent branching)
    builder.add_node("setup_workspace", setup_build_workspace_node)
    builder.add_node("paper_reader", partial(paper_reader_node, checkpointer=checkpointer))
    builder.add_node("weakness_analysis", partial(weakness_analysis_node, checkpointer=checkpointer))
    builder.add_node("iteration_critic", partial(iteration_critic_node, checkpointer=checkpointer))

    # Intermediate node for paper scoring (previously parallel fan-out, now direct to aggregator)
    def paper_search_start_node(state: MARBLEState) -> Dict[str, Any]:
        """Passthrough node for paper scoring workflow (uses local PDFs, no online search)."""
        logger.info("[PAPER_SEARCH_START] Proceeding to EmbeddingScorer (local PDFs)")
        return {"processing_logs": ["[PAPER_SEARCH_START] Starting local PDF scoring"]}

    builder.add_node("paper_search_start", paper_search_start_node)

    # Intermediate node for parallel paper reader fan-out
    def paper_reader_start_node(state: MARBLEState) -> Dict[str, Any]:
        """Passthrough node to enable parallel fan-out to all 5 paper readers."""
        logger.info("[PAPER_READER_START] Starting parallel paper reading (Papers 1-5)")
        return {"processing_logs": ["[PAPER_READER_START] Triggering parallel paper reading (5 papers)"]}

    builder.add_node("paper_reader_start", paper_reader_start_node)

    # Phase 2: Paper Scoring & Selection (uses EmbeddingScorer with local PDFs)
    # NOTE: pmc_researcher and openreview_researcher removed (online search disabled)
    builder.add_node("paper_aggregator", partial(paper_aggregator_node, checkpointer=checkpointer))
    builder.add_node("paper_selector", partial(paper_selector_node, checkpointer=checkpointer))
    builder.add_node("other_paper_reader_1", partial(other_paper_reader_1_node, checkpointer=checkpointer))
    builder.add_node("other_paper_reader_2", partial(other_paper_reader_2_node, checkpointer=checkpointer))
    builder.add_node("other_paper_reader_3", partial(other_paper_reader_3_node, checkpointer=checkpointer))
    builder.add_node("other_paper_reader_4", partial(other_paper_reader_4_node, checkpointer=checkpointer))
    builder.add_node("other_paper_reader_5", partial(other_paper_reader_5_node, checkpointer=checkpointer))

    # Phase 3: 5-Round Debate
    builder.add_node("init_debate", init_debate_node)
    builder.add_node("debate_round_1", partial(debate_round_1_node, checkpointer=checkpointer))
    builder.add_node("debate_round_2", partial(debate_round_2_node, checkpointer=checkpointer))
    builder.add_node("debate_round_3", partial(debate_round_3_node, checkpointer=checkpointer))
    builder.add_node("debate_round_4", partial(debate_round_4_node, checkpointer=checkpointer))
    builder.add_node("debate_round_5", partial(debate_round_5_node, checkpointer=checkpointer))

    # Phase 4: Implementation Proposal
    builder.add_node("proposal_agent", partial(proposal_agent_node, checkpointer=checkpointer))

    # Entry point
    builder.set_entry_point("setup_workspace")

    # Phase 1: Conditional routing after setup_workspace
    # - Iteration 1: paper_reader → weakness_analysis
    # - Iteration 2+: iteration_critic (analyzes previous iteration results)
    builder.add_conditional_edges(
        "setup_workspace",
        route_after_setup_workspace,
        {
            "paper_reader": "paper_reader",
            "iteration_critic": "iteration_critic",
        }
    )

    # Iteration 1 path: paper_reader → weakness_analysis → paper_search_start → parallel search
    builder.add_edge("paper_reader", "weakness_analysis")
    builder.add_edge("weakness_analysis", "paper_search_start")

    # Iteration 2+ path: iteration_critic → conditional routing
    # - skip_paper_search=False: paper_search_start → parallel search
    # - skip_paper_search=True: paper_selector
    # - should_terminate=True: END
    builder.add_conditional_edges(
        "iteration_critic",
        route_after_iteration_critic,
        {
            "paper_search": "paper_search_start",
            "paper_selector": "paper_selector",
            "end": END,
        }
    )

    # NEW: Direct to paper_aggregator (uses pre-downloaded PDFs with EmbeddingScorer)
    # REMOVED: pmc_researcher and openreview_researcher (online search disabled)
    builder.add_edge("paper_search_start", "paper_aggregator")

    # paper_selector: conditional routing based on fallback_to_paper_search
    # - fallback_to_paper_search=True: Go to paper_search_start (new search)
    # - fallback_to_paper_search=False: Go to paper_reader_start (parallel readers)
    builder.add_conditional_edges(
        "paper_selector",
        route_after_paper_selector,
        {
            "paper_search": "paper_search_start",
            "paper_reader": "paper_reader_start",
        }
    )

    # Continue after aggregation → paper_reader_start (fan-out node)
    builder.add_edge("paper_aggregator", "paper_reader_start")

    # PARALLEL paper readers: paper_reader_start → all 5 readers
    builder.add_edge("paper_reader_start", "other_paper_reader_1")
    builder.add_edge("paper_reader_start", "other_paper_reader_2")
    builder.add_edge("paper_reader_start", "other_paper_reader_3")
    builder.add_edge("paper_reader_start", "other_paper_reader_4")
    builder.add_edge("paper_reader_start", "other_paper_reader_5")

    # Fan-in: all 5 readers → init_debate (LangGraph waits for all parallel branches)
    builder.add_edge("other_paper_reader_1", "init_debate")
    builder.add_edge("other_paper_reader_2", "init_debate")
    builder.add_edge("other_paper_reader_3", "init_debate")
    builder.add_edge("other_paper_reader_4", "init_debate")
    builder.add_edge("other_paper_reader_5", "init_debate")
    builder.add_edge("init_debate", "debate_round_1")
    builder.add_edge("debate_round_1", "debate_round_2")
    builder.add_edge("debate_round_2", "debate_round_3")
    builder.add_edge("debate_round_3", "debate_round_4")
    builder.add_edge("debate_round_4", "debate_round_5")

    # Phase 4: Implementation Proposal
    builder.add_edge("debate_round_5", "proposal_agent")
    builder.add_edge("proposal_agent", END)

    # Compile
    subgraph = builder.compile(checkpointer=checkpointer).with_config(
        recursion_limit=LANGGRAPH_CONFIG["recursion_limit"]
    )

    logger.info("[BUILD_DEBATE] Subgraph created with weight adjustment routing (iter1: paper_reader, iter2+: iteration_critic with skip_paper_search support)")
    return subgraph, checkpointer


def get_build_debate_subgraph():
    """Factory function returning compiled subgraph."""
    global _BUILD_DEBATE_SUBGRAPH
    if _BUILD_DEBATE_SUBGRAPH is None:
        _BUILD_DEBATE_SUBGRAPH = create_build_debate_subgraph()
    return _BUILD_DEBATE_SUBGRAPH
