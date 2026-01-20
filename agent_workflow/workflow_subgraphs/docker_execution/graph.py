"""Docker Execution Subgraph.

Runs Docker-based testing for build workflow outputs.
Flow: docker_test → (pass) → END
                 → (fail) → code_expert_fix → docker_test (retry)

Input: From build_development_workflow
  - target_model: Model name (e.g., "stagate")
  - Workspace at experiments/build/

Output:
  - docker_test_success: bool
  - docker_test_error: Optional error message
"""

import asyncio
import getpass
import os
import subprocess
import threading
from functools import partial
from pathlib import Path
from typing import Any, Dict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from configs.config import LANGGRAPH_CONFIG
from agent_workflow.logger import logger
from agent_workflow.state import MARBLEState
from agent_workflow.utils import get_project_root, get_free_gpu

# Module-level cache
_DOCKER_EXECUTION_SUBGRAPH = None

# Constants
MAX_DOCKER_TEST_ROUNDS = 10
CODE_EXPERT_TIMEOUT = 600  # Timeout in seconds for CodeExpertAgent (10 minutes)


# ==============================================================================
# PATH HELPERS
# ==============================================================================

MODEL_WORKSPACE_CONFIG = {
    "stagate": {
        "workspace": "experiments/build",
        "src_dir": "src",  # where main.py is
        "main_script": "main.py",
    },
    "deeptta": {
        "workspace": "experiments/build",
        "src_dir": "src",  # same as stagate now (flat structure)
        "main_script": "main.py",
    },
    "deepst": {
        "workspace": "experiments/build",
        "src_dir": "src",
        "main_script": "main.py",
    },
    "hyperattentiondti": {
        "workspace": "experiments/build",
        "src_dir": "src",
        "main_script": "main.py",
    },
    "dlm-dti": {
        "workspace": "experiments/build",
        "src_dir": "src",
        "main_script": "main.py",
    },
    "deepdr": {
        "workspace": "experiments/build",
        "src_dir": "src",
        "main_script": "main.py",
    },
}


def _get_workspace_path(target_model: str, iteration: int = 1) -> Path:
    """Get workspace path for target model and iteration.

    Each iteration has its own workspace: experiments/build_{iteration}/
    """
    project_root = Path(get_project_root())
    return project_root / "experiments" / f"build_{iteration}"


def _get_src_dir(target_model: str, iteration: int = 1) -> str:
    """Get source directory path for target model and iteration."""
    workspace = _get_workspace_path(target_model, iteration)
    config = MODEL_WORKSPACE_CONFIG.get(target_model, MODEL_WORKSPACE_CONFIG["stagate"])
    return str(workspace / config.get("src_dir", "src"))


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _run_with_realtime_output(cmd: list, phase_name: str = "") -> tuple:
    """Run command with real-time output streaming (no timeout)."""
    stdout_lines = []
    stderr_lines = []

    def read_stream(stream, output_list, prefix=""):
        for line in iter(stream.readline, ''):
            if line:
                output_list.append(line)
                print(f"{prefix}{line}", end='', flush=True)
        stream.close()

    logger.info(f"[DOCKER] Starting: {phase_name}")
    print(f"\n{'='*60}\n[DOCKER] {phase_name}\n{'='*60}", flush=True)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, ""))
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, "[ERR] "))

    stdout_thread.start()
    stderr_thread.start()

    process.wait()

    stdout_thread.join()
    stderr_thread.join()

    print(f"{'='*60}\n[DOCKER] {phase_name} finished (code={process.returncode})\n", flush=True)

    return process.returncode, ''.join(stdout_lines), ''.join(stderr_lines)


def _extract_metrics_from_output(output: str) -> Dict[str, Any]:
    """Docker 출력에서 성능 메트릭 추출.

    Args:
        output: Docker training stdout 출력

    Returns:
        추출된 메트릭 딕셔너리

    지원 모델별 메트릭:
        - Drug Response (DeepTTA, DRPreter): rmse, mse, mae, pearson, spearman
        - DTI (DrugBAN, HyperAttentionDTI, DLM-DTI): auroc, auprc
        - Spatial (DeepST, STAGATE): ari, nmi, silhouette
    """
    import re
    import ast

    # 메트릭 이름 정규화 매핑 (Docker 출력 → 표준 이름)
    # improved 판단을 위해 baseline과 동일한 메트릭 이름 사용
    METRIC_NAME_NORMALIZE = {
        'auc_mean': 'auroc',
        'auc': 'auroc',
        'aupr_mean': 'auprc',
        'aupr': 'auprc',
        'pcc': 'pearson',
        'scc': 'spearman',
    }

    metrics = {}

    # === Step 1: Results: {...} 형식 직접 파싱 (최우선) ===
    # 예: Results: {'accuracy': 0.787, 'auroc': 0.625, 'precision': 0.619, 'f1': 0.693, 'auprc': 0.365}
    results_match = re.search(r'Results:\s*(\{[^}]+\})', output)
    if results_match:
        try:
            results_str = results_match.group(1)
            parsed = ast.literal_eval(results_str)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, (int, float)):
                        # 메트릭 이름 정규화 (auc_mean → auroc 등)
                        normalized_key = METRIC_NAME_NORMALIZE.get(k, k)
                        metrics[normalized_key] = float(v)
                logger.info(f"[DOCKER_TEST] Results dict 직접 파싱 성공: {metrics}")
        except Exception as e:
            logger.warning(f"[DOCKER_TEST] Results dict 파싱 실패: {e}")

    # 이미 메트릭을 찾았으면 바로 반환
    if metrics:
        return metrics

    # === Step 2: 기존 패턴 매칭 (fallback) ===
    # Preprocess: convert np.float64(x) -> x for all numpy types
    output = re.sub(r'np\.float64\(([0-9.e+-]+)\)', r'\1', output)
    output = re.sub(r'np\.float32\(([0-9.e+-]+)\)', r'\1', output)
    output = re.sub(r'np\.int64\(([0-9]+)\)', r'\1', output)
    output = re.sub(r'np\.int32\(([0-9]+)\)', r'\1', output)

    # Preprocess: convert dict format 'metric': value -> metric: value
    output = re.sub(r"'(\w+)':\s*([0-9.e+-]+)", r'\1: \2', output)

    # 모델별 메트릭 패턴들 (대소문자 무시)
    patterns = {
        # === Drug Response 메트릭 ===
        'rmse': [
            r'RMSE[:\s=]+([0-9.]+)',
            r'rmse[:\s=]+([0-9.]+)',
            r'Root Mean Square Error[:\s=]+([0-9.]+)',
        ],
        'mse': [
            r'MSE[:\s=]+([0-9.]+)',
            r'mse[:\s=]+([0-9.]+)',
            r'Mean Square Error[:\s=]+([0-9.]+)',
        ],
        'mae': [
            r'MAE[:\s=]+([0-9.]+)',
            r'mae[:\s=]+([0-9.]+)',
            r'Mean Absolute Error[:\s=]+([0-9.]+)',
        ],
        'pearson': [
            r'Pearson[:\s=]+([0-9.]+)',
            r'pearson[:\s=]+([0-9.]+)',
            r'PCC[:\s=]+([0-9.]+)',
            r'Pearson Correlation[:\s=]+([0-9.]+)',
        ],
        'spearman': [
            r'Spearman[:\s=]+([0-9.]+)',
            r'spearman[:\s=]+([0-9.]+)',
            r'SCC[:\s=]+([0-9.]+)',
        ],
        'r2': [
            r'R2[:\s=]+([0-9.]+)',
            r'r2[:\s=]+([0-9.]+)',
            r'R\^2[:\s=]+([0-9.]+)',
        ],

        # === DTI 메트릭 ===
        'auroc': [
            r'AUROC[:\s=]+([0-9.]+)',
            r'auroc[:\s=]+([0-9.]+)',
            r'AUC[:\s=]+([0-9.]+)',
            r'Test AUROC[:\s=]+([0-9.]+)',
            r'Val AUC[:\s=]+([0-9.]+)',
        ],
        'auprc': [
            r'AUPRC[:\s=]+([0-9.]+)',
            r'auprc[:\s=]+([0-9.]+)',
            r'Test AUPRC[:\s=]+([0-9.]+)',
        ],

        # === Spatial 메트릭 ===
        'ari': [
            r'ARI[:\s=]+([0-9.]+)',
            r'ari[:\s=]+([0-9.]+)',
            r'Adjusted Rand Index[:\s=]+([0-9.]+)',
        ],
        'nmi': [
            r'NMI[:\s=]+([0-9.]+)',
            r'nmi[:\s=]+([0-9.]+)',
            r'Normalized Mutual Info[:\s=]+([0-9.]+)',
        ],
        'silhouette': [
            r'Silhouette[:\s=]+([0-9.]+)',
            r'silhouette[:\s=]+([0-9.]+)',
        ],

        # === 공통 메트릭 ===
        'accuracy': [
            r'Accuracy[:\s=]+([0-9.]+)',
            r'accuracy[:\s=]+([0-9.]+)',
            r'Acc[:\s=]+([0-9.]+)',
        ],
        'precision': [
            r'Precision[:\s=]+([0-9.]+)',
            r'precision[:\s=]+([0-9.]+)',
        ],
        'f1': [
            r'F1[:\s=]+([0-9.]+)',
            r'f1[:\s=]+([0-9.]+)',
            r'F1[-_]?[Ss]core[:\s=]+([0-9.]+)',
        ],
        'loss': [
            r'(?:Final |Best |Test )?Loss[:\s=]+([0-9.]+)',
            r'loss[:\s=]+([0-9.]+)',
        ],
    }

    for metric_name, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    # 마지막 매칭값 사용 (최종 결과일 가능성 높음)
                    metrics[metric_name] = float(matches[-1])
                    break
                except ValueError:
                    continue

    logger.info(f"[DOCKER_TEST] 추출된 메트릭: {metrics}")
    return metrics


def _write_error_to_memory(workspace: Path, error_msg: str, error_type: str, round_num: int):
    """Write error to memory file for code expert to read."""
    memory_file = workspace / "docker_error_memory.md"
    content = f"""# Docker Test Error - Round {round_num + 1}

## Error Type
{error_type}

## Error Message
```
{error_msg}
```

## Instructions for Code Expert
Fix the error above. Focus on:
1. Import errors → check module paths and dependencies
2. Syntax errors → fix Python syntax
3. Runtime errors → check tensor shapes, device placement, missing arguments
"""
    memory_file.write_text(content, encoding='utf-8')
    logger.info(f"[DOCKER_TEST] Error written to {memory_file}")


# ==============================================================================
# NODES
# ==============================================================================

def docker_test_node(state: MARBLEState) -> Dict[str, Any]:
    """Run Docker to test the code (no build, direct execution)."""
    target_model = state.get("target_model", "stagate")
    current_iteration = state.get("current_iteration", 1)
    docker_test_round = state.get("docker_test_round", 0)

    workspace = _get_workspace_path(target_model, current_iteration)
    project_root = get_project_root()

    logger.info(f"[DOCKER_TEST] Iteration {current_iteration}, Round {docker_test_round + 1}/{MAX_DOCKER_TEST_ROUNDS}")
    logger.info(f"[DOCKER_TEST] Workspace: {workspace}")

    # Pre-built image name (from docker_images/build.sh)
    current_user = getpass.getuser()
    image_name = f"{target_model}-develop:MARBLE_{current_user}"

    # Dynamic main.py path based on iteration
    workspace_container = f"/workspace/experiments/build_{current_iteration}"
    main_script_path = f"{workspace_container}/src/main.py"
    model_app_paths = {
        "deeptta": "/app/deeptta_models",
        "stagate": "/app/stagate_models",
        "deepst": "/app/deepst_models",
        "dlm-dti": "/app/dlm-dti_models",
        "hyperattentiondti": "/app/hyperattentiondti_models",
        "deepdr": "/app/deepdr_models",
    }
    model_path = model_app_paths.get(target_model, f"/app/{target_model}_models")
    pythonpath = f"{workspace_container}:{workspace_container}/src:/app:/app/components:{model_path}"

    try:
        # ==================================================================
        # Run training directly (no build step)
        # ==================================================================
        # Select free GPU dynamically
        gpu_id = get_free_gpu()
        logger.info(f"[DOCKER_TEST] Using GPU {gpu_id}")

        returncode, stdout, stderr = _run_with_realtime_output(
            ["docker", "run", "--rm", "--gpus", "all",
             "--shm-size=12g",
             "--user", f"{os.getuid()}:{os.getgid()}",
             "-e", f"CUDA_VISIBLE_DEVICES={gpu_id}",
             "-e", f"PYTHONPATH={pythonpath}",
             "-w", workspace_container,
             "-v", f"{project_root}:/workspace",
             image_name,
             "python", main_script_path],
            phase_name="Training"
        )

        if returncode != 0:
            error_msg = stderr[-2000:] if stderr else stdout[-2000:]
            _write_error_to_memory(workspace, error_msg, "Training Error", docker_test_round)
            return {
                "docker_test_success": False,
                "docker_test_error": error_msg,
                "docker_test_round": docker_test_round,
                "processing_logs": ["[DOCKER_TEST] Training failed"],
            }

        # Success! 메트릭 추출 (stdout + stderr 모두 확인)
        logger.info("[DOCKER_TEST] Training passed!")
        combined_output = (stdout or "") + "\n" + (stderr or "")
        metrics = _extract_metrics_from_output(combined_output)

        return {
            "docker_test_success": True,
            "docker_test_error": None,
            "docker_test_round": docker_test_round,
            "docker_test_output": stdout[-5000:] if stdout else "",  # 마지막 5000자 저장
            "iteration_metrics": metrics,  # 추출된 메트릭
            "processing_logs": ["[DOCKER_TEST] Success!"],
        }

    except Exception as e:
        return {
            "docker_test_success": False,
            "docker_test_error": str(e),
            "docker_test_round": docker_test_round,
            "processing_logs": [f"[DOCKER_TEST] Exception: {e}"],
        }


async def code_expert_fix_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Code Expert fixes Docker test errors."""
    from langchain_core.messages import HumanMessage

    target_model = state.get("target_model", "stagate")
    current_iteration = state.get("current_iteration", 1)
    workspace = _get_workspace_path(target_model, current_iteration)
    error_memory = workspace / "docker_error_memory.md"

    logger.info(f"[CODE_EXPERT_FIX] Reading error and attempting fix...")

    # Read error memory
    error_content = ""
    if error_memory.exists():
        error_content = error_memory.read_text(encoding='utf-8')

    # Import CodeExpertAgent from build_development_workflow
    from agent_workflow.workflow_subgraphs.build_development_workflow.agents import CodeExpertAgent

    # CodeExpertAgent expects: proposal_path, target_component, template_path, validator_feedback, is_fix_iteration
    proposal_path = str(workspace / "build_debate_outputs" / "implementation_proposal.md")
    template_path = str(workspace / "src" / "model.py")

    code_expert = CodeExpertAgent(
        proposal_path=proposal_path,
        target_component="model",  # Generic component for docker fix
        template_path=template_path,
        validator_feedback=error_content,
        is_fix_iteration=True,
        checkpointer=checkpointer
    )
    compiled_agent = code_expert.create_agent()

    prompt = f"""Fix the Docker test error below.

## Error Details
{error_content}

## Workspace
{workspace}

## Instructions
1. Read the error carefully
2. Identify the problematic file(s) in {workspace}/src/
3. Fix the issue
4. Use write_file to save the fixed code

Focus on:
- Import errors: check module paths
- Syntax errors: fix Python syntax
- Runtime errors: tensor shapes, device issues, missing arguments
"""

    try:
        result = await asyncio.wait_for(
            compiled_agent.ainvoke({
                **state,
                "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
            }),
            timeout=CODE_EXPERT_TIMEOUT
        )
        return {
            "messages": result.get("messages", []),
            "processing_logs": [f"[CODE_EXPERT_FIX] Attempted to fix error"],
        }
    except asyncio.TimeoutError:
        logger.warning(f"[CODE_EXPERT_FIX] Timeout after {CODE_EXPERT_TIMEOUT}s, moving to next node")
        return {
            "processing_logs": [f"[CODE_EXPERT_FIX] TIMEOUT after {CODE_EXPERT_TIMEOUT}s"],
        }


def increment_docker_round(state: MARBLEState) -> Dict[str, Any]:
    """Increment docker test round counter."""
    new_round = state.get("docker_test_round", 0) + 1
    logger.info(f"[INCREMENT_ROUND] Round {new_round}")
    return {
        "docker_test_round": new_round,
    }


# ==============================================================================
# ROUTING
# ==============================================================================

def route_after_docker_test(state: MARBLEState) -> str:
    """Route after docker test."""
    docker_test_success = state.get("docker_test_success", False)
    docker_test_round = state.get("docker_test_round", 0)

    if docker_test_success:
        logger.info("[DOCKER_TEST] Success → END")
        return END

    if docker_test_round >= MAX_DOCKER_TEST_ROUNDS:
        logger.warning(f"[DOCKER_TEST] Max rounds ({MAX_DOCKER_TEST_ROUNDS}) reached → END")
        return END

    logger.info(f"[DOCKER_TEST] Failed, round {docker_test_round + 1} → code_expert_fix")
    return "code_expert_fix"


# ==============================================================================
# GRAPH BUILDER
# ==============================================================================

def get_docker_execution_subgraph(checkpointer=None):
    """Build and return the docker execution subgraph.

    Flow:
        docker_test → (success) → END
                   → (fail) → code_expert_fix → increment_round → docker_test
    """
    global _DOCKER_EXECUTION_SUBGRAPH

    if _DOCKER_EXECUTION_SUBGRAPH is not None:
        return _DOCKER_EXECUTION_SUBGRAPH

    if checkpointer is None:
        checkpointer = InMemorySaver()

    builder = StateGraph(MARBLEState)

    # Add nodes
    builder.add_node("docker_test", docker_test_node)
    builder.add_node("code_expert_fix", partial(code_expert_fix_node, checkpointer=checkpointer))
    builder.add_node("increment_round", increment_docker_round)

    # Entry point
    builder.set_entry_point("docker_test")

    # Edges
    builder.add_conditional_edges(
        "docker_test",
        route_after_docker_test,
        {
            "code_expert_fix": "code_expert_fix",
            END: END,
        }
    )
    builder.add_edge("code_expert_fix", "increment_round")
    builder.add_edge("increment_round", "docker_test")

    # Compile
    _DOCKER_EXECUTION_SUBGRAPH = builder.compile(checkpointer=checkpointer).with_config(
        recursion_limit=LANGGRAPH_CONFIG["recursion_limit"]
    )

    logger.info("[DOCKER_EXECUTION] Subgraph compiled")
    return _DOCKER_EXECUTION_SUBGRAPH
