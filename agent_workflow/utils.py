import asyncio
import getpass
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

# Environment variable for debug logging
DEBUG_LOGS = os.getenv("AUTODRP_DEBUG_LOGS", "false").lower() == "true"
COMPILE_MODE = "compile" in " ".join(sys.argv)

# Helper function for conditional logging
def debug_print(*args, **kwargs):
    """Print only if debug logging is enabled and not in compile mode."""
    if DEBUG_LOGS and not COMPILE_MODE:
        print(*args, **kwargs)


# ==============================================================================
# GPU UTILITIES
# ==============================================================================

def get_free_gpu(fallback: str = "0") -> str:
    """Find the GPU with lowest memory usage.

    Queries nvidia-smi to find all available GPUs and returns the one
    with the least memory currently in use.

    Args:
        fallback: GPU ID to return if detection fails (default: "0")

    Returns:
        GPU ID as string (e.g., "0", "1", "2", "3")

    Example:
        >>> gpu_id = get_free_gpu()
        >>> device = f"cuda:{gpu_id}"
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            debug_print(f"[GPU] nvidia-smi failed, using GPU {fallback}")
            return fallback

        # Parse output: "0, 1234\n1, 5678\n..."
        gpu_memory = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                if len(parts) == 2:
                    gpu_id = parts[0].strip()
                    mem_used = int(parts[1].strip())
                    gpu_memory.append((gpu_id, mem_used))

        if not gpu_memory:
            debug_print(f"[GPU] No GPU found, using GPU {fallback}")
            return fallback

        # Sort by memory usage (ascending) and pick the one with lowest usage
        gpu_memory.sort(key=lambda x: x[1])
        selected_gpu = gpu_memory[0][0]

        debug_print(f"[GPU] Selected GPU {selected_gpu} ({gpu_memory[0][1]} MiB used)")
        return selected_gpu

    except Exception as e:
        debug_print(f"[GPU] Error: {e}, using GPU {fallback}")
        return fallback


def get_free_gpu_device() -> str:
    """Get PyTorch device string for the freest GPU.

    Returns:
        Device string like "cuda:3" or "cpu" if no GPU available

    Example:
        >>> device = get_free_gpu_device()
        >>> model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_id = get_free_gpu()
            return f"cuda:{gpu_id}"
        return "cpu"
    except ImportError:
        return "cpu"


def get_model_source_path(model_name: str, project_root: str = "/workspace") -> str:
    """
    Get the actual source path for a model.

    Each model has its own directory with the same name as the model.

    Args:
        model_name: Model identifier (deeptta, deepdr, stagate, deepst, dlm-dti, hyperattentiondti)
        project_root: Project root path (default: /workspace)

    Returns:
        str: Full path to model source directory

    Examples:
        >>> get_model_source_path("deeptta")
        '/workspace/experiments/experiments/deeptta'
        >>> get_model_source_path("deepdr")
        '/workspace/experiments/experiments/deepdr'
    """
    # Each model uses its own directory name
    model_dir = model_name.lower()
    return f"{project_root}/experiments/experiments/{model_dir}"


def get_user_id() -> str:
    """
    동적으로 USER_ID를 감지합니다.

    우선순위:
    1. 환경변수 USER_ID
    2. 현재 시스템 사용자명 (getpass.getuser())
    3. 홈 디렉토리에서 추출
    4. 기본값 "default_user"

    Returns:
        str: 감지된 USER_ID
    """
    # 1순위: 환경변수
    if user_id := os.getenv('USER_ID'):
        user_id = user_id.strip()
        if user_id:
            return user_id

    # 2순위: 현재 시스템 사용자명
    try:
        if user_id := getpass.getuser():
            return user_id
    except Exception:
        pass

    # 3순위: 홈 디렉토리에서 추출
    try:
        home_path = os.path.expanduser('~')
        if user_id := os.path.basename(home_path):
            if user_id and user_id != '~':  # 유효한 사용자명인지 확인
                return user_id
    except Exception:
        pass

    # 4순위: 기본값
    return "default_user"


def get_project_root() -> str:
    """
    프로젝트 루트 경로를 반환합니다.

    우선순위:
    1. 환경변수 PROJECT_ROOT (커스터마이징 필요시)
    2. /data1/project/{username}/MARBLE (기본 패턴)

    Returns:
        str: 프로젝트 루트 절대 경로
    """
    # 1순위: 환경변수로 오버라이드 가능
    if project_root := os.getenv('PROJECT_ROOT'):
        return project_root.strip()

    # 2순위 (기본): /data1/project/{username}/MARBLE 패턴
    username = get_user_id()
    return f"/data1/project/{username}/MARBLE"


def add_processing_log(state: Dict[str, Any], log_message: str) -> Dict[str, Any]:
    """
    Utility function to add processing log entries to state.

    Args:
        state: Current state dictionary
        log_message: Log message to add

    Returns:
        Updated state with new log entry
    """
    return {
        "processing_logs": state.get("processing_logs", []) + [log_message]
    }


def add_multiple_processing_logs(state: Dict[str, Any], log_messages: List[str]) -> Dict[str, Any]:
    """
    Utility function to add multiple processing log entries to state.

    Args:
        state: Current state dictionary
        log_messages: List of log messages to add

    Returns:
        Updated state with new log entries
    """
    return {
        "processing_logs": state.get("processing_logs", []) + log_messages
    }


def save_debate_history_entry(
    state: Dict[str, Any],
    speaker: str,
    content: str,
    max_preview_chars: int = 1500,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist long-form debate content to disk and return a lightweight history entry.

    Stores all debate turns in a single accumulated file: experiments/reports/<target>/debate_transcript.md
    Each turn is appended with speaker identification and turn number in sequential order.

    Note: Deduplication is now handled automatically by the add_debate_history_with_deduplication
    reducer in state.py, so this function always returns an entry.
    """
    turn_number = state.get("turn_count", 0) + 1
    target_model = state.get("target_model") or "general"

    # Single accumulated file path
    transcript_dir = Path(f"experiments/reports/{target_model}")
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / "debate_transcript.md"

    session_id = state.get("debate_session_id") or "unknown"

    preview = content.strip()
    if len(preview) > max_preview_chars:
        preview = preview[:max_preview_chars].rstrip() + "…"

    entry: Dict[str, Any] = {
        "turn": turn_number,
        "speaker": speaker,
        "content": preview,
        "content_path": str(transcript_path),
        "timestamp": state.get("turn_count", 0)
    }

    if session_id:
        entry["session_id"] = session_id

    if metadata:
        entry.update(metadata)

    # Append full content to single accumulated file
    with open(transcript_path, "a", encoding="utf-8") as f:
        f.write("\n---\n")
        f.write(f"# Turn {turn_number} - {speaker}\n")
        f.write(f"**Session**: {session_id}\n\n")
        f.write(content)
        f.write("\n\n")

    return entry


def get_current_debate_history(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return debate history filtered for the active debate session."""
    history = state.get("debate_history", []) or []
    session_id = state.get("debate_session_id")

    if not session_id:
        return history

    filtered = [entry for entry in history if entry.get("session_id") == session_id]
    return filtered if filtered else history


def build_session_log_update(state: Dict[str, Any], *messages: str) -> Dict[str, Dict[str, List[str]]]:
    """Create update payload appending session-specific log messages."""
    session_id = state.get("debate_session_id")
    if not session_id or not messages:
        return {}

    existing_logs = state.get("debate_session_logs", {}).get(session_id, [])
    new_logs = list(existing_logs)
    for message in messages:
        if new_logs and new_logs[-1] == message:
            continue
        if message in new_logs:
            continue
        new_logs.append(message)

    if new_logs == list(existing_logs):
        return {}

    return {"debate_session_logs": {session_id: new_logs}}


class DynamicMCPConfig:
    """
    사용자별 MCP 컨테이너 이름을 동적으로 생성하는 설정 클래스.
    
    각 사용자가 자신의 MCP 컨테이너를 사용할 수 있도록 
    USER_ID를 기반으로한 컨테이너 이름을 생성합니다.
    """
    
    def __init__(self):
        self.user_id = get_user_id()
        # Load MCP configuration to check enabled status
        self._load_mcp_config()
    
    @property
    def SEQUENTIAL_MCP(self) -> str:
        """Sequential thinking MCP 서버 컨테이너 이름"""
        return f"mcp-sequential_{self.user_id}"
    
    @property
    def DESKTOP_MCP(self) -> str:
        """Desktop commander MCP 서버 컨테이너 이름"""
        return f"mcp-desktop-commander_{self.user_id}"
    
    @property
    def CONTEXT7_MCP(self) -> str:
        """Context7 MCP 서버 컨테이너 이름"""
        return f"mcp-context7_{self.user_id}"
    
    @property
    def SERENA_MCP(self) -> str:
        """Serena MCP 서버 컨테이너 이름"""
        return f"mcp-serena_{self.user_id}"
    
    @property
    def ARXIV_SEMANTIC_MCP(self) -> str:
        """ArXiv Semantic MCP 서버 컨테이너 이름"""
        return f"mcp-arxiv-semantic_{self.user_id}"

    @property
    def DRP_VIS_MCP(self) -> str:
        """DRP Visualization MCP 서버 컨테이너 이름"""
        return f"drp-vis-mcp_{self.user_id}"

    def _load_mcp_config(self):
        """Load MCP configuration to check which servers are enabled"""
        import json
        import os
        self.enabled_servers = {}
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mcp.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
                for server_name, server_config in config.get('servers', {}).items():
                    self.enabled_servers[server_name] = server_config.get('enabled', False)
        except Exception:
            # If config can't be loaded, assume all are enabled (except disabled ones)
            self.enabled_servers = {
                'mcp-sequential': True,
                'mcp-desktop-commander': True,
                'mcp-context7': True,
                'mcp-serena': True,
                'mcp-arxiv-semantic': False,  # 미사용 - 비활성화
                # 'mcp-zotero': False,  # Disabled
            }

    @property
    def ALL_MCP_CONTAINERS(self) -> list:
        """모든 활성화된 MCP 컨테이너 이름 리스트"""
        containers = []
        if self.enabled_servers.get('mcp-sequential', True):
            containers.append(self.SEQUENTIAL_MCP)
        if self.enabled_servers.get('mcp-desktop-commander', True):
            containers.append(self.DESKTOP_MCP)
        if self.enabled_servers.get('mcp-context7', True):
            containers.append(self.CONTEXT7_MCP)
        if self.enabled_servers.get('mcp-serena', True):
            containers.append(self.SERENA_MCP)
        if self.enabled_servers.get('mcp-arxiv-semantic', True):
            containers.append(self.ARXIV_SEMANTIC_MCP)
        # if self.enabled_servers.get('mcp-execution', True):  # Legacy - removed
        #     containers.append(self.EXECUTION_MCP)
        if self.enabled_servers.get('drp-vis', True):
            containers.append(self.DRP_VIS_MCP)
        return containers
    
    def get_container_info(self) -> Dict[str, str]:
        """Get all configured container info"""
        return {
            "user_id": self.user_id,
            "sequential": self.SEQUENTIAL_MCP,
            "desktop": self.DESKTOP_MCP,
            "context7": self.CONTEXT7_MCP,
            "serena": self.SERENA_MCP,
            "arxiv_semantic": self.ARXIV_SEMANTIC_MCP,
            "drp_vis": self.DRP_VIS_MCP,
        }


class GlobalStateManager:
    _global_state: Dict[str, Any] = {}

    @classmethod
    def initialize(cls):
        """Initializes the global state. Call once at application start."""
        if not cls._global_state:
            cls._global_state = {
                "mcp_manager": None, # Add a slot for MCPManager
                # ... other initial state ...
            }
            # Silently initialize
            pass
        else:
            # Already initialized, skip
            pass

    @classmethod
    def update_state(cls, updates: Dict[str, Any], caller_id: str = "UNKNOWN", node_name: str = "UNKNOWN"):
        """Updates the global state with new key-value pairs."""
        for key, value in updates.items():
            cls._global_state[key] = value
        # print(f"[GlobalStateManager] Updated by {caller_id} ({node_name}): {updates.keys()}")

    @classmethod
    def get_state(cls, key: Optional[str] = None) -> Any:
        """Retrieves the entire global state or a specific key."""
        if key:
            return cls._global_state.get(key)
        return cls._global_state

    @classmethod
    def set_mcp_manager(cls, manager_instance):
        """Sets the MCPManager instance in the global state."""
        cls._global_state["mcp_manager"] = manager_instance
        # MCPManager set silently

    @classmethod
    def get_mcp_manager_instance(cls):
        """Retrieves the MCPManager instance from the global state."""
        return cls._global_state.get("mcp_manager")



class DeepResearchEngine:
    """
    open_deep_research의 3-phase 워크플로우 구현.

    Phase 1: Scoping - 연구 질문 생성
    Phase 2: Research - 병렬 쿼리 실행
    Phase 3: Report - 종합 보고서 생성
    """

    async def phase1_scoping(self, model_name: str, llm: Any) -> Dict[str, List[str]]:
        """
        Phase 1: Generate 15 research questions per model (5 arch, 5 perf, 5 improvement).

        Args:
            model_name: Model name (deeptta, deepdr, stagate, deepst, dlm-dti, hyperattentiondti)
            llm: LLM instance

        Returns:
            Research questions dict by category
        """

        # 모델별 특화된 프롬프트 생성
        scoping_prompt = f"""# {model_name.upper()} Deep Research Planning

        Generate 15 specific research questions for {model_name} model analysis:

        ## Architecture Analysis (5 questions)
        - Core structure and design patterns
        - Key components and interactions
        - Data flow and processing methods
        - Technical implementation details
        - Scalability and efficiency aspects

        ## Performance Analysis (5 questions)
        - Computational complexity and efficiency
        - Memory usage and optimization
        - Training speed and convergence
        - Prediction accuracy metrics
        - Bottlenecks and limitations

        ## Improvement Opportunities (5 questions)
        - Current limitations and pain points
        - Integration possibilities with other models
        - Latest techniques applicable
        - Optimization strategies
        - Future enhancement directions

        Return as JSON with three categories: architecture, performance, improvement
        Each category should have exactly 5 specific, research-worthy questions.
        """

        # LLM 호출하여 연구 질문 생성
        response = await llm.ainvoke(scoping_prompt)

        # 기본 구조 (실제로는 LLM 응답을 파싱)
        research_brief = {
            "architecture": [
                f"How does {model_name}'s core architecture process drug-cell line data?",
                f"What are the key components of {model_name} and how do they interact?",
                f"How does {model_name} handle feature extraction and representation?",
                f"What design patterns does {model_name} use for scalability?",
                f"How does {model_name}'s architecture compare to traditional approaches?"
            ],
            "performance": [
                f"What is {model_name}'s computational complexity for training and inference?",
                f"How much memory does {model_name} require for typical datasets?",
                f"What is {model_name}'s average training time and convergence rate?",
                f"What accuracy metrics does {model_name} achieve on benchmark datasets?",
                f"What are the main performance bottlenecks in {model_name}?"
            ],
            "improvement": [
                f"What are {model_name}'s current limitations in drug response prediction?",
                f"How can {model_name} be integrated with other DRP models?",
                f"Which recent ML techniques could enhance {model_name}'s performance?",
                f"What optimization strategies could reduce {model_name}'s resource usage?",
                f"How can {model_name}'s interpretability be improved?"
            ]
        }

        debug_print(f"📋 [Phase 1] Generated 15 research questions for {model_name}")
        return research_brief

    async def phase2_research(self, research_brief: Dict[str, List[str]],
                              model_name: str, mcp_tools: Any) -> List[Dict[str, Any]]:
        """
        Phase 2: 15개 쿼리를 병렬로 연구 실행.

        Args:
            research_brief: 연구 질문 딕셔너리
            model_name: 모델 이름
            mcp_tools: MCP 도구 인스턴스

        Returns:
            연구 결과 리스트
        """

        # 모든 쿼리를 플랫 리스트로 변환
        all_queries = []
        for category, questions in research_brief.items():
            for question in questions:
                all_queries.append({
                    "category": category,
                    "question": question,
                    "model": model_name
                })

        # 병렬 태스크 생성
        debug_print(f"🔬 [Phase 2] Starting parallel research for {len(all_queries)} queries")

        research_tasks = []
        for query_info in all_queries:
            task = asyncio.create_task(
                self._research_single_query(query_info, mcp_tools)
            )
            research_tasks.append(task)

        # 모든 연구 동시 실행
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*research_tasks, return_exceptions=True)
        elapsed = asyncio.get_event_loop().time() - start_time

        debug_print(f"⚡ [Phase 2] Completed {len(results)} queries in {elapsed:.2f}s (parallel)")

        # 예외 처리
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                debug_print(f"❌ Query failed: {r}")
            else:
                valid_results.append(r)

        return valid_results

    async def _research_single_query(self, query_info: Dict, mcp_tools: Any) -> Dict[str, Any]:
        """
        개별 쿼리에 대한 심층 연구 수행.

        Args:
            query_info: 쿼리 정보 (category, question, model)
            mcp_tools: MCP 도구

        Returns:
            연구 결과
        """

        question = query_info["question"]
        model = query_info["model"]
        category = query_info["category"]

        try:
            # Serena MCP로 코드 분석 (실제 구현시 활성화)
            # code_analysis = await mcp_tools.call(
            #     "mcp__serena__search_for_pattern",
            #     {"pattern": model, "directory": f"/workspace/experiments/{model}_agent/"}
            # )

            # Sequential thinking으로 심층 분석 (실제 구현시 활성화)
            # deep_thought = await mcp_tools.call(
            #     "mcp__sequential-thinking__sequentialthinking",
            #     {"thought": question, "nextThoughtNeeded": True}
            # )

            # 임시 결과 (실제로는 MCP 도구 사용)
            result = {
                "query": question,
                "category": category,
                "model": model,
                "findings": f"Research findings for: {question}",
                "code_evidence": f"Code analysis for {model}",
                "deep_insights": f"Deep analysis insights",
                "confidence": 0.85
            }

            return result

        except Exception as e:
            debug_print(f"❌ Error researching query: {e}")
            return {
                "query": question,
                "category": category,
                "model": model,
                "error": str(e)
            }

    async def phase3_report(self, research_results: List[Dict],
                            model_name: str, llm: Any) -> Dict[str, Any]:
        """
        Phase 3: 연구 결과를 종합하여 최종 보고서 생성.

        Args:
            research_results: Phase 2의 연구 결과들
            model_name: 모델 이름
            llm: LLM 인스턴스

        Returns:
            종합 보고서
        """

        # 카테고리별로 결과 정리
        categorized_results = {
            "architecture": [],
            "performance": [],
            "improvement": []
        }

        for result in research_results:
            if "error" not in result:
                category = result.get("category", "unknown")
                if category in categorized_results:
                    categorized_results[category].append(result)

        # 종합 보고서 생성 프롬프트
        synthesis_prompt = f"""# {model_name.upper()} Deep Research Synthesis

        Based on {len(research_results)} research queries, create a comprehensive report:

        ## Architecture Insights ({len(categorized_results['architecture'])} findings)
        Synthesize the architectural findings

        ## Performance Analysis ({len(categorized_results['performance'])} findings)
        Summarize performance characteristics

        ## Improvement Opportunities ({len(categorized_results['improvement'])} findings)
        Identify key improvement areas

        ## Key Recommendations
        Provide actionable recommendations based on all findings
        """

        # 최종 보고서 구조
        final_report = {
            "model": model_name,
            "total_queries": len(research_results),
            "successful_queries": len([r for r in research_results if "error" not in r]),
            "architecture_insights": categorized_results["architecture"],
            "performance_analysis": categorized_results["performance"],
            "improvement_opportunities": categorized_results["improvement"],
            "synthesis": f"Comprehensive synthesis for {model_name}",
            "key_findings": [
                f"{model_name} shows strong performance in specific areas",
                f"Integration opportunities identified with other models",
                f"Several optimization strategies recommended"
            ],
            "recommendations": [
                "Consider implementing suggested optimizations",
                "Explore integration with complementary models",
                "Focus on identified bottlenecks for improvement"
            ]
        }

        debug_print(f"📊 [Phase 3] Generated comprehensive report for {model_name}")
        return final_report


# ============= File Reading Tools =============

@tool
def read_multiple_files(
    file_paths: list[str],
    start_lines: list[int] | None = None,
    end_lines: list[int] | None = None
) -> str:
    """Read one or more source files with optional line range control.

    This tool efficiently reads multiple files in a single operation, with optional
    line range specification for each file. Perfect for Step 1 context gathering.

    Args:
        file_paths: List of file paths relative to /workspace/ (or absolute paths)
                   Can be a single file: ["file.py"] or multiple: ["file1.py", "file2.py"]
        start_lines: Optional list of 0-based start line indices for each file.
                    If None, reads from beginning. Must match length of file_paths if provided.
        end_lines: Optional list of 0-based end line indices (inclusive) for each file.
                  If None, reads to end. Must match length of file_paths if provided.

    Returns:
        Concatenated content of all files with clear separators between them.
        Each file section starts with a banner showing the file path and line range.

    Examples:
        Read entire files:
        >>> read_multiple_files([
        ...     "experiments/deeptta/README.md",
        ...     "experiments/deeptta/main.py"
        ... ])

        Read single file with line range (lines 10-50):
        >>> read_multiple_files(
        ...     file_paths=["experiments/deeptta/model.py"],
        ...     start_lines=[10],
        ...     end_lines=[50]
        ... )

        Read multiple files with different line ranges:
        >>> read_multiple_files(
        ...     file_paths=["file1.py", "file2.py"],
        ...     start_lines=[0, 100],
        ...     end_lines=[50, 200]
        ... )

    Note:
        - Uses UTF-8 encoding for all files
        - Line indices are 0-based (first line is line 0)
        - end_line is inclusive (will include that line)
        - Handles missing files gracefully with error messages
        - If start_lines/end_lines are None, reads entire files
        - Paths are relative to current working directory or absolute
    """
    result = []

    # Validate line range parameters
    if start_lines is not None and len(start_lines) != len(file_paths):
        return f"ERROR: start_lines length ({len(start_lines)}) must match file_paths length ({len(file_paths)})"
    if end_lines is not None and len(end_lines) != len(file_paths):
        return f"ERROR: end_lines length ({len(end_lines)}) must match file_paths length ({len(file_paths)})"

    for idx, f in enumerate(file_paths):
        # Use path as-is (relative to cwd or absolute)
        full_path = f

        # Get line range for this file
        start = start_lines[idx] if start_lines is not None else None
        end = end_lines[idx] if end_lines is not None else None

        # Build file separator with line range info
        if start is not None or end is not None:
            range_info = f" [lines {start if start is not None else 0}:{end if end is not None else 'EOF'}]"
        else:
            range_info = ""
        result.append(f"\n{'='*80}\nFILE: {f}{range_info}\n{'='*80}")

        try:
            with open(full_path, 'r', encoding='utf-8') as fp:
                lines = fp.readlines()

            # Apply line range
            if start is not None or end is not None:
                start_idx = start if start is not None else 0
                end_idx = (end + 1) if end is not None else len(lines)
                lines = lines[start_idx:end_idx]

            content = ''.join(lines)
            result.append(content)

        except FileNotFoundError:
            result.append(f"ERROR: File not found at {full_path}")
        except PermissionError:
            result.append(f"ERROR: Permission denied reading {full_path}")
        except Exception as e:
            result.append(f"ERROR: {type(e).__name__}: {e}")

    return '\n'.join(result)
