import operator
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


# =============================================================================
# CUSTOM REDUCERS FOR MEMORY OPTIMIZATION
# =============================================================================

def add_messages_with_limit(
    left: List[BaseMessage],
    right: List[BaseMessage]
) -> List[BaseMessage]:
    """Custom reducer: Limit messages in STATE (LangGraph official pattern).

    Keeps the system prompt plus the most recent messages without breaking
    AI/tool call ordering.

    args:
        left: Existing messages
        right: New messages to add

    Returns:
        Combined message list with size limit (max 10) and aggressive pruning
    """
    combined = list(add_messages(left, right))
    MAX_MESSAGES = 20  # Increased from 10 to 20 to prevent context loss

    # aggressive pruning of old tool outputs within the kept messages
    # We truncate large tool outputs in messages that are not the very last one
    for i in range(len(combined) - 1):
        msg = combined[i]
        if isinstance(msg, ToolMessage) and len(msg.content) > 3000:
            # Keep first 1500 chars for context (prevents losing critical evidence)
            msg.content = msg.content[:1500] + "... [TRUNCATED TO SAVE TOKENS]"

    if len(combined) <= MAX_MESSAGES:
        return combined

    system_message = None
    remaining_messages = combined

    if combined and isinstance(combined[0], SystemMessage):
        system_message = combined[0]
        remaining_messages = combined[1:]

    allowed = MAX_MESSAGES - (1 if system_message else 0)

    # Drop oldest entries while keeping AI/tool call groups intact
    while len(remaining_messages) > allowed:
        oldest = remaining_messages.pop(0)

        if isinstance(oldest, AIMessage) and getattr(oldest, "tool_calls", None):
            # Remove tool responses paired with this AI message
            while remaining_messages and isinstance(remaining_messages[0], ToolMessage):
                remaining_messages.pop(0)

    if system_message:
        return [system_message] + remaining_messages
    return remaining_messages


def add_logs_with_limit(
    left: List[str],
    right: List[str]
) -> List[str]:
    """Custom reducer that limits processing logs to prevent unbounded growth.

    Keeps only the most recent 100 log entries for debugging.
    This is a 2-parameter reducer compatible with LangGraph requirements.

    Args:
        left: Existing logs
        right: New logs to add

    Returns:
        Combined log list with size limit (max 100 entries)
    """
    MAX_LOGS = 100
    combined = (left or []) + (right or [])

    # If over limit, keep only recent N
    if len(combined) > MAX_LOGS:
        return combined[-MAX_LOGS:]
    return combined


def add_debate_history_with_deduplication(
    left: List[Dict],
    right: List[Dict]
) -> List[Dict]:
    """Custom reducer: Add debate history entries with automatic deduplication.

    Prevents duplicate entries during rapid node executions by using content_path
    as a unique identifier. This solves the issue where save_debate_history_entry()
    checks stale state before LangGraph applies state updates.

    Args:
        left: Existing debate history entries
        right: New debate history entries to add

    Returns:
        Combined history list with duplicates removed based on content_path
    """
    combined = list(left or [])

    # Build set of existing content_paths for O(1) lookup
    existing_paths = {
        entry.get("content_path")
        for entry in combined
        if entry.get("content_path")
    }

    # Add only new entries (not already in history)
    for entry in (right or []):
        content_path = entry.get("content_path")

        if content_path and content_path not in existing_paths:
            # New entry with unique content_path
            combined.append(entry)
            existing_paths.add(content_path)
        elif not content_path:
            # Fallback: no content_path, check content + speaker combination
            content = entry.get("content", "")
            speaker = entry.get("speaker", "")
            turn = entry.get("turn", 0)

            # Check if identical entry exists (same speaker, content, and turn)
            is_duplicate = any(
                e.get("speaker") == speaker
                and e.get("content") == content
                and e.get("turn") == turn
                for e in combined
            )

            if not is_duplicate:
                combined.append(entry)

    return combined


class MARBLEState(TypedDict):
    """LangGraph StateGraph compatible state schema for MARBLE.
    
    This state schema uses LangGraph reducers for automatic state updates:
    - List fields with operator.add: append new items to existing lists
    - Dict fields with operator.or_: merge new key-value pairs into existing dicts
    - Regular fields: overwrite with new values
    
    Nodes should return dictionaries with the fields they want to update,
    and LangGraph will automatically apply the appropriate reducers.
    
    State Design: Simplified structure with 1-2 representative fields per node.
    Detailed information is stored in sub-JSON structures within each field,
    enabling complete workflow understanding while maintaining simplicity.
    """
    
    # Core conversation (limited to 30 messages to prevent debate context overflow)
    messages: Annotated[List[BaseMessage], add_messages_with_limit]

    # Required by create_react_agent for step tracking
    remaining_steps: Optional[int] = None

    # Routing information
    current_node: Optional[str] = None
    next_node: Optional[str] = None
    last_speaker: Optional[str] = None

    # Entry Router Agent results
    router_decision: Optional[str] = None
    router_reasoning: Optional[str] = None
    
    # ============= WORKFLOW TRACKING =============

    # Processing logs (limited to 100 entries to prevent unbounded growth)
    processing_logs: Annotated[List[str], add_logs_with_limit]
    
    # ============= UNIFIED MODEL STATES (LangGraph Best Practice) =============
    
    # Single field managing all model states with nested structure
    model_states: Annotated[Dict[str, Any], operator.or_]
    """
    Unified model state structure following LangGraph patterns:
    {
        "deeptta": {
            "training": {"log_path": str, "status": str, "config": dict},
            "analysis": {"results": dict, "report_path": str, "summary": dict},
            "research": {"insights": dict, "report_path": str}
        },
        "deepdr": {...same structure...},
        "stagate": {...same structure...}
    }
    """
    

    # ============= DEEP RESEARCH INTEGRATION =============

    # Research consensus from deep research workflow
    """
    Deep research consensus structure:
    {
        "deeptta": {deep_research_report or regular_insights},
        "deepdr": {deep_research_report or regular_insights},
        "stagate": {deep_research_report or regular_insights},
        "unified_recommendations": [...],
        "integration_plan": {...}
    }
    """

    # ============= TASK ROUTING =============

    # Task mode routing
    task_mode: Optional[str] = None
    """Task mode: 'train' | 'develop' - determines workflow routing"""

    # Component specification
    component: Optional[str] = None
    """Component to modify: 'decoder' | 'cell_encoder' | 'drug_encoder' | None"""

    # Config path
    generated_config_path: Optional[str] = None
    """Path to generated baseline config YAML"""

    # ============= UNIFIED EXECUTION (통합 실행: Baseline + Refined + 4 Models) =============

    # Execution mode selection
    execution_mode: Optional[str] = None
    """Execution mode: 'baseline' or 'refined' (optional, for compatibility)"""

    # Container and environment
    # Note: model_name is stored in current_model field (line 276)
    container_id: Optional[str] = None
    """Docker container ID created by env_builder"""

    model_source_path: Optional[str] = None
    """Path to model source code (e.g., /workspace/experiments/deeptta_agent)"""

    execution_command: Optional[str] = None
    """Command to execute in container (e.g., 'python main.py --mode train')"""

    # Execution results
    log_file_path: Optional[str] = None
    """Path to execution log file"""

    execution_report: Annotated[Dict[str, Any], operator.or_]
    """
    Execution report from Reporter (Pydantic validated):
    {
        "execution_success": bool,
        "metrics": {"accuracy": 0.85, "mse": 0.12, "r2": 0.78},
        "summary": "Model training completed successfully...",
        "error_message": str | null,
        "log_file_path": str
    }
    """

    # Research phase
    research_config: Annotated[Dict[str, Any], operator.or_]
    models_to_execute: Annotated[List[str], operator.add]  # Which models to research
    current_model: Optional[str] = None  # Current model being processed
    models_executed: Annotated[List[str], operator.add]  # List of executed models

    # Debate phase - Core fields only
    debate_session_id: Optional[str] = None
    """Unique identifier for the active debate session (prevents overlap across runs)"""

    agenda_report_path: Optional[str] = None
    """
    Path to agenda report file generated by agenda_generator.
    Contains debate topics, target model analysis, and session planning.

    Example: experiments/debate_sessions/debate_20250114_153045/agenda_report.md

    Used by:
    - debate_moderator: Reference debate topics and priorities
    - debate_reporter: Include agenda context in final report
    """

    # Debate: Target model selection
    target_model: Optional[str] = None
    """
    Single target model selected by agenda_generator for improvement analysis.
    Possible values: 'deeptta', 'deepdr', 'stagate', 'deepst', 'dlm-dti', 'hyperattentiondti'

    Routing logic:
    - Each model researcher in ACTIVE_RESEARCHER_MODELS generates proposals
    - model_researcher processes all models in single execution
    """

    # Debate: Current topic for focused discussion
    current_topic: Optional[Dict[str, str]] = None
    """
    Single topic selected by agenda_generator for focused debate discussion.
    Structure: {
        "component": str,  # HUMAN-READABLE: "Cell Encoder Architecture", "Drug Encoder", etc.
        "limitation": str,  # e.g., "Limited feature representation capacity"
        "question": str    # e.g., "How can we enhance encoder expressiveness?"
    }

    NOTE: current_topic["component"] is HUMAN-READABLE for display/debate purposes.
    For machine-readable component ID, use target_component field instead.

    Used by:
    - model_researcher: Generate proposals specifically addressing this topic
    - debate participants: Focus discussion on single architectural improvement
    """

    # Debate: Target component specification
    target_component: Optional[str] = None
    """
    Machine-readable component identifier for implementation.
    MUST be one of: 'drug_encoder', 'cell_encoder', or 'decoder'

    NOTE: This is MACHINE-READABLE for config/code generation.
    For human-readable display name, use current_topic["component"] instead.

    Relationship with current_topic:
    - target_component: "cell_encoder" (exact string for code)
    - current_topic["component"]: "Cell Encoder Architecture" (descriptive name for debate)

    Set by: agenda_generator (Round 1)
    Used by: model_researcher, debate_moderator, representative_researcher, ml_planner

    Purpose: Ensures component clarity throughout debate → implementation transition.
    Prevents ambiguity like "the encoder" vs "drug_encoder" vs "cell_encoder".
    """

    # Debate: File-based transcript management
    debate_history: Annotated[List[Dict], add_debate_history_with_deduplication]
    """
    Full debate transcript with automatic deduplication. Each turn: {
        "turn": 1,
        "speaker": "representative_researcher",
        "content": "...",
        "content_path": "experiments/.../turn_01_speaker.md",
        "timestamp": 0
    }

    Note: Uses custom reducer to prevent duplicates during rapid node executions.
    Deduplication based on content_path (unique file path per turn).
    """

    # Debate: Turn management
    turn_count: int = 0
    """Total number of debate turns"""

    max_turns: int = 50
    """Overall safety limit for entire debate (unified from max_total_turns)"""

    # Debate round counters
    debate_round: int = 0
    """Current debate round (0-3)"""

    code_fix_round: int = 0
    """Current code fix iteration"""

    # ============= PARALLEL PAPER SEARCH (PMC + OpenReview) =============

    pmc_results_path: Optional[str] = None
    """Path to PMC researcher results JSON (from parallel search)"""

    openreview_results_path: Optional[str] = None
    """Path to OpenReview researcher results JSON (from parallel search)"""

    aggregated_papers_path: Optional[str] = None
    """Path to aggregated and scored papers JSON (from paper_aggregator)"""

    scoring_weights: Annotated[Dict[str, float], operator.or_]
    """
    Adaptive scoring weights for paper selection: {
        "domain": 0.4,       # Domain similarity (DRP relevance)
        "architecture": 0.5, # Architecture similarity (to target model)
        "novelty": 0.1       # Novelty (1 - architecture similarity)
    }
    Updated based on iteration feedback (-1/0/+1 per metric).
    """

    # Docker test
    docker_test_round: int = 0
    """Current Docker test round (0-5)"""

    docker_test_success: bool = False
    """Flag indicating Docker test passed"""

    docker_test_error: Optional[str] = None
    """Error message from Docker test if failed"""

    docker_test_output: Optional[str] = None
    """Output from successful Docker test"""

    # Config validation (Phase 1)
    config_validation_passed: bool = False
    """Flag indicating config validation passed in Phase 1"""

    config_fix_attempt: int = 0
    """Current config fix attempt counter (max 5)"""

    config_retry_count: int = 0
    """Number of times config was reset after max attempts"""

    # Type validation (Phase 2)
    type_validation_passed: bool = False
    """Flag indicating type validation passed in Phase 2"""

    type_fix_attempt: int = 0
    """Current type fix attempt counter (max 5)"""

    # Phase tracking
    phase: Optional[str] = None
    """Current phase: 'config_debate' | 'type_debate' | 'docker_test'"""

    phase1_success: bool = False
    """Flag indicating Phase 1 (config debate) completed successfully"""

    phase2_success: bool = False
    """Flag indicating Phase 2 (type debate) completed successfully"""

    # Original config path for reset functionality
    original_config_path: Optional[str] = None
    """Original config file path for resetting after failed attempts"""

    # Debate result file paths
    debate_config_result_path: Optional[str] = None
    """Path to debate_config_result.md (Phase 1 output)"""

    debate_another_result_path: Optional[str] = None
    """Path to debate_another_result.md (Phase 2 output)"""

    next_speaker: str = ""
    """Moderator's routing decision: who should speak next"""

    last_speaker: str = ""
    """Who spoke in previous turn (for context)"""

    # Debate phase tracking - Participant conclusion states
    current_phase: Optional[str] = None
    """Current debate phase: exploration | discussion | evaluation | consensus"""

    rep_conclusion: str = ""
    """Representative's latest conclusion: PRESENTED | EVALUATED | PRIORITIZED"""

    ml_conclusion: str = ""
    """ML Specialist's latest conclusion: REVIEWED | FEASIBLE | NEEDS_REVISION | AGREED"""

    critic_conclusion: str = ""
    """Critic's latest conclusion: REVIEWED | SUPPORTED | NEEDS_REVISION | AGREED"""

    # Debate: Completion tracking
    debate_complete: bool = False
    """Flag indicating debate has concluded"""

    forced_consensus_active: bool = False
    """Flag indicating forced consensus mode is active (turn >= 30)"""

    final_report_path: Optional[str] = None
    """Path to final debate decision report (MD file)"""

    final_report_generated: bool = False
    """Flag indicating final report has been generated (used for routing)"""

    # Debate: Session-specific logging
    debate_session_logs: Annotated[Dict[str, List[str]], operator.or_]
    """Per-session log messages keyed by debate_session_id"""

    # Debate artifacts and workspaces
    debate_workspace: Optional[str] = None
    """Workspace path for debate outputs"""

    problem_md_path: Optional[str] = None
    """Path to problem.md produced by problem-raiser agent"""

    debate_result_md_path: Optional[str] = None
    """Path to debate_result.md produced after proposals are validated"""

    target_config_path: Optional[str] = None
    """Full path to the target config YAML"""

    encoder_decoder_inventory: Annotated[Dict[str, Any], operator.or_]
    """Scanned enc/dec files across docker_images/* for grounded proposals"""

    proposed_changes: Annotated[Dict[str, Any], operator.or_]
    """Structured proposal summary used by the code expert"""

    code_copy_path: Optional[str] = None
    """Path where the target model code was copied for debate-driven edits"""

    # Model profiles for debate-informed decisions
    model_profiles: Annotated[Dict[str, Any], operator.or_]
    """
    저장된 모델 프로필 (DeepDR, DRPreter, DeepTTA)
    Structure:
    {
        "deepdr": {
            "model_name": "DeepDR",
            "paper_key_findings": {...},
            "architecture_design": {...},
            "component_constraints": {...},
            "strengths_vs_other_models": {...},
            "weaknesses_known": [...]
        },
        "deeptta": {...},
        "deepdr": {...}
    }

    Generated by model_profile_generator_agent at debate workflow start.
    Saved as JSON files in experiments/debate_files/model_profiles/.
    """

    profile_dir: Optional[str] = None
    # 모델 프로필 디렉토리 경로 (예: experiments/debate_files/model_profiles)

    regenerate_profiles: bool = False
    # 프로필 강제 재생성 플래그 (--regenerate-profiles CLI 플래그)

    # Structured output support for agents using response_format
    structured_response: Optional[BaseModel] = None
    """
    Structured response from agents configured with response_format parameter.
    Currently used by:
    - debate_moderator: RouterDecision Pydantic model for routing decisions
    """

    # Note: Expert assessments and synthesis removed from state
    # All debate content available in conversation history via checkpointer
    # Reporter reads saved files for final report generation

    # ============= DEVELOPMENT SUBGRAPH (Simplified) =============
    # Essential metadata for workflow routing and decisions

    development_iteration_count: int = 0
    """Current iteration (0=creation, 1+=fixing)"""

    drp_framework_workspace: Optional[str] = None
    """Isolated framework path: experiments/drp_framework_refined_{model}/"""

    component_registry_analysis: Annotated[Dict[str, str], operator.or_]
    """
    Component type mapping from ml_planner (ml_engineer can update).
    Converted from list[ComponentRegistryEntry] to dict in graph.py.
    Example:
    {
      "drug_encoder": "gin",
      "cell_encoder": "mlp",
      "decoder": "concat_mlp"
    }
    """

    component_actions: Annotated[Dict[str, str], operator.or_]
    """
    Component action decisions from ml_planner.
    Converted from list[ComponentActionEntry] to dict in graph.py.
    Example:
    {
      "drug_encoder": "CONFIG_ONLY",
      "cell_encoder": "IMPLEMENT_NEW",
      "decoder": "CONFIG_ONLY"
    }
    """

    validation_status: Annotated[Dict[str, Any], operator.or_]
    """
    Current validation state (summary only):
    {
      "passed": False,
      "error_count": 3,
      "iteration": 1
    }
    """

    validation_summary: Optional[str] = None
    """
    Brief validation summary showing error counts per stage.
    Format: "Stage 1: X errors | Stage 2: Y errors | Stage 3: Z errors"

    Full validation logs are saved to MD file (see validation_errors_path).
    This reduces state memory usage while maintaining access to detailed errors.
    """

    validation_errors_path: Optional[str] = None
    """
    Path to structured validation errors MD file (e.g., experiments/drp_framework_refined_deeptta/validation_reports/VALIDATION_ERRORS_iter1.md).

    ml_engineer MUST read this file in FIXING mode (iteration > 0) to understand:
    - Structured error details grouped by stage (1: Syntax, 2: Import, 3: Forward Pass)
    - File paths and line numbers for each error
    - Suggested fixes for each error type
    - Config files causing issues

    Format: VALIDATION_ERRORS_iter{N}.md (one file per iteration)
    """

    implementation_plan_path: Optional[str] = None
    """
    Path to implementation plan MD file generated by ml_planner.
    Format: experiments/reports/{target_model}/proposals/implementation_plan_detailed.md

    Contains:
    - Component registry analysis (CONFIG_ONLY vs IMPLEMENT_NEW)
    - Detailed implementation instructions
    - Config YAML templates
    - Execution summary
    """

    # Optional: Audit trail (file paths and change records)
    modified_files: Annotated[List[Dict[str, Any]], operator.add]
    """
    Change history with iteration tracking:
    [
      {
        "iteration": 1,
        "timestamp": "2025-10-30 01:14:52",
        "file_path": "...",
        "added_imports": [...],
        "created_components": [...],
        "modified_components": [...],
        "deleted_components": [...]
      }
    ]
    """

    # Refined execution results
    refined_config: Annotated[Dict[str, Any], operator.or_]
    refined_env_setup: Annotated[Dict[str, Any], operator.or_]
    refined_execution_results: Annotated[Dict[str, Any], operator.or_]
    refined_analysis_results: Annotated[Dict[str, Any], operator.or_]
    refinements_applied: Annotated[List[str], operator.add]
    refined_model_outputs: Annotated[Dict[str, Any], operator.or_]
    improvements_applied: Annotated[Dict[str, Any], operator.or_]
    performance_comparison: Annotated[Dict[str, Any], operator.or_]
    overall_improvement: Optional[float] = None
    best_refined_model: Optional[str] = None
    performance_improvement: Optional[float] = None

    # Assessment phase
    improvement_assessment: Annotated[Dict[str, Any], operator.or_]
    metrics_comparison: Annotated[Dict[str, Any], operator.or_]
    iteration_recommended: Optional[bool] = None

    # Overall workflow status
    performance: Annotated[Dict[str, Any], operator.or_]  # Current performance metrics

    # ============= BUILD DEBATE WORKFLOW (Dynamic Rounds) =============

    build_debate_round: int = 0
    """Current debate round number (1, 2, 3, ...)"""

    build_debate_max_rounds: int = 5
    """Maximum number of debate rounds before forced conclusion"""

    build_debate_consensus_reached: bool = False
    """Flag indicating consensus has been reached"""

    build_debate_consensus_reason: Optional[str] = None
    """Reason for consensus (e.g., 'FULL_ADOPTION', 'STRONG_CONSENSUS', 'MAX_ROUNDS')"""

    article1_proposal_status: str = "active"
    """Article 1 proposal status: 'active' | 'withdrawn' | 'adopted'"""

    article2_proposal_status: str = "active"
    """Article 2 proposal status: 'active' | 'withdrawn' | 'adopted'"""

    article3_proposal_status: str = "active"
    """Article 3 proposal status: 'active' | 'withdrawn' | 'adopted'"""

    article4_proposal_status: str = "active"
    """Article 4 proposal status: 'active' | 'withdrawn' | 'adopted'"""

    article5_proposal_status: str = "active"
    """Article 5 proposal status: 'active' | 'withdrawn' | 'adopted'"""

    build_debate_transcript_path: Optional[str] = None
    """Path to debate_transcript.md file containing all rounds"""

    build_debate_step: int = 0
    """Current step within a round (1-8)"""

    # ============= ITERATION WORKFLOW =============

    current_iteration: int = 0
    """현재 iteration 번호 (1부터 시작, 0은 시작 전)"""

    total_iterations: int = 1
    """총 iteration 횟수 (--iter 플래그 값, 기본값 1)"""

    iteration_context: Optional[str] = None
    """프롬프트 주입용 이전 iteration 컨텍스트 (EvolvingMemory에서 생성)"""

    iteration_metrics: Annotated[Dict[str, Any], operator.or_]
    """
    현재 iteration의 성능 메트릭:
    {
        "rmse": 0.85,
        "pearson": 0.73,
        "mse": 0.72,
        ...
    }
    Docker 실행 후 추출되어 저장됨.
    """

    # ============= WEIGHT ADJUSTMENT (Iteration 2+) =============

    skip_paper_search: bool = False
    """
    Flag to skip paper search and use next papers from aggregated_results.json.
    Set by iteration_critic when:
    - Previous iteration failed (performance dropped)
    - consecutive_failures < 3

    When True, paper_selector node picks next unused papers from last_paper_search_iteration's
    aggregated_results.json instead of doing new paper search.
    If aggregated_results.json not found, fallback_to_paper_search is triggered.
    """

    should_terminate: bool = False
    """
    Flag to terminate the iteration loop.
    Set when novelty weight >= 0.8 (no more relevant papers to explore).
    """

    fallback_to_paper_search: bool = False
    """
    Flag to trigger new paper search as fallback.
    Set by paper_selector when aggregated_results.json doesn't exist.
    Used by routing logic to redirect to paper_search_start instead of other_paper_reader.
    """

    is_continue_mode: bool = False
    """
    Flag for continue mode (--task continue).
    When True:
    - EvolvingMemory is NOT cleared (preserves previous iterations)
    - Existing build_N folders are NOT deleted
    - current_iteration starts from user-specified --iter value
    """

    continue_stage: Optional[str] = None
    """
    Stage to resume from in continue mode (--stage flag).
    Valid values: "debate", "development", "docker"
    - debate: Start from build_debate_subgraph (default)
    - development: Skip debate, start from build_development_subgraph
    - docker: Skip debate and development, start from docker_execution_subgraph
    """

    # ============= REWARD SETTINGS =============

    reward_patience: int = 10
    """
    Reward 적용 주기 (--patience 플래그 값, 기본값 10).
    iter 1~patience: reward 미적용
    iter patience+1 ~ 2*patience: iter patience 스냅샷 적용
    예: patience=5 → iter 6-10에서 iter 5 스냅샷 적용
    """

    reward_weight: float = 0.1
    """
    Reward 가중치 (--weight 플래그 값, 기본값 0.1).
    Formula: V_i = Sim_i + (w × (N_success - N_failure) / (N_total + 1))
    """


class StateManager:
    """State management utilities for MARBLE.
    
    Note: Most state updates now use LangGraph reducers directly.
    This class maintains essential utility methods for backward compatibility.
    
    Preferred approach: Nodes return reducer-compatible dictionaries.
    """
    
    # === Utility Methods ===
    @staticmethod
    def get_current_timestamp() -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    # Utility methods maintained for backward compatibility


__all__ = [
    "MARBLEState", 
    "StateManager"
]
