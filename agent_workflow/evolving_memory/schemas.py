"""Pydantic schemas for EvolvingMemory iteration data."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class IterationPerformance(BaseModel):
    """Performance metrics for a single iteration."""
    # DRP metrics
    rmse: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    pcc: Optional[float] = None
    scc: Optional[float] = None
    pearson: Optional[float] = None
    spearman: Optional[float] = None
    # Spatial metrics
    ari: Optional[float] = None
    nmi: Optional[float] = None
    silhouette: Optional[float] = None
    # DTI/Drug Repurposing metrics
    accuracy: Optional[float] = None
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    f1: Optional[float] = None
    loss: Optional[float] = None


class IterationChanges(BaseModel):
    """Changes made in an iteration."""
    component: str  # e.g., "drug_encoder", "cell_encoder", "decoder"
    description: str  # e.g., "Changed GCN to GAT"
    files_modified: List[str] = Field(default_factory=list)

    # Detailed content extracted from md files
    implementation: Optional[str] = None  # Decision Summary + Architecture Overview
    weakness: Optional[str] = None  # Key content from weakness_of_target_model.md


class IterationAnalysis(BaseModel):
    """Iteration result analysis and lessons."""
    improved: Optional[bool] = None  # None for first iteration
    delta: Optional[Dict[str, float]] = None  # e.g., {"rmse": -0.13, "pearson": +0.07}
    reason: str = ""  # Reason for this result
    lessons: List[str] = Field(default_factory=list)  # Key lessons


class IterationWeights(BaseModel):
    """Iteration weights for EmbeddingScorer formula.

    Formula: S_total = w_d × S_domain + w_a × [β × S_arch + (1-β) × novelty]

    - w_d: Domain weight (0.9 → 0.1 over iterations, auto-calculated)
    - w_a: Architecture weight (0.1 → 0.9 over iterations, auto-calculated)
    - beta: Novelty coefficient (1.0=similarity focus, 0.0=novelty focus, LLM-adjustable)
    """
    w_d: Optional[float] = None
    w_a: Optional[float] = None
    beta: Optional[float] = None


class BaselineRecord(BaseModel):
    """iter0: Original model baseline."""
    model_name: str
    description: str
    domain: str = ""
    performance: Dict[str, float] = Field(default_factory=dict)


class PaperRewardRecord(BaseModel):
    """Per-paper reward tracking record.

    Formula: V_i = Sim_i + (w × (N_success - N_failure) / (N_total + 1))
    """
    paper_id: str
    title: str
    n_success: int = 0
    n_failure: int = 0
    n_total: int = 0
    last_used_iteration: Optional[int] = None


class IterationRecord(BaseModel):
    """Complete record for a single iteration."""
    iteration: int
    timestamp: datetime = Field(default_factory=datetime.now)
    performance: IterationPerformance = Field(default_factory=IterationPerformance)
    changes: IterationChanges = Field(default_factory=lambda: IterationChanges(component="unknown", description=""))
    analysis: IterationAnalysis = Field(default_factory=IterationAnalysis)
    weights: IterationWeights = Field(default_factory=IterationWeights)

    # Papers used in this iteration
    papers_used: List[str] = Field(default_factory=list)

    # Artifact paths
    debate_outputs_path: Optional[str] = None
    src_path: Optional[str] = None


class EvolvingMemoryData(BaseModel):
    """EvolvingMemory storage - iterations are accumulated."""
    total_iterations: int = 0
    planned_iterations: Optional[int] = None
    target_model: Optional[str] = None
    best_iteration: Optional[int] = None
    best_performance: Optional[IterationPerformance] = None

    # iter0: Original model baseline
    baseline: Optional[BaselineRecord] = None
    iterations: List[IterationRecord] = Field(default_factory=list)

    # Accumulated learning for prompt injection
    key_lessons: List[str] = Field(default_factory=list)
    failed_approaches: List[str] = Field(default_factory=list)

    # Consecutive failure tracking (increments on degradation, resets on improvement)
    consecutive_failures: int = 0

    # Global paper list (accumulated from all iterations)
    used_papers: List[str] = Field(default_factory=list)
    paper_rewards: Dict[str, PaperRewardRecord] = Field(default_factory=dict)
    paper_reward_snapshots: Dict[int, Dict[str, float]] = Field(default_factory=dict)
    reward_weight: float = 0.1
    reward_patience: int = 10
    last_paper_search_iteration: Optional[int] = None
    current_weights: IterationWeights = Field(default_factory=IterationWeights)
    current_beta: float = 1.0
    skip_paper_search: bool = True
