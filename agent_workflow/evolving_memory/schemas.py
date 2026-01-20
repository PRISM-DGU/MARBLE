"""Pydantic schemas for EvolvingMemory iteration data."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class IterationPerformance(BaseModel):
    """단일 iteration의 성능 메트릭."""
    # DRP metrics
    rmse: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    pcc: Optional[float] = None  # Pearson Correlation Coefficient
    scc: Optional[float] = None  # Spearman Correlation Coefficient
    pearson: Optional[float] = None  # alias for pcc
    spearman: Optional[float] = None  # alias for scc
    # Spatial metrics
    ari: Optional[float] = None  # Adjusted Rand Index
    nmi: Optional[float] = None  # Normalized Mutual Information
    silhouette: Optional[float] = None
    # DTI/Drug Repurposing metrics
    accuracy: Optional[float] = None
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    f1: Optional[float] = None
    loss: Optional[float] = None
    # Other metrics
    custom_metrics: Dict[str, float] = Field(default_factory=dict)


class IterationChanges(BaseModel):
    """해당 iteration에서 수행한 변경 사항."""
    component: str  # 예: "drug_encoder", "cell_encoder", "decoder"
    description: str  # 예: "GCN을 GAT로 변경"
    files_modified: List[str] = Field(default_factory=list)

    # md 파일에서 추출한 상세 내용
    implementation: Optional[str] = None  # Decision Summary + Architecture Overview
    weakness: Optional[str] = None  # weakness_of_target_model.md 핵심 내용


class IterationAnalysis(BaseModel):
    """iteration 결과 분석 및 교훈."""
    improved: Optional[bool] = None  # 첫 iteration은 None
    delta: Optional[Dict[str, float]] = None  # 예: {"rmse": -0.13, "pearson": +0.07}
    reason: str = ""  # 이 결과가 나온 이유
    lessons: List[str] = Field(default_factory=list)  # 핵심 교훈


class IterationWeights(BaseModel):
    """Iteration weights for EmbeddingScorer formula.

    Formula: S_total = w_d × S_domain + w_a × [β × S_arch + (1-β) × novelty]

    - w_d: Domain weight (0.9 → 0.1 over iterations, auto-calculated)
    - w_a: Architecture weight (0.1 → 0.9 over iterations, auto-calculated)
    - beta: Novelty coefficient (1.0=similarity focus, 0.0=novelty focus, LLM-adjustable)
    """
    w_d: Optional[float] = None  # Domain weight (auto-calculated)
    w_a: Optional[float] = None  # Architecture weight (auto-calculated)
    beta: Optional[float] = None  # Novelty coefficient (LLM-adjustable)


class BaselineRecord(BaseModel):
    """iter0: 오리지널 모델 베이스라인."""
    model_name: str
    description: str  # 모델 설명
    domain: str = ""  # spatial, dti, drug_repurposing, drp
    performance: Dict[str, float] = Field(default_factory=dict)  # 모델별 성능 지표 직접 저장


class PaperRewardRecord(BaseModel):
    """논문별 reward 추적 레코드.

    Formula: V_i = Sim_i + (w × (N_success - N_failure) / (N_total + 1))
    """
    paper_id: str  # title digest
    title: str
    n_success: int = 0  # 성능 향상에 기여한 횟수
    n_failure: int = 0  # 성능 하락에 기여한 횟수
    n_total: int = 0    # 총 사용 횟수
    last_used_iteration: Optional[int] = None


class IterationRecord(BaseModel):
    """단일 iteration의 전체 기록."""
    iteration: int
    timestamp: datetime = Field(default_factory=datetime.now)
    performance: IterationPerformance = Field(default_factory=IterationPerformance)
    changes: IterationChanges = Field(default_factory=lambda: IterationChanges(component="unknown", description=""))
    analysis: IterationAnalysis = Field(default_factory=IterationAnalysis)
    weights: IterationWeights = Field(default_factory=IterationWeights)

    # 해당 iteration에서 사용한 논문들
    papers_used: List[str] = Field(default_factory=list)

    # 산출물 경로
    debate_outputs_path: Optional[str] = None
    src_path: Optional[str] = None


class EvolvingMemoryData(BaseModel):
    """EvolvingMemory storage - iterations are accumulated."""
    total_iterations: int = 0  # 현재까지 완료된 iteration 수
    planned_iterations: Optional[int] = None  # 원래 계획한 총 iteration 횟수 (--iter N)
    target_model: Optional[str] = None  # 타겟 모델명 (continue 모드용)
    best_iteration: Optional[int] = None
    best_performance: Optional[IterationPerformance] = None

    # iter0: 오리지널 모델 베이스라인
    baseline: Optional[BaselineRecord] = None

    iterations: List[IterationRecord] = Field(default_factory=list)

    # 프롬프트 주입용 누적 학습 내용
    key_lessons: List[str] = Field(default_factory=list)
    failed_approaches: List[str] = Field(default_factory=list)

    # 연속 실패 횟수 추적 (성능 하락 시 증가, 개선 시 리셋)
    consecutive_failures: int = 0

    # 전역 사용된 논문 목록 (모든 iteration에서 사용된 논문 누적)
    used_papers: List[str] = Field(default_factory=list)

    # Paper reward tracking - 논문별 성과 추적
    # Formula: V_i = Sim_i + (w × (N_success - N_failure) / (N_total + 1))
    paper_rewards: Dict[str, PaperRewardRecord] = Field(default_factory=dict)
    # Reward snapshots at block ends (e.g., iter 5, 10, 15...)
    paper_reward_snapshots: Dict[int, Dict[str, float]] = Field(default_factory=dict)
    reward_weight: float = 0.1  # w 파라미터 (--weight 플래그, 기본값 0.1)
    reward_patience: int = 10  # 블록 크기 (--patience 플래그, 기본값 10)

    # 마지막으로 논문 검색을 수행한 iteration (paper_selector에서 사용)
    # paper_selector는 best_iteration이 아닌 이 값의 aggregated_results.json 사용
    last_paper_search_iteration: Optional[int] = None

    # 현재 가중치 (최신 iteration의 가중치 반영)
    current_weights: IterationWeights = Field(default_factory=IterationWeights)

    # 현재 beta 값 (LLM이 자율 조절, 1.0=유사도 중시, 0.0=참신성 중시)
    current_beta: float = 1.0

    # 가중치 변경 플래그 (iteration_critic에서 사용)
    # True = 일반 동작 (prev_improved 체크하여 결정)
    # False = 가중치 변경됨, 다음 iteration은 반드시 새 논문 검색
    # 주의: 이 필드는 "skip" 여부가 아니라 "가중치 변경 여부"를 나타냄
    skip_paper_search: bool = True  # 기본값: 일반 동작
