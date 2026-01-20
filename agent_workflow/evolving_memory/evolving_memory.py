"""Iteration learning evolving memory storage and retrieval."""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .schemas import (
    EvolvingMemoryData,
    IterationRecord,
    IterationPerformance,
    IterationChanges,
    IterationAnalysis,
    IterationWeights,
    BaselineRecord,
    PaperRewardRecord,
)
from agent_workflow.logger import logger
from .baselines import MODEL_BASELINES


class EvolvingMemory:
    """Persistent memory for iteration-based learning.

    Storage location: experiments/evolving_memory/memory.json
    All iteration data is accumulated within a session.
    """

    def __init__(self, workspace_path: str = "experiments"):
        """Initialize EvolvingMemory.

        Args:
            workspace_path: Base path for experiments directory
        """
        self.workspace = Path(workspace_path)
        self.memory_dir = self.workspace / "evolving_memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "memory.json"

        # Load existing data or create new
        self.data = self._load_or_create()

    def _load_or_create(self) -> EvolvingMemoryData:
        """Load existing memory or create new."""
        if self.memory_file.exists():
            try:
                raw = json.loads(self.memory_file.read_text(encoding="utf-8"))
                logger.info(f"[EvolvingMemory] Loaded existing memory: {self.memory_file}")
                return EvolvingMemoryData(**raw)
            except Exception as e:
                logger.warning(f"[EvolvingMemory] Load failed: {e}, creating new")

        logger.info(f"[EvolvingMemory] Creating new memory: {self.memory_file}")
        return EvolvingMemoryData()

    def save(self) -> None:
        """메모리를 디스크에 저장."""
        self.memory_file.write_text(
            self.data.model_dump_json(indent=2),
            encoding="utf-8"
        )
        logger.info(f"[EvolvingMemory] 저장 완료: {self.memory_file}")

    def clear(self) -> None:
        """Clear all data."""
        self.data = EvolvingMemoryData()
        self.save()
        logger.info("[EvolvingMemory] All data cleared")

    def init_baseline(self, target_model: str) -> None:
        """iter0 베이스라인 설정 및 초기 best로 등록.

        baseline이 초기 best가 되어 iter1부터 비교 가능.
        이를 통해 iter1도 성공/실패 판단이 가능해짐.

        Args:
            target_model: 타겟 모델 이름 (stagate, deeptta 등)
        """
        if target_model not in MODEL_BASELINES:
            logger.warning(f"[EvolvingMemory] 베이스라인 없음: {target_model}")
            return

        baseline_data = MODEL_BASELINES[target_model]

        self.data.baseline = BaselineRecord(
            model_name=target_model,
            description=baseline_data["description"],
            domain=baseline_data.get("domain", ""),
            performance=baseline_data["performance"],  # 직접 저장
        )

        # baseline을 초기 best로 설정 (iter1이 baseline과 비교 가능하도록)
        baseline_perf = baseline_data["performance"]
        perf_dict = {}
        custom_dict = {}
        for key, val in baseline_perf.items():
            if key in self.STANDARD_PERF_FIELDS:
                perf_dict[key] = val
            else:
                custom_dict[key] = val
        perf_dict['custom_metrics'] = custom_dict

        self.data.best_iteration = 0  # baseline = iter0
        self.data.best_performance = IterationPerformance(**perf_dict)

        self.save()
        logger.info(f"[EvolvingMemory] 베이스라인 설정 (초기 best로 등록): {target_model}")

    def set_session_info(self, planned_iterations: int, target_model: str) -> None:
        """세션 정보 설정 (총 iteration 횟수, 타겟 모델).

        일반 모드에서 init_iteration_node가 호출.
        Continue 모드에서 이 정보를 읽어서 재개.

        Args:
            planned_iterations: 원래 계획한 총 iteration 횟수 (--iter N)
            target_model: 타겟 모델명
        """
        self.data.planned_iterations = planned_iterations
        self.data.target_model = target_model
        self.save()
        logger.info(f"[EvolvingMemory] 세션 정보 설정: planned={planned_iterations}, model={target_model}")

    def get_session_info(self) -> Dict[str, Any]:
        """세션 정보 조회.

        Returns:
            {
                "planned_iterations": int or None,
                "target_model": str or None,
                "completed_iterations": int (완료된 iteration 수)
            }
        """
        return {
            "planned_iterations": self.data.planned_iterations,
            "target_model": self.data.target_model,
            "completed_iterations": self.data.total_iterations,
        }

    def set_reward_settings(self, patience: int, weight: float) -> None:
        """Reward 설정 (--patience, --weight 플래그).

        Args:
            patience: Reward 블록 크기 (기본값 10)
            weight: Reward 가중치 (기본값 0.1)
        """
        self.data.reward_patience = patience
        self.data.reward_weight = weight
        self.save()
        logger.info(f"[EvolvingMemory] Reward 설정: patience={patience}, weight={weight}")

    def get_reward_settings(self) -> Dict[str, Any]:
        """Reward 설정 조회.

        Returns:
            {"patience": int, "weight": float}
        """
        return {
            "patience": self.data.reward_patience,
            "weight": self.data.reward_weight,
        }

    # 표준 성능 메트릭 필드 (나머지는 custom_metrics로 분류)
    # DRP: rmse, mse, mae, pcc, scc, pearson, spearman
    # Spatial: ari, nmi, silhouette
    # DTI/Drug Repurposing: accuracy, auroc, auprc, f1
    STANDARD_PERF_FIELDS = {
        'rmse', 'mse', 'mae', 'pcc', 'scc', 'pearson', 'spearman',  # DRP
        'ari', 'nmi', 'silhouette',  # Spatial
        'accuracy', 'auroc', 'auprc', 'f1', 'loss'  # DTI/Drug Repurposing
    }

    def add_iteration(
        self,
        iteration: int,
        performance: Dict[str, float],
        changes: Dict[str, Any],
        analysis: Dict[str, Any],
        artifacts: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None,
        papers_used: Optional[List[str]] = None,
    ) -> IterationRecord:
        """새 iteration 기록 추가.

        Args:
            iteration: iteration 번호 (1부터 시작)
            performance: 메트릭 딕셔너리 (rmse, pearson 등)
            changes: 변경 사항 (component, description, files_modified)
            analysis: 분석 결과 (improved, delta, reason, lessons)
            artifacts: 산출물 경로 (debate_outputs_path, src_path)
            weights: 가중치 (domain, architecture, novelty)
            papers_used: 해당 iteration에서 사용한 논문 목록

        Returns:
            생성된 IterationRecord
        """
        # 표준 필드와 커스텀 필드 분리
        if performance:
            perf_dict = {}
            custom_dict = {}
            for key, val in performance.items():
                if key in self.STANDARD_PERF_FIELDS:
                    perf_dict[key] = val
                else:
                    custom_dict[key] = val
            perf_dict['custom_metrics'] = custom_dict
            perf_obj = IterationPerformance(**perf_dict)
        else:
            perf_obj = IterationPerformance()

        record = IterationRecord(
            iteration=iteration,
            timestamp=datetime.now(),
            performance=perf_obj,
            changes=IterationChanges(**changes) if changes else IterationChanges(component="unknown", description=""),
            analysis=IterationAnalysis(**analysis) if analysis else IterationAnalysis(),
            weights=IterationWeights(**weights) if weights else IterationWeights(),
            papers_used=papers_used or [],
            debate_outputs_path=artifacts.get("debate_outputs_path") if artifacts else None,
            src_path=artifacts.get("src_path") if artifacts else None,
        )

        # === 중복 체크: 같은 iteration 번호가 이미 있으면 업데이트 ===
        existing_idx = None
        for idx, existing_record in enumerate(self.data.iterations):
            if existing_record.iteration == iteration:
                existing_idx = idx
                break

        if existing_idx is not None:
            # 새 메트릭이 유효한 경우에만 업데이트
            new_has_metrics = performance and any(v is not None for v in performance.values())
            if new_has_metrics:
                self.data.iterations[existing_idx] = record
                logger.info(f"[EvolvingMemory] iteration {iteration} 업데이트 (기존 데이터 대체)")
            else:
                logger.info(f"[EvolvingMemory] iteration {iteration} 이미 존재, 새 메트릭 없어서 스킵")
                return self.data.iterations[existing_idx]
        else:
            # 새 record 추가
            self.data.iterations.append(record)

        self.data.total_iterations = len(self.data.iterations)

        # 최고 성능 업데이트
        self._update_best(record)

        # 교훈 누적
        self._accumulate_lessons(record)

        # 연속 실패 카운트 및 전역 데이터 업데이트
        self._update_global_state(record)

        self.save()
        logger.info(f"[EvolvingMemory] iteration {iteration} 추가 완료")
        return record

    # 메트릭 방향 정의: True = 높을수록 좋음, False = 낮을수록 좋음
    METRIC_HIGHER_BETTER = {
        # DRP (lower is better)
        'rmse': False, 'mse': False, 'mae': False, 'loss': False,
        # DRP (higher is better)
        'pcc': True, 'scc': True, 'pearson': True, 'spearman': True,
        # Spatial (higher is better)
        'ari': True, 'nmi': True, 'silhouette': True,
        # DTI/Drug Repurposing (higher is better)
        'accuracy': True, 'auroc': True, 'auprc': True, 'f1': True,
        'auc': True, 'roc_auc': True,
    }

    # Model-specific primary metrics (takes priority over DOMAIN_PRIMARY_METRIC)
    MODEL_PRIMARY_METRIC = {
        'hyperattentiondti': 'auprc',
        'dlm-dti': 'auprc',
    }

    STRICT_PRIMARY_METRIC_MODELS = set(MODEL_PRIMARY_METRIC.keys())

    DOMAIN_PRIMARY_METRIC = {
        'drp': 'rmse',           # Drug Response Prediction
        'drug_response': 'rmse',
        'spatial': 'ari',        # Spatial Clustering
        'dti': 'auprc',          # Drug-Target Interaction
    }

    def _get_current_model_name(self) -> Optional[str]:
        """Get current model name from baseline or session info."""
        if self.data.baseline and self.data.baseline.model_name:
            return self.data.baseline.model_name.lower()
        if self.data.target_model:
            return self.data.target_model.lower()
        return None

    def _get_primary_metric(self, performance: IterationPerformance) -> tuple[Optional[str], Optional[float], bool]:
        """성능에서 도메인 기반 주요 메트릭 추출.

        도메인별 주요 메트릭:
        - drp (Drug Response): RMSE
        - spatial: ARI
        - dti/dta: AUROC (model overrides may apply)

        Returns:
            (metric_name, value, higher_is_better) 튜플
        """
        # 도메인 확인
        domain = None
        if self.data.baseline and self.data.baseline.domain:
            domain = self.data.baseline.domain.lower()

        model_name = self._get_current_model_name()
        primary_metric = None
        strict_primary = False
        if model_name and model_name in self.MODEL_PRIMARY_METRIC:
            primary_metric = self.MODEL_PRIMARY_METRIC[model_name]
            strict_primary = True
        else:
            # 도메인 기반 주요 메트릭 결정
            primary_metric = self.DOMAIN_PRIMARY_METRIC.get(domain)

        if primary_metric:
            # 표준 메트릭에서 확인
            val = getattr(performance, primary_metric, None)
            if val is not None:
                higher = self.METRIC_HIGHER_BETTER.get(primary_metric, True)
                return (primary_metric, val, higher)

            # custom_metrics에서 확인
            if performance.custom_metrics and primary_metric in performance.custom_metrics:
                val = performance.custom_metrics[primary_metric]
                if val is not None:
                    higher = self.METRIC_HIGHER_BETTER.get(primary_metric, True)
                    return (primary_metric, val, higher)

            if strict_primary:
                higher = self.METRIC_HIGHER_BETTER.get(primary_metric, True)
                return (primary_metric, None, higher)

        # 도메인 메트릭 없으면 fallback: 있는 메트릭 중 우선순위대로
        fallback_order = ['rmse', 'ari', 'auprc', 'auroc', 'pearson', 'nmi', 'accuracy', 'silhouette', 'mse']

        for metric in fallback_order:
            val = getattr(performance, metric, None)
            if val is not None:
                higher = self.METRIC_HIGHER_BETTER.get(metric, True)
                return (metric, val, higher)

        # custom_metrics에서 fallback
        if performance.custom_metrics:
            for metric in fallback_order:
                if metric in performance.custom_metrics and performance.custom_metrics[metric] is not None:
                    val = performance.custom_metrics[metric]
                    higher = self.METRIC_HIGHER_BETTER.get(metric, True)
                    return (metric, val, higher)

        return (None, None, True)

    def _update_best(self, record: IterationRecord) -> None:
        """최고 성능 iteration 업데이트 (주요 메트릭 기준)."""
        metric_name, curr_val, higher_is_better = self._get_primary_metric(record.performance)

        if curr_val is None:
            return

        if self.data.best_performance is None:
            self.data.best_iteration = record.iteration
            self.data.best_performance = record.performance
            logger.info(f"[EvolvingMemory] 첫 best 설정: iteration {record.iteration} ({metric_name}: {curr_val:.4f})")
            return

        # best에서 같은 메트릭 추출
        _, best_val, _ = self._get_primary_metric(self.data.best_performance)

        if best_val is None:
            self.data.best_iteration = record.iteration
            self.data.best_performance = record.performance
            return

        # 비교: higher_is_better면 curr > best가 개선, 아니면 curr < best가 개선
        is_better = (curr_val > best_val) if higher_is_better else (curr_val < best_val)

        if is_better:
            self.data.best_iteration = record.iteration
            self.data.best_performance = record.performance
            direction = "↑" if higher_is_better else "↓"
            logger.info(f"[EvolvingMemory] 새로운 최고 성능: iteration {record.iteration} ({metric_name}: {curr_val:.4f} {direction})")

    def _accumulate_lessons(self, record: IterationRecord) -> None:
        """iteration에서 얻은 교훈 누적."""
        # 새 교훈 추가
        for lesson in record.analysis.lessons:
            if lesson and lesson not in self.data.key_lessons:
                self.data.key_lessons.append(lesson)

        # 실패한 접근법 추적
        if record.analysis.improved is False:
            approach = f"{record.changes.component}: {record.changes.description}"
            if approach not in self.data.failed_approaches:
                self.data.failed_approaches.append(approach)

        # 리스트 크기 관리 (교훈 20개, 실패 10개 유지)
        self.data.key_lessons = self.data.key_lessons[-20:]
        self.data.failed_approaches = self.data.failed_approaches[-10:]

    def _update_global_state(self, record: IterationRecord) -> None:
        """연속 실패 카운트, 전역 논문 목록, 현재 가중치 업데이트."""
        # 연속 실패 카운트 업데이트 (best 성능 대비)
        # best보다 나쁘면 실패, best 갱신하면 리셋
        metric_name, curr_val, higher_is_better = self._get_primary_metric(record.performance)

        if curr_val is not None and self.data.best_performance is not None:
            _, best_val, _ = self._get_primary_metric(self.data.best_performance)

            if best_val is not None:
                # best 대비 비교
                is_worse = (curr_val < best_val) if higher_is_better else (curr_val > best_val)

                if is_worse:
                    # best보다 나쁨 → 실패
                    self.data.consecutive_failures += 1
                    cmp = "<" if higher_is_better else ">"
                    logger.info(f"[EvolvingMemory] 연속 실패 (best 대비): {self.data.consecutive_failures}회 ({metric_name}: {curr_val:.4f} {cmp} best {best_val:.4f})")
                else:
                    # best와 같거나 나음 → 리셋
                    if self.data.consecutive_failures > 0:
                        logger.info(f"[EvolvingMemory] 연속 실패 리셋 (이전: {self.data.consecutive_failures}회)")
                    self.data.consecutive_failures = 0
            else:
                # best_val이 없으면 fallback
                self._update_failures_by_improved(record)
        else:
            # curr_val이 없거나 best가 없으면 fallback
            self._update_failures_by_improved(record)

        # 전역 논문/가중치 업데이트 (항상 실행)
        self._update_papers_and_weights(record)

    def _update_failures_by_improved(self, record: IterationRecord) -> None:
        """improved 플래그 기반으로 연속 실패 업데이트 (fallback)."""
        if record.analysis.improved is False:
            self.data.consecutive_failures += 1
            logger.info(f"[EvolvingMemory] 연속 실패 (직전 대비): {self.data.consecutive_failures}회")
        elif record.analysis.improved is True:
            if self.data.consecutive_failures > 0:
                logger.info(f"[EvolvingMemory] 연속 실패 리셋 (이전: {self.data.consecutive_failures}회)")
            self.data.consecutive_failures = 0

    def _update_papers_and_weights(self, record: IterationRecord) -> None:
        """전역 논문 목록과 현재 가중치 업데이트."""
        # 전역 논문 목록 업데이트 (중복 제거)
        for paper in record.papers_used:
            if paper and paper not in self.data.used_papers:
                self.data.used_papers.append(paper)

        # 현재 가중치 업데이트
        if record.weights:
            self.data.current_weights = record.weights

    def get_last_iteration(self) -> Optional[IterationRecord]:
        """가장 최근 iteration 조회."""
        if self.data.iterations:
            return self.data.iterations[-1]
        return None

    def get_2nd_best_iteration(self) -> Optional[int]:
        """성능 순위 2등 iteration 반환.

        baseline(iter 0)도 후보에 포함.
        best_iteration과 동일하면 None 반환.

        Returns:
            2nd best iteration 번호 (없으면 None)
        """
        # 후보 목록: (iteration_number, performance)
        candidates: List[tuple] = []

        # baseline (iter 0) 추가
        if self.data.baseline and self.data.baseline.performance:
            baseline_perf = self.data.baseline.performance
            # baseline.performance는 dict 형태이므로 IterationPerformance로 변환
            if isinstance(baseline_perf, dict):
                perf_dict = {}
                custom_dict = {}
                for key, val in baseline_perf.items():
                    if key in self.STANDARD_PERF_FIELDS:
                        perf_dict[key] = val
                    else:
                        custom_dict[key] = val
                perf_dict['custom_metrics'] = custom_dict
                baseline_perf_obj = IterationPerformance(**perf_dict)
            else:
                baseline_perf_obj = baseline_perf
            candidates.append((0, baseline_perf_obj))

        # 모든 iteration 추가
        for record in self.data.iterations:
            candidates.append((record.iteration, record.performance))

        # 후보가 2개 미만이면 2등이 없음
        if len(candidates) < 2:
            return None

        # 주요 메트릭 기준으로 정렬
        _, _, higher_is_better = self._get_primary_metric(candidates[0][1])

        def get_metric_value(perf: IterationPerformance) -> float:
            _, val, _ = self._get_primary_metric(perf)
            if val is None:
                return float('-inf') if higher_is_better else float('inf')
            return val

        # 성능 순으로 정렬 (best가 앞에 오도록)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: get_metric_value(x[1]),
            reverse=higher_is_better  # higher_is_better면 내림차순, 아니면 오름차순
        )

        # best를 제외하고 2등 반환
        best_iter = self.data.best_iteration
        for iter_num, _ in sorted_candidates:
            if iter_num != best_iter:
                return iter_num

        return None

    def get_prompt_context(self) -> str:
        """프롬프트 주입용 컨텍스트 문자열 생성.

        Returns:
            마크다운 형식의 컨텍스트:
            - 직전 iteration 결과
            - 현재까지 최고 성능
            - 누적된 핵심 교훈
            - 피해야 할 실패한 접근법
        """
        if not self.data.iterations:
            return ""

        lines = ["## Previous Iteration Results", ""]

        # 직전 iteration
        last = self.data.iterations[-1]
        lines.append(f"### Iteration {last.iteration} (직전)")

        # 성능
        perf_parts = []
        if last.performance.rmse is not None:
            perf_parts.append(f"RMSE {last.performance.rmse:.4f}")
        if last.performance.pearson is not None:
            perf_parts.append(f"Pearson {last.performance.pearson:.4f}")
        if perf_parts:
            lines.append(f"- **Performance**: {', '.join(perf_parts)}")

        # 변경 사항
        lines.append(f"- **Changed**: {last.changes.component} - {last.changes.description}")

        # 결과
        if last.analysis.improved is True:
            delta_str = self._format_delta(last.analysis.delta)
            lines.append(f"- **Result**: ✅ 성능 개선 ({delta_str})")
        elif last.analysis.improved is False:
            delta_str = self._format_delta(last.analysis.delta)
            lines.append(f"- **Result**: ❌ 성능 하락 ({delta_str})")
        else:
            lines.append("- **Result**: 첫 번째 iteration - baseline")

        # 이유와 교훈
        if last.analysis.reason:
            lines.append(f"- **Reason**: {last.analysis.reason}")
        if last.analysis.lessons:
            lines.append(f"- **Lesson**: {last.analysis.lessons[0]}")

        # 현재까지 최고 성능
        if self.data.best_iteration and self.data.best_performance:
            lines.append("")
            lines.append(f"### Best So Far (Iteration {self.data.best_iteration})")
            best_parts = []
            if self.data.best_performance.rmse is not None:
                best_parts.append(f"RMSE {self.data.best_performance.rmse:.4f}")
            if self.data.best_performance.pearson is not None:
                best_parts.append(f"Pearson {self.data.best_performance.pearson:.4f}")
            if best_parts:
                lines.append(f"- **Performance**: {', '.join(best_parts)}")

        # 핵심 교훈
        if self.data.key_lessons:
            lines.append("")
            lines.append("### Key Lessons (누적)")
            for lesson in self.data.key_lessons[-5:]:
                lines.append(f"- {lesson}")

        # 실패한 접근법
        if self.data.failed_approaches:
            lines.append("")
            lines.append("### Avoid (실패한 접근)")
            for approach in self.data.failed_approaches[-5:]:
                lines.append(f"- {approach}")

        return "\n".join(lines)

    def _format_delta(self, delta: Optional[Dict[str, float]]) -> str:
        """delta 딕셔너리를 보기 좋게 포맷팅."""
        if not delta:
            return ""
        parts = []
        for key, val in delta.items():
            sign = "+" if val > 0 else ""
            parts.append(f"{key.upper()} {sign}{val:.4f}")
        return ", ".join(parts)

    # =========================================================================
    # Paper Reward System
    # Formula: V_i = Sim_i + (w × (N_success - N_failure) / (N_total + 1))
    # =========================================================================

    def update_paper_rewards(
        self,
        papers_used: List[str],
        iteration: int,
        improved: Optional[bool]
    ) -> None:
        """iteration 결과에 따라 논문들의 reward 업데이트.

        Args:
            papers_used: 해당 iteration에서 사용된 논문 제목 목록
            iteration: 현재 iteration 번호
            improved: 성능 향상 여부 (True=향상, False=하락, None=baseline/판단불가)
        """
        papers_used = papers_used or []
        updated = False

        if papers_used:
            for title in papers_used:
                paper_id = self._get_paper_id_from_title(title)

                # Get or create reward record
                if paper_id not in self.data.paper_rewards:
                    self.data.paper_rewards[paper_id] = PaperRewardRecord(
                        paper_id=paper_id,
                        title=title
                    )

                record = self.data.paper_rewards[paper_id]
                record.n_total += 1
                record.last_used_iteration = iteration

                if improved is True:
                    record.n_success += 1
                elif improved is False:
                    record.n_failure += 1
                # None (baseline/첫 iteration) → success/failure 업데이트 안함

            updated = True

        snapshot_saved = False
        if iteration % self.data.reward_patience == 0:
            snapshot = self.get_all_paper_rewards()
            self.data.paper_reward_snapshots[iteration] = snapshot
            snapshot_saved = True
            next_block_start = iteration + 1
            next_block_end = iteration + self.data.reward_patience
            logger.info("=" * 60)
            logger.info(f"[REWARD UPDATE] iter {iteration} 스냅샷 저장!")
            logger.info(f"  - 저장된 논문 수: {len(snapshot)}개")
            logger.info(f"  - 다음 블록 (iter {next_block_start}~{next_block_end})에서 이 스냅샷 적용")
            logger.info(f"  - patience={self.data.reward_patience}, weight={self.data.reward_weight}")
            logger.info("=" * 60)

        if updated or snapshot_saved:
            self.save()

        if updated:
            logger.info(f"[EvolvingMemory] Paper rewards 업데이트: {len(papers_used)}개 논문 (improved={improved})")

    def get_all_paper_rewards(self) -> Dict[str, float]:
        """모든 논문의 reward contribution 반환.

        Returns:
            Dict[paper_id, reward_score]: 각 논문의 reward 기여분
            Formula: reward = w × (N_success - N_failure) / (N_total + 1)
        """
        rewards = {}
        w = self.data.reward_weight

        for paper_id, record in self.data.paper_rewards.items():
            # V_i = Sim_i + reward 에서 reward 부분만 반환
            # Sim_i는 embedding_scorer에서 별도로 계산
            reward = w * (record.n_success - record.n_failure) / (record.n_total + 1)
            rewards[paper_id] = reward

        return rewards

    def get_paper_reward_snapshot(self, iteration: int) -> Optional[Dict[str, float]]:
        """블록 종료 iteration에서 저장된 reward 스냅샷 조회."""
        return self.data.paper_reward_snapshots.get(iteration)

    def save_paper_reward_snapshot(self, iteration: int) -> Dict[str, float]:
        """블록 종료 iteration의 reward 스냅샷을 저장."""
        snapshot = self.get_all_paper_rewards()
        self.data.paper_reward_snapshots[iteration] = snapshot
        self.save()
        logger.info(f"[EvolvingMemory] Paper rewards 스냅샷 저장: iter {iteration} ({len(snapshot)}개)")
        return snapshot

    def _get_paper_id_from_title(self, title: str) -> str:
        """논문 제목에서 paper_id 생성 (EmbeddingScorer와 동일한 방식)."""
        normalized = title.lower().strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
