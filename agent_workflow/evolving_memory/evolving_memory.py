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
        """Save memory to disk."""
        self.memory_file.write_text(
            self.data.model_dump_json(indent=2),
            encoding="utf-8"
        )
        logger.info(f"[EvolvingMemory] Saved: {self.memory_file}")

    def clear(self) -> None:
        """Clear all data."""
        self.data = EvolvingMemoryData()
        self.save()
        logger.info("[EvolvingMemory] All data cleared")

    def init_baseline(self, target_model: str) -> None:
        """Set iter0 baseline and register as initial best.

        Baseline becomes the initial best, enabling comparison from iter1.
        This allows success/failure determination starting from iter1.

        Args:
            target_model: Target model name (stagate, deeptta, etc.)
        """
        if target_model not in MODEL_BASELINES:
            logger.warning(f"[EvolvingMemory] No baseline for: {target_model}")
            return

        baseline_data = MODEL_BASELINES[target_model]

        self.data.baseline = BaselineRecord(
            model_name=target_model,
            description=baseline_data["description"],
            domain=baseline_data.get("domain", ""),
            performance=baseline_data["performance"],
        )

        # Set baseline as initial best (so iter1 can compare against baseline)
        baseline_perf = baseline_data["performance"]
        perf_dict = {k: v for k, v in baseline_perf.items() if k in self.STANDARD_PERF_FIELDS}

        self.data.best_iteration = 0
        self.data.best_performance = IterationPerformance(**perf_dict)

        self.save()
        logger.info(f"[EvolvingMemory] Baseline set (registered as initial best): {target_model}")

    def set_session_info(self, planned_iterations: int, target_model: str) -> None:
        """Set session info (total iterations, target model).

        Called by init_iteration_node in normal mode.
        Continue mode reads this info to resume.

        Args:
            planned_iterations: Originally planned total iterations (--iter N)
            target_model: Target model name
        """
        self.data.planned_iterations = planned_iterations
        self.data.target_model = target_model
        self.save()
        logger.info(f"[EvolvingMemory] Session info set: planned={planned_iterations}, model={target_model}")

    def get_session_info(self) -> Dict[str, Any]:
        """Get session info.

        Returns:
            {
                "planned_iterations": int or None,
                "target_model": str or None,
                "completed_iterations": int (number of completed iterations)
            }
        """
        return {
            "planned_iterations": self.data.planned_iterations,
            "target_model": self.data.target_model,
            "completed_iterations": self.data.total_iterations,
        }

    def set_reward_settings(self, patience: int, weight: float) -> None:
        """Set reward settings (--patience, --weight flags).

        Args:
            patience: Reward block size (default 10)
            weight: Reward weight (default 0.1)
        """
        self.data.reward_patience = patience
        self.data.reward_weight = weight
        self.save()
        logger.info(f"[EvolvingMemory] Reward settings: patience={patience}, weight={weight}")

    def get_reward_settings(self) -> Dict[str, Any]:
        """Get reward settings.

        Returns:
            {"patience": int, "weight": float}
        """
        return {
            "patience": self.data.reward_patience,
            "weight": self.data.reward_weight,
        }

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
        """Add new iteration record.

        Args:
            iteration: Iteration number (starts from 1)
            performance: Metrics dict (rmse, pearson, etc.)
            changes: Changes made (component, description, files_modified)
            analysis: Analysis results (improved, delta, reason, lessons)
            artifacts: Artifact paths (debate_outputs_path, src_path)
            weights: Weights (domain, architecture, novelty)
            papers_used: Papers used in this iteration

        Returns:
            Created IterationRecord
        """
        # Extract standard fields only
        if performance:
            perf_dict = {k: v for k, v in performance.items() if k in self.STANDARD_PERF_FIELDS}
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

        # Duplicate check: update if same iteration number exists
        existing_idx = None
        for idx, existing_record in enumerate(self.data.iterations):
            if existing_record.iteration == iteration:
                existing_idx = idx
                break

        if existing_idx is not None:
            new_has_metrics = performance and any(v is not None for v in performance.values())
            if new_has_metrics:
                self.data.iterations[existing_idx] = record
                logger.info(f"[EvolvingMemory] Iteration {iteration} updated (replaced existing)")
            else:
                logger.info(f"[EvolvingMemory] Iteration {iteration} exists, no new metrics, skipped")
                return self.data.iterations[existing_idx]
        else:
            self.data.iterations.append(record)

        self.data.total_iterations = len(self.data.iterations)
        self._update_best(record)
        self._accumulate_lessons(record)
        self._update_global_state(record)
        self.save()
        logger.info(f"[EvolvingMemory] Iteration {iteration} added")
        return record

    # Metric direction: True = higher is better, False = lower is better
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
        'drp': 'rmse',
        'drug_response': 'rmse',
        'spatial': 'ari',
        'dti': 'auprc',
    }

    def _get_current_model_name(self) -> Optional[str]:
        """Get current model name from baseline or session info."""
        if self.data.baseline and self.data.baseline.model_name:
            return self.data.baseline.model_name.lower()
        if self.data.target_model:
            return self.data.target_model.lower()
        return None

    def _get_primary_metric(self, performance: IterationPerformance) -> tuple[Optional[str], Optional[float], bool]:
        """Extract domain-based primary metric from performance.

        Domain-specific primary metrics:
        - drp (Drug Response): RMSE
        - spatial: ARI
        - dti/dta: AUROC (model overrides may apply)

        Returns:
            (metric_name, value, higher_is_better) tuple
        """

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
            primary_metric = self.DOMAIN_PRIMARY_METRIC.get(domain)

        if primary_metric:
            val = getattr(performance, primary_metric, None)
            if val is not None:
                higher = self.METRIC_HIGHER_BETTER.get(primary_metric, True)
                return (primary_metric, val, higher)

            if strict_primary:
                higher = self.METRIC_HIGHER_BETTER.get(primary_metric, True)
                return (primary_metric, None, higher)

        fallback_order = ['rmse', 'ari', 'auprc', 'auroc', 'pearson', 'nmi', 'accuracy', 'silhouette', 'mse']

        for metric in fallback_order:
            val = getattr(performance, metric, None)
            if val is not None:
                higher = self.METRIC_HIGHER_BETTER.get(metric, True)
                return (metric, val, higher)

        return (None, None, True)

    def _update_best(self, record: IterationRecord) -> None:
        """Update best iteration (based on primary metric)."""
        metric_name, curr_val, higher_is_better = self._get_primary_metric(record.performance)

        if curr_val is None:
            return

        if self.data.best_performance is None:
            self.data.best_iteration = record.iteration
            self.data.best_performance = record.performance
            logger.info(f"[EvolvingMemory] First best set: iteration {record.iteration} ({metric_name}: {curr_val:.4f})")
            return

        _, best_val, _ = self._get_primary_metric(self.data.best_performance)

        if best_val is None:
            self.data.best_iteration = record.iteration
            self.data.best_performance = record.performance
            return

        is_better = (curr_val > best_val) if higher_is_better else (curr_val < best_val)

        if is_better:
            self.data.best_iteration = record.iteration
            self.data.best_performance = record.performance
            direction = "↑" if higher_is_better else "↓"
            logger.info(f"[EvolvingMemory] New best: iteration {record.iteration} ({metric_name}: {curr_val:.4f} {direction})")

    def _accumulate_lessons(self, record: IterationRecord) -> None:
        """Accumulate lessons from iteration."""
        # Add new lessons
        for lesson in record.analysis.lessons:
            if lesson and lesson not in self.data.key_lessons:
                self.data.key_lessons.append(lesson)

        # Track failed approaches
        if record.analysis.improved is False:
            approach = f"{record.changes.component}: {record.changes.description}"
            if approach not in self.data.failed_approaches:
                self.data.failed_approaches.append(approach)

        # Manage list sizes (keep 20 lessons, 10 failures)
        self.data.key_lessons = self.data.key_lessons[-20:]
        self.data.failed_approaches = self.data.failed_approaches[-10:]

    def _update_global_state(self, record: IterationRecord) -> None:
        """Update consecutive failure count, global papers, and current weights."""
        # Update consecutive failures (compared to best performance)
        # Worse than best = failure, beating best = reset
        metric_name, curr_val, higher_is_better = self._get_primary_metric(record.performance)

        if curr_val is not None and self.data.best_performance is not None:
            _, best_val, _ = self._get_primary_metric(self.data.best_performance)

            if best_val is not None:
                # Compare against best
                is_worse = (curr_val < best_val) if higher_is_better else (curr_val > best_val)

                if is_worse:
                    # Worse than best = failure
                    self.data.consecutive_failures += 1
                    cmp = "<" if higher_is_better else ">"
                    logger.info(f"[EvolvingMemory] Consecutive failure (vs best): {self.data.consecutive_failures} ({metric_name}: {curr_val:.4f} {cmp} best {best_val:.4f})")
                else:
                    # Equal or better than best = reset
                    if self.data.consecutive_failures > 0:
                        logger.info(f"[EvolvingMemory] Consecutive failures reset (was: {self.data.consecutive_failures})")
                    self.data.consecutive_failures = 0
            else:
                # Fallback if best_val is None
                self._update_failures_by_improved(record)
        else:
            # Fallback if curr_val is None or no best
            self._update_failures_by_improved(record)

        # Update global papers and weights (always runs)
        self._update_papers_and_weights(record)

    def _update_failures_by_improved(self, record: IterationRecord) -> None:
        """Update consecutive failures based on improved flag (fallback)."""
        if record.analysis.improved is False:
            self.data.consecutive_failures += 1
            logger.info(f"[EvolvingMemory] Consecutive failure (vs prev): {self.data.consecutive_failures}")
        elif record.analysis.improved is True:
            if self.data.consecutive_failures > 0:
                logger.info(f"[EvolvingMemory] Consecutive failures reset (was: {self.data.consecutive_failures})")
            self.data.consecutive_failures = 0

    def _update_papers_and_weights(self, record: IterationRecord) -> None:
        """Update global paper list and current weights."""
        # Update global paper list (deduplicated)
        for paper in record.papers_used:
            if paper and paper not in self.data.used_papers:
                self.data.used_papers.append(paper)

        # Update current weights
        if record.weights:
            self.data.current_weights = record.weights

    def get_last_iteration(self) -> Optional[IterationRecord]:
        """Get most recent iteration."""
        if self.data.iterations:
            return self.data.iterations[-1]
        return None

    def get_2nd_best_iteration(self) -> Optional[int]:
        """Get 2nd best iteration by performance.

        Includes baseline (iter 0) as candidate.
        Returns None if same as best_iteration.

        Returns:
            2nd best iteration number (None if not found)
        """
        # Candidate list: (iteration_number, performance)
        candidates: List[tuple] = []

        # Add baseline (iter 0)
        if self.data.baseline and self.data.baseline.performance:
            baseline_perf = self.data.baseline.performance
            # Convert baseline.performance dict to IterationPerformance
            if isinstance(baseline_perf, dict):
                perf_dict = {k: v for k, v in baseline_perf.items() if k in self.STANDARD_PERF_FIELDS}
                baseline_perf_obj = IterationPerformance(**perf_dict)
            else:
                baseline_perf_obj = baseline_perf
            candidates.append((0, baseline_perf_obj))

        # Add all iterations
        for record in self.data.iterations:
            candidates.append((record.iteration, record.performance))

        # Need at least 2 candidates for 2nd place
        if len(candidates) < 2:
            return None

        # Sort by primary metric
        _, _, higher_is_better = self._get_primary_metric(candidates[0][1])

        def get_metric_value(perf: IterationPerformance) -> float:
            _, val, _ = self._get_primary_metric(perf)
            if val is None:
                return float('-inf') if higher_is_better else float('inf')
            return val

        # Sort by performance (best first)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: get_metric_value(x[1]),
            reverse=higher_is_better  # Descending if higher_is_better, ascending otherwise
        )

        # Return 2nd place (excluding best)
        best_iter = self.data.best_iteration
        for iter_num, _ in sorted_candidates:
            if iter_num != best_iter:
                return iter_num

        return None

    def get_prompt_context(self) -> str:
        """Generate context string for prompt injection.

        Returns:
            Markdown-formatted context:
            - Previous iteration results
            - Best performance so far
            - Accumulated key lessons
            - Failed approaches to avoid
        """
        if not self.data.iterations:
            return ""

        lines = ["## Previous Iteration Results", ""]

        # Previous iteration
        last = self.data.iterations[-1]
        lines.append(f"### Iteration {last.iteration} (Previous)")

        # Performance
        perf_parts = []
        if last.performance.rmse is not None:
            perf_parts.append(f"RMSE {last.performance.rmse:.4f}")
        if last.performance.pearson is not None:
            perf_parts.append(f"Pearson {last.performance.pearson:.4f}")
        if perf_parts:
            lines.append(f"- **Performance**: {', '.join(perf_parts)}")

        # Changes
        lines.append(f"- **Changed**: {last.changes.component} - {last.changes.description}")

        # Result
        if last.analysis.improved is True:
            delta_str = self._format_delta(last.analysis.delta)
            lines.append(f"- **Result**: ✅ Improved ({delta_str})")
        elif last.analysis.improved is False:
            delta_str = self._format_delta(last.analysis.delta)
            lines.append(f"- **Result**: ❌ Degraded ({delta_str})")
        else:
            lines.append("- **Result**: First iteration - baseline")

        # Reason and lessons
        if last.analysis.reason:
            lines.append(f"- **Reason**: {last.analysis.reason}")
        if last.analysis.lessons:
            lines.append(f"- **Lesson**: {last.analysis.lessons[0]}")

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

        # Key lessons
        if self.data.key_lessons:
            lines.append("")
            lines.append("### Key Lessons (Accumulated)")
            for lesson in self.data.key_lessons[-5:]:
                lines.append(f"- {lesson}")

        # Failed approaches
        if self.data.failed_approaches:
            lines.append("")
            lines.append("### Avoid (Failed Approaches)")
            for approach in self.data.failed_approaches[-5:]:
                lines.append(f"- {approach}")

        return "\n".join(lines)

    def _format_delta(self, delta: Optional[Dict[str, float]]) -> str:
        """Format delta dict for display."""
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
        """Update paper rewards based on iteration results.

        Args:
            papers_used: Paper titles used in this iteration
            iteration: Current iteration number
            improved: Performance improvement (True=improved, False=degraded, None=baseline/unknown)
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
                # None (baseline/first iteration) = no success/failure update

            updated = True

        snapshot_saved = False
        if iteration % self.data.reward_patience == 0:
            snapshot = self.get_all_paper_rewards()
            self.data.paper_reward_snapshots[iteration] = snapshot
            snapshot_saved = True
            next_block_start = iteration + 1
            next_block_end = iteration + self.data.reward_patience
            logger.info("=" * 60)
            logger.info(f"[REWARD UPDATE] iter {iteration} snapshot saved!")
            logger.info(f"  - Papers saved: {len(snapshot)}")
            logger.info(f"  - Next block (iter {next_block_start}~{next_block_end}) will use this snapshot")
            logger.info(f"  - patience={self.data.reward_patience}, weight={self.data.reward_weight}")
            logger.info("=" * 60)

        if updated or snapshot_saved:
            self.save()

        if updated:
            logger.info(f"[EvolvingMemory] Paper rewards updated: {len(papers_used)} papers (improved={improved})")

    def get_all_paper_rewards(self) -> Dict[str, float]:
        """Get reward contribution for all papers.

        Returns:
            Dict[paper_id, reward_score]: Reward contribution for each paper
            Formula: reward = w × (N_success - N_failure) / (N_total + 1)
        """
        rewards = {}
        w = self.data.reward_weight

        for paper_id, record in self.data.paper_rewards.items():
            # V_i = Sim_i + reward, returning only reward part
            # Sim_i is calculated separately in embedding_scorer
            reward = w * (record.n_success - record.n_failure) / (record.n_total + 1)
            rewards[paper_id] = reward

        return rewards

    def get_paper_reward_snapshot(self, iteration: int) -> Optional[Dict[str, float]]:
        """Get reward snapshot saved at block-end iteration."""
        return self.data.paper_reward_snapshots.get(iteration)

    def save_paper_reward_snapshot(self, iteration: int) -> Dict[str, float]:
        """Save reward snapshot at block-end iteration."""
        snapshot = self.get_all_paper_rewards()
        self.data.paper_reward_snapshots[iteration] = snapshot
        self.save()
        logger.info(f"[EvolvingMemory] Paper rewards snapshot saved: iter {iteration} ({len(snapshot)} papers)")
        return snapshot

    def _get_paper_id_from_title(self, title: str) -> str:
        """Generate paper_id from title (same method as EmbeddingScorer)."""
        normalized = title.lower().strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
