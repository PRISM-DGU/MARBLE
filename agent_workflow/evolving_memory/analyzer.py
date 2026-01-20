"""IterationAnalyzerAgent for analyzing iteration results."""

from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

from agent_workflow.logger import logger
from .schemas import IterationAnalysis


# Metric improvement direction (lower is better vs higher is better)
METRIC_DIRECTION = {
    # Lower is better
    'rmse': 'lower',
    'mse': 'lower',
    'mae': 'lower',
    'loss': 'lower',
    # Higher is better
    'pcc': 'higher',
    'scc': 'higher',
    'pearson': 'higher',
    'spearman': 'higher',
    'r2': 'higher',
    'accuracy': 'higher',
    'auroc': 'higher',
    'auprc': 'higher',
    'f1': 'higher',
    'ari': 'higher',
    'nmi': 'higher',
    'silhouette': 'higher',
}

# Primary metrics by domain (used to determine improvement)
# Can be overridden by passing primary_metrics parameter
PRIMARY_METRICS = ['rmse', 'ari', 'auroc', 'accuracy']


class IterationAnalyzerAgent:
    """Agent that analyzes iteration results and extracts lessons.

    Analyzes Docker execution results to produce:
    - improved: whether performance improved
    - delta: metric changes
    - reason: explanation of results (LLM-generated)
    - lessons: key lessons learned (LLM-generated)
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize IterationAnalyzerAgent.

        Args:
            model_name: LLM model name to use
        """
        self.llm = init_chat_model(model_name, temperature=0.3)

    def analyze(
        self,
        current_iteration: int,
        current_metrics: Dict[str, float],
        current_changes: Dict[str, Any],
        prev_metrics: Optional[Dict[str, float]] = None,
        docker_output: str = "",
        primary_metrics: Optional[List[str]] = None,
    ) -> IterationAnalysis:
        """Analyze iteration results.

        Args:
            current_iteration: Current iteration number
            current_metrics: Current performance metrics
            current_changes: Changes made in current iteration
            prev_metrics: Previous iteration metrics (None for first iteration)
            docker_output: Docker execution output (last part)
            primary_metrics: Override primary metrics list (uses default if None)

        Returns:
            IterationAnalysis object
        """
        # 1. Calculate delta
        delta = self._calculate_delta(current_metrics, prev_metrics)

        # 2. Determine improvement
        improved = self._check_improvement(delta, current_metrics, prev_metrics, primary_metrics)

        # 3. Generate reason and lessons via LLM
        reason, lessons = self._generate_analysis(
            current_iteration=current_iteration,
            current_metrics=current_metrics,
            current_changes=current_changes,
            prev_metrics=prev_metrics,
            delta=delta,
            improved=improved,
            docker_output=docker_output,
        )

        analysis = IterationAnalysis(
            improved=improved,
            delta=delta,
            reason=reason,
            lessons=lessons,
        )

        logger.info(f"[IterationAnalyzer] Analysis complete: improved={improved}, lessons={len(lessons)}")
        return analysis

    def _calculate_delta(
        self,
        current: Dict[str, float],
        prev: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        """Calculate metric changes."""
        if prev is None:
            return None

        delta = {}
        for metric, curr_val in current.items():
            if metric in prev and prev[metric] is not None:
                delta[metric] = round(curr_val - prev[metric], 6)

        return delta if delta else None

    def _check_improvement(
        self,
        delta: Optional[Dict[str, float]],
        current: Dict[str, float],
        prev: Optional[Dict[str, float]],
        primary_metrics: Optional[List[str]] = None,
    ) -> Optional[bool]:
        """Determine if performance improved (based on primary metrics).

        Uses same criteria as EvolvingMemory.consecutive_failures:
        - Only uses primary metrics (rmse, ari, auroc, etc.)
        - Does not count all metrics equally
        """
        if prev is None or delta is None:
            return None  # First iteration

        # Check primary metrics in priority order
        metrics_to_check = primary_metrics or PRIMARY_METRICS
        for primary_metric in metrics_to_check:
            if primary_metric in delta and primary_metric in current:
                change = delta[primary_metric]
                direction = METRIC_DIRECTION.get(primary_metric, 'higher')

                # Check if change is significant (noise filter)
                if abs(change) < 0.0001:
                    continue  # No change, check next metric

                if direction == 'lower':
                    # Lower is better: decrease means improvement
                    return change < 0
                else:
                    # Higher is better: increase means improvement
                    return change > 0

        if primary_metrics is not None:
            return None

        # Fallback: count all metrics if no primary metrics found
        improvements = 0
        degradations = 0

        for metric, change in delta.items():
            direction = METRIC_DIRECTION.get(metric, 'higher')

            if direction == 'lower':
                if change < -0.0001:
                    improvements += 1
                elif change > 0.0001:
                    degradations += 1
            else:
                if change > 0.0001:
                    improvements += 1
                elif change < -0.0001:
                    degradations += 1

        if improvements > degradations:
            return True
        elif degradations > improvements:
            return False
        else:
            return None  # No change

    def _generate_analysis(
        self,
        current_iteration: int,
        current_metrics: Dict[str, float],
        current_changes: Dict[str, Any],
        prev_metrics: Optional[Dict[str, float]],
        delta: Optional[Dict[str, float]],
        improved: Optional[bool],
        docker_output: str,
    ) -> tuple[str, List[str]]:
        """Generate analysis reason and lessons using LLM."""

        # First iteration
        if prev_metrics is None:
            return self._generate_baseline_analysis(
                current_metrics, current_changes, docker_output
            )

        # Subsequent iterations
        return self._generate_comparison_analysis(
            current_iteration=current_iteration,
            current_metrics=current_metrics,
            current_changes=current_changes,
            prev_metrics=prev_metrics,
            delta=delta,
            improved=improved,
            docker_output=docker_output,
        )

    def _generate_baseline_analysis(
        self,
        metrics: Dict[str, float],
        changes: Dict[str, Any],
        docker_output: str,
    ) -> tuple[str, List[str]]:
        """Analyze first iteration (baseline) results."""
        system_prompt = """You are an ML experiment analysis expert.
Analyze the first iteration's baseline results.

CRITICAL: Lessons must be SPECIFIC and ACTIONABLE for the next iteration.

Good lesson examples:
- "hidden_dim=128 may be insufficient for 17737-dim gene expression; consider 256-512"
- "dropout=0.1 with small batch_size=32 may cause underfitting; try dropout=0.05"
- "3-layer MLP [4096,2048,512] loses information; add skip connections or reduce layers"

Bad lesson examples (too vague - DO NOT write like this):
- "Model performance is okay"
- "Need to improve encoder"
- "Training is stable"

Response format (JSON):
{
    "reason": "Brief explanation of baseline performance (1-2 sentences)",
    "lessons": ["specific lesson 1", "specific lesson 2"]
}

Each lesson MUST include:
- Specific component/parameter name
- Current value or state
- Suggested direction or change"""

        user_prompt = f"""## Iteration 1 (Baseline)

### Changes
- Component: {changes.get('component', 'unknown')}
- Description: {changes.get('description', '')}

### Performance Metrics
{self._format_metrics(metrics)}

### Docker Output (last part)
```
{docker_output[-1500:] if docker_output else '(none)'}
```

Analyze this baseline result and extract actionable lessons for future iterations."""

        return self._call_llm(system_prompt, user_prompt)

    def _generate_comparison_analysis(
        self,
        current_iteration: int,
        current_metrics: Dict[str, float],
        current_changes: Dict[str, Any],
        prev_metrics: Dict[str, float],
        delta: Optional[Dict[str, float]],
        improved: Optional[bool],
        docker_output: str,
    ) -> tuple[str, List[str]]:
        """Compare with previous iteration and analyze."""
        system_prompt = """You are an ML experiment analysis expert.
Analyze performance changes compared to the previous iteration.

CRITICAL: Lessons must be SPECIFIC and DIRECTLY ACTIONABLE.

Lesson format requirements:
1. ARCHITECTURE lessons: "[component].[parameter] = [value] caused [effect]; recommend [action]"
   Example: "cell_encoder.hidden_layers=[4096,2048,512] improved PCC but bottleneck at 512; try [4096,2048,1024,512]"

2. HYPERPARAMETER lessons: "[param]=[value] resulted in [metric change]; next try [suggestion]"
   Example: "learning_rate=0.001 with weight_decay=0.00001 caused overfitting after epoch 50; reduce lr to 0.0005"

3. TRAINING lessons: "Training pattern [observation]; adjust [parameter] to [value]"
   Example: "Loss plateaued at epoch 30; increase epochs to 150 or add learning rate scheduler"

Response format (JSON):
{
    "reason": "Specific cause of improvement/degradation (1-2 sentences, include numbers)",
    "lessons": ["actionable lesson 1", "actionable lesson 2"]
}

IMPORTANT: Each lesson should be copy-paste ready for the next iteration's config or code change."""

        result_emoji = "IMPROVED" if improved else ("DEGRADED" if improved is False else "NO_CHANGE")
        user_prompt = f"""## Iteration {current_iteration} Analysis

### Result: {result_emoji}

### Changes Made
- Component: {current_changes.get('component', 'unknown')}
- Description: {current_changes.get('description', '')}

### Previous → Current Metrics
{self._format_comparison(prev_metrics, current_metrics, delta)}

### Docker Output (last part)
```
{docker_output[-1000:] if docker_output else '(none)'}
```

Explain why this result occurred and provide actionable lessons for the next iteration."""

        return self._call_llm(system_prompt, user_prompt)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, List[str]]:
        """Call LLM to generate analysis results."""
        import json

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            # Parse JSON
            content = response.content.strip()
            # Handle ```json ... ``` format
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            result = json.loads(content)
            reason = result.get("reason", "No analysis result")
            lessons = result.get("lessons", [])

            # Convert string to list if needed
            if isinstance(lessons, str):
                lessons = [lessons]

            return reason, lessons

        except Exception as e:
            logger.warning(f"[IterationAnalyzer] LLM analysis failed: {e}")
            return "Error during analysis", ["LLM analysis failed; unable to extract lessons"]

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display."""
        lines = []
        for k, v in metrics.items():
            lines.append(f"- {k.upper()}: {v:.4f}")
        return "\n".join(lines) if lines else "(no metrics)"

    def _format_comparison(
        self,
        prev: Dict[str, float],
        curr: Dict[str, float],
        delta: Optional[Dict[str, float]],
    ) -> str:
        """Format previous/current metrics comparison."""
        lines = []
        all_keys = set(prev.keys()) | set(curr.keys())

        for k in sorted(all_keys):
            prev_val = prev.get(k)
            curr_val = curr.get(k)
            delta_val = delta.get(k) if delta else None

            if prev_val is not None and curr_val is not None:
                sign = "+" if delta_val and delta_val > 0 else ""
                delta_str = f" ({sign}{delta_val:.4f})" if delta_val else ""
                lines.append(f"- {k.upper()}: {prev_val:.4f} → {curr_val:.4f}{delta_str}")
            elif curr_val is not None:
                lines.append(f"- {k.upper()}: (none) → {curr_val:.4f}")

        return "\n".join(lines) if lines else "(no metrics)"


def analyze_iteration(
    current_iteration: int,
    current_metrics: Dict[str, float],
    current_changes: Dict[str, Any],
    prev_metrics: Optional[Dict[str, float]] = None,
    docker_output: str = "",
    primary_metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convenience function: analyze iteration and return dict.

    Args:
        current_iteration: Current iteration number
        current_metrics: Current performance metrics
        current_changes: Changes made (component, description)
        prev_metrics: Previous performance metrics
        docker_output: Docker output
        primary_metrics: Override primary metrics list (uses default if None)

    Returns:
        Analysis result dict (improved, delta, reason, lessons)
    """
    analyzer = IterationAnalyzerAgent()
    analysis = analyzer.analyze(
        current_iteration=current_iteration,
        current_metrics=current_metrics,
        current_changes=current_changes,
        prev_metrics=prev_metrics,
        docker_output=docker_output,
        primary_metrics=primary_metrics,
    )
    return analysis.model_dump()
