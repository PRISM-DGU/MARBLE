"""Paper Aggregator Agent for build_debate workflow.

NEW Workflow (Local PDF-based scoring with EmbeddingScorer):
1. score_papers_with_embedding() - Score papers from local PDF directory using EmbeddingScorer
2. generate_top20_summary() - Generate summary for Top-20
3. LLM selects final 5 papers from Top-20 summary
4. clone_selected_repos() - Git clone for selected 5 papers

Scoring formula (from embedding_scorer.py):
S_total = w_d × S_domain + w_a × [β × S_arch + (1 - β) × novelty]

Where S_arch uses momentum (score-level, not embedding-level):
S_arch(t) = 0.9 × Sim(method, E_code_t) + 0.1 × S_arch(t-1)
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from agent_workflow.utils import get_project_root
from .common_tools import (
    create_read_file_tool,
    create_write_file_tool,
    create_list_directory_tool,
)
from .embedding_scorer import EmbeddingScorer


# Number of papers to select for debate (expanded from 2 to 5)
NUM_PAPERS_TO_SELECT = 5


class PaperAggregatorAgent(BaseAgentNode):
    """Agent that scores and selects papers using EmbeddingScorer.

    NEW Workflow (Local PDF-based scoring):
    1. score_papers_with_embedding() - Score papers from local PDF directory using EmbeddingScorer
    2. generate_top20_summary() - Generate summary for Top-20
    3. LLM reviews summary and selects final 5 papers (expanded from 2)
    4. clone_selected_repos() - Git clone for selected 5 papers

    Key features:
    - Uses BAAI/bge-m3 embeddings via EmbeddingScorer
    - Momentum update: S_arch(t) = 0.9 × Sim(method, E_code_t) + 0.1 × S_arch(t-1)
    - Dynamic weight scheduling by iteration
    - Beta (novelty coefficient) can be adjusted by agent
    """

    def __init__(
        self,
        target_model: str = "",
        current_iteration: int = 1,
        total_iterations: int = 1,
        build_dir: str = "",
        output_path: str = "",
        repos_dir: str = "",
        beta: float = 1.0,
        model_config: Optional[Dict[str, Any]] = None,
        used_papers: Optional[List[str]] = None,
        checkpointer=None,
        reward_patience: int = 10,
        reward_weight: float = 0.1,
    ):
        """Initialize Paper Aggregator Agent with EmbeddingScorer.

        Args:
            target_model: Model name (e.g., 'stagate', 'deeptta')
            current_iteration: Current iteration number (1-based)
            total_iterations: Total number of iterations (for w_d/w_a scheduling)
            build_dir: Path to current build_N directory
            output_path: Output file path for aggregated results
            repos_dir: Directory to clone GitHub repos
            beta: Novelty coefficient (0.0 to 1.0, default 1.0)
            model_config: Model configuration dict
            used_papers: List of previously used paper titles to exclude
            checkpointer: Optional LangGraph checkpointer
            reward_patience: Reward block size (--patience flag, default 10)
            reward_weight: Reward weight (--weight flag, default 0.1)
        """
        super().__init__("paper_aggregator", checkpointer=checkpointer)

        self.target_model = target_model.lower() if target_model else "deeptta"
        self.current_iteration = current_iteration
        self.total_iterations = max(1, total_iterations)
        self.build_dir = Path(build_dir) if build_dir else Path(get_project_root()) / "experiments" / f"build_{current_iteration}"
        self.output_path = output_path
        self.repos_dir = repos_dir
        self.model_config = model_config
        self.used_papers = list(used_papers) if used_papers else []

        # Beta is fixed at 1.0 (no longer adjustable)
        self.beta = 1.0

        # Reward settings (from CLI flags)
        self.reward_patience = reward_patience
        self.reward_weight = reward_weight

        # EmbeddingScorer will be initialized lazily
        self._scorer = None
        self._scored_papers = []
        self._stratified_results = {}  # For stratified scoring

        # Load memory data for prompt
        self._memory_data = self._load_memory_data()

        logger.info(f"[PaperAggregatorAgent] Initialized for {self.target_model} iter {current_iteration}/{total_iterations}")
        logger.info(f"[PaperAggregatorAgent] Build dir: {self.build_dir}")
        logger.info(f"[PaperAggregatorAgent] Beta: {self.beta} (fixed)")

    def _get_scorer(self) -> EmbeddingScorer:
        """Lazy initialization of EmbeddingScorer."""
        if self._scorer is None:
            self._scorer = EmbeddingScorer(
                target_model=self.target_model,
                current_iteration=self.current_iteration,
                total_iterations=self.total_iterations,
                build_dir=self.build_dir,
                beta=self.beta,
                used_papers=self.used_papers,
                reward_patience=self.reward_patience,
                reward_weight=self.reward_weight,
            )
        return self._scorer

    def get_prompt(self):
        """Return system prompt for paper aggregation with stratified selection."""
        model_name = self.model_config.get("model_name", "Unknown") if self.model_config else "Unknown"
        perf_history = self._get_performance_history_summary()

        return f"""You are a Paper Aggregator Agent that selects the best papers for improving {model_name}.

## Your Task
Select 5 papers using STRATIFIED scoring with 3 different weight configurations.

## Performance History (from memory.json)
{perf_history}

## Stratified Scoring System (Iteration {self.current_iteration}/{self.total_iterations})

Papers are scored with 3 different weight configurations:
1. **high_domain** (w_d=0.9, w_a=0.1): Prioritizes domain relevance → Select 2 papers
2. **balanced** (w_d=0.5, w_a=0.5): Equal weight → Select 1 paper
3. **high_arch** (w_d=0.1, w_a=0.9): Prioritizes architecture similarity → Select 2 papers

This ensures diversity: domain-focused, balanced, and architecture-focused papers.

## WORKFLOW (4 Steps - Follow in ORDER!)

### Step 1: Score Papers with Stratified Weights
Call: `score_papers_stratified()`

Scores all papers with 3 weight configurations, generating 3 separate Top-20 lists.
- Uses BAAI/bge-m3 embeddings
- Formula: S_total = w_d × S_domain + w_a × S_arch
- Momentum (iter 2+): S_arch = 0.9 × current + 0.1 × previous

### Step 2: Generate Stratified Summaries
Call: `generate_stratified_summary()`

Creates 3 markdown files for review:
- `top20_high_domain.md` (w_d=0.9, w_a=0.1)
- `top20_balanced.md` (w_d=0.5, w_a=0.5)
- `top20_high_arch.md` (w_d=0.1, w_a=0.9)

### Step 3: Select Papers from Each Category (YOU DECIDE!)
Read all 3 summary files using `read_file`, then call:

```
select_stratified_papers(
    high_domain_indices=[idx1, idx2],     # Select 2 from high_domain Top-20
    balanced_indices=[idx1],               # Select 1 from balanced Top-20
    high_arch_indices=[idx1, idx2],        # Select 2 from high_arch Top-20
    reasons=["reason1", "reason2", "reason3", "reason4", "reason5"]
)
```

**Selection Criteria:**
1. Higher score within each category = better candidate
2. GitHub code MUST be available
3. Method should be transferable to {model_name}'s architecture
4. NOTE: If same paper appears in multiple Top-20s and is selected multiple times,
   duplicates are automatically replaced with next-best paper from that category.

### Step 4: Clone Selected Repositories
Call: `clone_selected_repos()`

Clones GitHub repositories for the 5 selected papers.
If clone fails, automatically tries next ranked paper.

## CRITICAL RULES
1. Call tools in order: Step 1 → Step 2 → Step 3 → Step 4
2. Read ALL 3 summary files before Step 3
3. Select exactly: 2 from high_domain + 1 from balanced + 2 from high_arch = 5 papers
4. Indices are 0-indexed (0-19 for each Top-20)
"""

    def get_additional_tools(self):
        """Return tools for paper aggregation (Stratified scoring workflow)."""
        return [
            create_read_file_tool(max_length=20000),
            create_write_file_tool(),
            create_list_directory_tool(),
            # Stratified Workflow Tools (4-step)
            self._create_score_stratified_tool(),           # Step 1: Score with 3 weight configs
            self._create_generate_stratified_summary_tool(),# Step 2: Generate 3 Top-20 summaries
            self._create_select_stratified_papers_tool(),   # Step 3: LLM selects 2+1+2=5
            self._create_clone_selected_repos_tool(),       # Step 4: Git clone selected 5
        ]

    # ==========================================================================
    # STRATIFIED WORKFLOW TOOLS
    # ==========================================================================

    def _create_score_stratified_tool(self):
        """Create tool to score papers with 3 weight configurations."""

        agent = self

        @tool
        def score_papers_stratified() -> Dict[str, Any]:
            """Score papers with 3 different weight configurations (stratified scoring).

            This is Step 1 of the stratified workflow.

            Weight configurations:
            - high_domain (w_d=0.9, w_a=0.1): Domain relevance focused → Top-20
            - balanced (w_d=0.5, w_a=0.5): Equal weight → Top-20
            - high_arch (w_d=0.1, w_a=0.9): Architecture similarity focused → Top-20

            Returns:
                Dict with 3 Top-20 lists and scoring metadata
            """
            try:
                scorer = agent._get_scorer()

                # Score with stratified weights
                stratified_results, metadata = scorer.score_all_papers_stratified()

                if not stratified_results:
                    return {"success": False, "error": "No papers could be scored"}

                # Store for later use
                agent._stratified_results = stratified_results

                # Save results
                output_dir = agent.build_dir / "embeddings"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "stratified_scoring.json"
                scorer.save_stratified_results(stratified_results, metadata, output_path)

                # Store output path for other tools
                agent._stratified_output_path = str(output_path)

                logger.info(f"[STRATIFIED_SCORE] ========== Scoring Complete ==========")
                for label, data in stratified_results.items():
                    logger.info(f"[STRATIFIED_SCORE] {label}: {data['total_scored']} scored, Top-20 selected")

                return {
                    "success": True,
                    "categories": {
                        label: {
                            "weights": data["weights"],
                            "select_count": data["select_count"],
                            "top_3_preview": [
                                {
                                    "rank": i + 1,
                                    "title": p.get("title", "")[:60],
                                    "score": p.get("scores", {}).get("final", 0),
                                    "github": p.get("github_urls", ["N/A"])[0] if p.get("github_urls") else "N/A",
                                }
                                for i, p in enumerate(data["top_20"][:3])
                            ]
                        }
                        for label, data in stratified_results.items()
                    },
                    "message": "Stratified scoring complete. Now call generate_stratified_summary() to review papers."
                }
            except Exception as e:
                logger.error(f"[STRATIFIED_SCORE] Error: {e}")
                return {"success": False, "error": str(e)}

        return score_papers_stratified

    def _create_generate_stratified_summary_tool(self):
        """Create tool to generate 3 Top-20 summary markdown files."""

        agent = self

        @tool
        def generate_stratified_summary() -> Dict[str, Any]:
            """Generate markdown summaries for each of the 3 Top-20 lists.

            This is Step 2 of the stratified workflow.
            Creates 3 files: top20_high_domain.md, top20_balanced.md, top20_high_arch.md

            Returns:
                Dict with paths to the 3 summary files
            """
            stratified_path = getattr(agent, '_stratified_output_path', None)
            if not stratified_path or not os.path.exists(stratified_path):
                return {"success": False, "error": "Run score_papers_stratified first"}

            with open(stratified_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            stratified_results = data.get("stratified_results", {})
            if not stratified_results:
                return {"success": False, "error": "No stratified results found"}

            output_dir = os.path.dirname(stratified_path)
            summary_paths = {}

            category_info = {
                "high_domain": {"title": "High Domain Relevance", "desc": "w_d=0.9, w_a=0.1", "select": 2},
                "balanced": {"title": "Balanced", "desc": "w_d=0.5, w_a=0.5", "select": 1},
                "high_arch": {"title": "High Architecture Similarity", "desc": "w_d=0.1, w_a=0.9", "select": 2},
            }

            for label, cat_data in stratified_results.items():
                top_20 = cat_data.get("top_20", [])
                weights = cat_data.get("weights", {})
                select_count = cat_data.get("select_count", 0)
                info = category_info.get(label, {"title": label, "desc": "", "select": 0})

                md_lines = [
                    f"# Top-20 Papers: {info['title']}",
                    f"**Weights**: {info['desc']} | **Select**: {select_count} paper(s)",
                    "",
                    "---",
                    ""
                ]

                for i, paper in enumerate(top_20, 1):
                    title = paper.get("title", "Unknown")
                    abstract = paper.get("abstract", "")[:400]
                    scores = paper.get("scores", {})
                    github_urls = paper.get("github_urls", [])
                    year = paper.get("year", "N/A")

                    # Extract method summary
                    method_summary = ""
                    if abstract:
                        sentences = re.split(r'(?<=[.!?])\s+', abstract)
                        method_keywords = ['propose', 'introduce', 'present', 'develop', 'method', 'approach', 'model']
                        for sent in sentences[:5]:
                            if any(kw in sent.lower() for kw in method_keywords):
                                method_summary = sent.strip()
                                break
                        if not method_summary and sentences:
                            method_summary = sentences[0].strip()

                    md_lines.extend([
                        f"## #{i}. {title}",
                        f"- **Year**: {year} | **Score**: {scores.get('final', 0):.3f} (domain: {scores.get('domain', 0):.2f}, arch: {scores.get('architecture', 0):.2f})",
                        f"- **GitHub**: {github_urls[0] if github_urls else 'N/A'}",
                        f"- **Method**: {method_summary[:180]}..." if len(method_summary) > 180 else f"- **Method**: {method_summary}",
                        ""
                    ])

                md_lines.extend([
                    "---",
                    f"**Select {select_count} paper(s) from this list (indices 0-19)**",
                    ""
                ])

                summary_path = os.path.join(output_dir, f"top20_{label}.md")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(md_lines))

                summary_paths[label] = summary_path
                logger.info(f"[STRATIFIED_SUMMARY] Generated: {summary_path}")

            return {
                "success": True,
                "summary_files": summary_paths,
                "message": "Read all 3 files, then call select_stratified_papers() with indices from each category."
            }

        return generate_stratified_summary

    def _create_select_stratified_papers_tool(self):
        """Create tool to select papers from each stratified category."""

        agent = self

        @tool
        def select_stratified_papers(
            high_domain_indices: List[int],
            balanced_indices: List[int],
            high_arch_indices: List[int],
            reasons: List[str]
        ) -> Dict[str, Any]:
            """Select papers from each stratified category (2 + 1 + 2 = 5 total).

            Args:
                high_domain_indices: 2 indices from high_domain Top-20 (0-19)
                balanced_indices: 1 index from balanced Top-20 (0-19)
                high_arch_indices: 2 indices from high_arch Top-20 (0-19)
                reasons: 5 reasons for each selection

            Returns:
                Dict with 5 selected papers (duplicates auto-replaced)
            """
            stratified_path = getattr(agent, '_stratified_output_path', None)
            if not stratified_path or not os.path.exists(stratified_path):
                return {"success": False, "error": "Run score_papers_stratified first"}

            with open(stratified_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            stratified_results = data.get("stratified_results", {})

            # Validate indices count
            if len(high_domain_indices) != 2:
                return {"success": False, "error": "high_domain_indices must have exactly 2 indices"}
            if len(balanced_indices) != 1:
                return {"success": False, "error": "balanced_indices must have exactly 1 index"}
            if len(high_arch_indices) != 2:
                return {"success": False, "error": "high_arch_indices must have exactly 2 indices"}

            selection_config = [
                ("high_domain", high_domain_indices),
                ("balanced", balanced_indices),
                ("high_arch", high_arch_indices),
            ]

            selected_papers = []
            selected_titles = set()
            reason_idx = 0

            for category, indices in selection_config:
                cat_data = stratified_results.get(category, {})
                top_20 = cat_data.get("top_20", [])

                for idx in indices:
                    if not (0 <= idx < len(top_20)):
                        return {"success": False, "error": f"Invalid index {idx} for {category} (must be 0-{len(top_20)-1})"}

                    paper = top_20[idx].copy()
                    title = paper.get("title", "")

                    # Check for duplicates
                    if title in selected_titles:
                        # Find next available paper from this category
                        found_replacement = False
                        for fallback_idx, fallback_paper in enumerate(top_20):
                            fallback_title = fallback_paper.get("title", "")
                            if fallback_title not in selected_titles:
                                paper = fallback_paper.copy()
                                title = fallback_title
                                logger.info(f"[STRATIFIED_SELECT] Duplicate replaced: idx {idx} → {fallback_idx} in {category}")
                                found_replacement = True
                                break

                        if not found_replacement:
                            logger.warning(f"[STRATIFIED_SELECT] No replacement found for duplicate in {category}")
                            continue

                    paper["selection_category"] = category
                    paper["selection_reason"] = reasons[reason_idx] if reason_idx < len(reasons) else "N/A"
                    paper["selection_rank"] = len(selected_papers) + 1

                    selected_papers.append(paper)
                    selected_titles.add(title)
                    reason_idx += 1

            # Save to stratified_scoring.json
            data["final_selected"] = selected_papers
            data["selection_timestamp"] = datetime.now().isoformat()

            with open(stratified_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Also update agent's internal state
            agent._final_selected = selected_papers

            logger.info(f"[STRATIFIED_SELECT] Selected {len(selected_papers)} papers (2+1+2)")
            for p in selected_papers:
                logger.info(f"[STRATIFIED_SELECT] ✅ [{p['selection_category']}] {p['title'][:50]}...")

            return {
                "success": True,
                "selected_count": len(selected_papers),
                "selected_papers": [
                    {
                        "category": p.get("selection_category"),
                        "title": p.get("title"),
                        "score": p.get("scores", {}).get("final"),
                        "reason": p.get("selection_reason"),
                        "github": p.get("github_urls", ["N/A"])[0] if p.get("github_urls") else "N/A"
                    }
                    for p in selected_papers
                ],
                "message": f"Selected {len(selected_papers)} papers. Now call clone_selected_repos() to clone repositories."
            }

        return select_stratified_papers

    # ==========================================================================
    # LEGACY TOOLS (kept for backward compatibility, not used in stratified workflow)
    # ==========================================================================

    def _create_select_final_papers_tool(self):
        """[LEGACY] Create tool to select final 5 papers from single Top-20."""

        output_path = self.output_path
        agent = self

        @tool
        def select_final_papers(
            paper_indices: List[int],
            reasons: List[str]
        ) -> Dict[str, Any]:
            """[LEGACY] Select final 5 papers from Top-20 for the debate workflow.

            NOTE: This is a legacy tool. Use select_stratified_papers() instead.

            Args:
                paper_indices: List of indices (0-19) of selected papers from Top-20 (select 5)
                reasons: List of reasons for each selection

            Returns:
                Dict with selected papers
            """
            if not output_path or not os.path.exists(output_path):
                return {"success": False, "error": "Scored results not found. Run score_papers_with_embedding first."}

            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                top_papers = data.get("top_20", [])

                selected = []
                for i, idx in enumerate(paper_indices[:NUM_PAPERS_TO_SELECT]):
                    if 0 <= idx < len(top_papers):
                        paper = top_papers[idx].copy()
                        paper["selection_reason"] = reasons[i] if i < len(reasons) else "N/A"
                        paper["selection_rank"] = i + 1
                        selected.append(paper)

                if not selected:
                    return {"success": False, "error": f"Invalid indices. Must be 0-{len(top_papers)-1}"}

                # Update file with selections
                data["final_selected"] = selected
                data["selection_timestamp"] = datetime.now().isoformat()

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # Also update agent's internal state
                agent._final_selected = selected

                logger.info(f"[AGGREGATOR] Selected {len(selected)} final papers from Top-{len(top_papers)}")
                for p in selected:
                    logger.info(f"[AGGREGATOR] ✅ {p['title'][:50]}...")

                return {
                    "success": True,
                    "selected_count": len(selected),
                    "selected_papers": [
                        {
                            "title": p["title"],
                            "source": p.get("source"),
                            "scores": p.get("scores"),
                            "reason": p.get("selection_reason"),
                            "github": p.get("github_urls", ["N/A"])[0] if p.get("github_urls") else "N/A"
                        }
                        for p in selected
                    ],
                    "message": f"Selected {len(selected)} papers. Now call clone_selected_repos() to clone GitHub repositories."
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        return select_final_papers

    # ==========================================================================
    # LEGACY TOOLS (kept for reference)
    # ==========================================================================

    def _create_score_with_embedding_tool(self):
        """Create tool to score papers using EmbeddingScorer."""

        output_path = self.output_path
        agent = self

        @tool
        def score_papers_with_embedding() -> Dict[str, Any]:
            """Score papers from local PDF directory using EmbeddingScorer.

            This is Step 1 of the new workflow.

            Scoring formula:
            S_total = w_d × S_domain + w_a × [β × S_arch + (1 - β) × novelty]

            Where S_arch uses momentum (score-level):
            S_arch(t) = 0.9 × Sim(method, E_code_t) + 0.1 × S_arch(t-1)

            Returns:
                Dict with Top-20 papers and their scores
            """
            try:
                scorer = agent._get_scorer()

                # Score all papers
                scored_papers, metadata = scorer.score_all_papers()

                if not scored_papers:
                    return {"success": False, "error": "No papers could be scored"}

                # Get Top-20
                top_20 = scored_papers[:20]

                # Store for later use
                agent._scored_papers = scored_papers
                agent._top_20 = top_20

                # Save results
                if output_path:
                    scorer.save_results(scored_papers, metadata, Path(output_path))

                logger.info(f"[EMBEDDING_SCORE] ========== Scoring Complete ==========")
                logger.info(f"[EMBEDDING_SCORE] Scored {len(scored_papers)} papers, Top-20 selected")
                for i, p in enumerate(top_20[:5], 1):
                    scores = p.get("scores", {})
                    logger.info(f"[EMBEDDING_SCORE] #{i} ({scores.get('final', 0):.3f}): {p.get('title', '')[:50]}...")

                return {
                    "success": True,
                    "total_scored": len(scored_papers),
                    "top_20_count": len(top_20),
                    "scoring_params": {
                        "w_d": scorer.w_d,
                        "w_a": scorer.w_a,
                        "beta": scorer.beta,
                        "momentum": "0.9*current + 0.1*prev" if agent.current_iteration > 1 else "disabled",
                    },
                    "top_5_preview": [
                        {
                            "rank": i + 1,
                            "title": p.get("title", "")[:60],
                            "score": p.get("scores", {}).get("final", 0),
                            "github": p.get("github_urls", ["N/A"])[0] if p.get("github_urls") else "N/A",
                        }
                        for i, p in enumerate(top_20[:5])
                    ],
                    "message": "Scoring complete. Now call generate_top20_summary() to review papers."
                }
            except Exception as e:
                logger.error(f"[EMBEDDING_SCORE] Error: {e}")
                return {"success": False, "error": str(e)}

        return score_papers_with_embedding

    def _create_generate_top20_summary_tool(self):
        """Create tool to generate Top-20 summary markdown."""

        output_path = self.output_path

        @tool
        def generate_top20_summary() -> Dict[str, Any]:
            """Generate a markdown summary of Top-20 papers for LLM selection.

            This is Step 3 of the new workflow.
            Creates a concise summary with 1-line method description for each paper.

            Returns:
                Dict with summary file path
            """
            # Load scored results
            if not output_path or not os.path.exists(output_path):
                return {"success": False, "error": "Run score_papers_with_pdf_content first"}

            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            top_20 = data.get("top_20", [])
            if not top_20:
                return {"success": False, "error": "No Top-20 papers found"}

            # Get scoring params for guidance
            scoring_params = data.get("metadata", {}).get("weights", {})
            w_d = scoring_params.get("w_d", 0.7)
            w_a = scoring_params.get("w_a", 0.3)
            beta = scoring_params.get("beta", 1.0)

            # Generate markdown summary
            md_lines = [
                "# Top-20 Paper Candidates for Model Improvement",
                "",
                f"Review each paper and select the **best {NUM_PAPERS_TO_SELECT}** for implementation.",
                "",
                "## Scoring Formula",
                "```",
                "S_total = w_d × S_domain + w_a × [β × S_arch + (1-β) × novelty]",
                "```",
                "",
                f"**Current Parameters:**",
                f"- w_d (domain weight): {w_d}",
                f"- w_a (architecture weight): {w_a}",
                f"- β (novelty coefficient): {beta}",
                "",
                "**Higher total score = better candidate**",
                "",
                "## Selection Criteria",
                f"1. Prioritize papers with higher weighted scores",
                "2. GitHub code MUST be available and implementable",
                "3. Select diverse approaches that address different weaknesses",
                "4. Architecture should be transferable to the target domain",
                "",
                "---",
                ""
            ]

            for i, paper in enumerate(top_20, 1):
                title = paper.get("title", "Unknown")
                abstract = paper.get("abstract", "")[:500]
                source = paper.get("source", "unknown")
                scores = paper.get("scores", {})
                github_urls = paper.get("github_urls", [])
                year = paper.get("year", "N/A")
                venue = paper.get("venue", paper.get("venue_status", "N/A"))

                # Extract key method from abstract (first 1-2 sentences focusing on method)
                method_summary = ""
                if abstract:
                    sentences = re.split(r'(?<=[.!?])\s+', abstract)
                    method_keywords = ['propose', 'introduce', 'present', 'develop', 'method', 'approach', 'model', 'network', 'architecture']
                    for sent in sentences[:5]:
                        if any(kw in sent.lower() for kw in method_keywords):
                            method_summary = sent.strip()
                            break
                    if not method_summary and sentences:
                        method_summary = sentences[0].strip()

                # Format scores with new naming
                score_final = scores.get("final", 0)
                score_domain = scores.get("domain", scores.get("domain_raw", 0))
                score_arch = scores.get("architecture", scores.get("architecture_current", 0))
                score_novelty = scores.get("novelty", 0)

                md_lines.extend([
                    f"## #{i}. {title}",
                    f"- **Source**: {source.upper()} | **Year**: {year} | **Venue**: {venue}",
                    f"- **Score**: {score_final:.3f} (domain: {score_domain:.2f}, arch: {score_arch:.2f}, novelty: {score_novelty:.2f})",
                    f"- **GitHub**: {github_urls[0] if github_urls else 'N/A'}",
                    f"- **Method**: {method_summary[:200]}..." if len(method_summary) > 200 else f"- **Method**: {method_summary}",
                    ""
                ])

            md_lines.extend([
                "---",
                "",
                "## Your Task",
                f"Select exactly **{NUM_PAPERS_TO_SELECT} papers** that best address the model's weaknesses.",
                "For each selection, provide:",
                "1. Paper number (0-19, 0-indexed)",
                "2. Why this paper's method is suitable",
                "3. Which component it can improve (drug_encoder / cell_encoder / decoder / etc.)",
                "",
                "NOTE: Select diverse approaches that address different weaknesses.",
                ""
            ])

            # Save markdown
            summary_path = os.path.join(os.path.dirname(output_path), "top20_summary.md")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md_lines))

            logger.info(f"[TOP20_SUMMARY] Generated summary at: {summary_path}")

            return {
                "success": True,
                "summary_path": summary_path,
                "paper_count": len(top_20),
                "message": f"Top-20 summary saved. Review the file and select {NUM_PAPERS_TO_SELECT} papers using select_final_papers."
            }

        return generate_top20_summary

    def _create_clone_selected_repos_tool(self):
        """Create tool to clone GitHub repos for selected papers only."""

        repos_dir = self.repos_dir
        agent = self

        def _try_clone_paper(paper: Dict, repos_dir: str) -> Dict:
            """Try to clone a single paper's repository."""
            github_urls = paper.get("github_urls", [])
            title = paper.get("title", "Unknown")[:50]

            if not github_urls:
                return {"title": title, "status": "no_github_url", "paper": None}

            github_url = github_urls[0]
            # Normalize URL
            if not github_url.startswith('http'):
                github_url = f'https://{github_url}'
            github_url = github_url.rstrip('/').rstrip('.git')

            repo_name = github_url.split('/')[-1]
            clone_path = os.path.join(repos_dir, repo_name)

            # Check if already cloned
            if os.path.exists(clone_path) and os.path.isdir(os.path.join(clone_path, ".git")):
                paper["cloned_repo_path"] = clone_path
                return {"title": title, "status": "already_exists", "path": clone_path, "paper": paper}

            # Clone
            try:
                env = os.environ.copy()
                env["GIT_TERMINAL_PROMPT"] = "0"
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", f"{github_url}.git", clone_path],
                    capture_output=True, text=True, timeout=120, env=env
                )
                if result.returncode == 0:
                    paper["cloned_repo_path"] = clone_path
                    logger.info(f"[CLONE] ✅ {title}: {clone_path}")
                    return {"title": title, "status": "success", "path": clone_path, "paper": paper}
                else:
                    logger.warning(f"[CLONE] ❌ {title}: {result.stderr[:100]}")
                    return {"title": title, "status": "failed", "error": result.stderr[:100], "paper": None}
            except Exception as e:
                logger.error(f"[CLONE] Error for {title}: {e}")
                return {"title": title, "status": "error", "error": str(e), "paper": None}

        @tool
        def clone_selected_repos() -> Dict[str, Any]:
            """Clone GitHub repositories for the selected papers (up to 5).

            This is Step 4 of the stratified workflow.
            If clone fails, automatically tries next ranked paper from the same category.

            Returns:
                Dict with clone results
            """
            # Try stratified output first, then legacy output
            stratified_path = getattr(agent, '_stratified_output_path', None)
            output_path = agent.output_path

            data_path = None
            if stratified_path and os.path.exists(stratified_path):
                data_path = stratified_path
            elif output_path and os.path.exists(output_path):
                data_path = output_path

            if not data_path:
                return {"success": False, "error": "No selection found. Run select_stratified_papers first."}

            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            selected = data.get("final_selected", [])
            if not selected:
                return {"success": False, "error": "No papers selected. Run select_stratified_papers first."}

            # Get fallback candidates from stratified results or legacy top_20
            stratified_results = data.get("stratified_results", {})
            all_fallback = []
            if stratified_results:
                # Stratified mode: combine all Top-20s
                for cat_data in stratified_results.values():
                    all_fallback.extend(cat_data.get("top_20", []))
            else:
                # Legacy mode: use single top_20
                all_fallback = data.get("top_20", [])

            selected_titles = {p.get("title") for p in selected}

            # Build fallback queue (excluding already selected)
            fallback_queue = [p for p in all_fallback if p.get("title") not in selected_titles]

            os.makedirs(repos_dir, exist_ok=True)

            clone_results = []
            successfully_cloned = []
            target_count = len(selected)  # Usually 5

            # First, try to clone selected papers
            for paper in selected:
                result = _try_clone_paper(paper, repos_dir)
                clone_results.append(result)
                if result["status"] in ["success", "already_exists"]:
                    successfully_cloned.append(result["paper"])

            # If some clones failed, try fallback papers
            fallback_idx = 0
            while len(successfully_cloned) < target_count and fallback_idx < len(fallback_queue):
                fallback_paper = fallback_queue[fallback_idx]
                fallback_idx += 1

                logger.info(f"[CLONE] Trying fallback paper #{fallback_idx}: {fallback_paper.get('title', '')[:40]}...")
                result = _try_clone_paper(fallback_paper, repos_dir)
                result["is_fallback"] = True
                clone_results.append(result)

                if result["status"] in ["success", "already_exists"]:
                    successfully_cloned.append(result["paper"])
                    logger.info(f"[CLONE] ✅ Fallback success: {fallback_paper.get('title', '')[:40]}")

            # Update final_selected with successfully cloned papers
            data["final_selected"] = successfully_cloned
            data["clone_results"] = clone_results
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            success_count = len(successfully_cloned)
            fallback_count = sum(1 for r in clone_results if r.get("is_fallback") and r["status"] in ["success", "already_exists"])

            logger.info(f"[CLONE] ========== Clone Complete ==========")
            logger.info(f"[CLONE] Successfully cloned: {success_count}/{target_count}")
            if fallback_count > 0:
                logger.info(f"[CLONE] Used {fallback_count} fallback paper(s) from Top-20")

            return {
                "success": True,
                "target": target_count,
                "cloned": success_count,
                "fallback_used": fallback_count,
                "results": clone_results,
                "final_papers": [
                    {"title": p.get("title", "")[:60], "path": p.get("cloned_repo_path", "")}
                    for p in successfully_cloned
                ]
            }

        return clone_selected_repos

    def _load_memory_data(self) -> Dict[str, Any]:
        """Load memory data for performance history analysis."""
        try:
            memory_path = Path(get_project_root()) / "experiments" / "evolving_memory" / "memory.json"
            if memory_path.exists():
                with open(memory_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"[PaperAggregatorAgent] Failed to load memory data: {e}")
        return {}

    def _get_performance_history_summary(self) -> str:
        """Generate performance history summary for LLM decision making."""
        if not self._memory_data:
            return "No previous iteration data available."

        iterations = self._memory_data.get("iterations", [])
        if not iterations:
            baseline = self._memory_data.get("baseline", {})
            if baseline:
                perf = baseline.get("performance", {})
                return f"Baseline only: {perf}"
            return "No iteration history. This is the first iteration."

        # Build summary
        summary_parts = []

        # Baseline
        baseline = self._memory_data.get("baseline", {})
        if baseline:
            summary_parts.append(f"Baseline ({baseline.get('model_name', 'unknown')}): {baseline.get('performance', {})}")

        # Last 3 iterations
        recent = iterations[-3:] if len(iterations) > 3 else iterations
        for iter_data in recent:
            iter_num = iter_data.get("iteration", "?")
            perf = iter_data.get("performance", {})
            analysis = iter_data.get("analysis", {})
            improved = analysis.get("improved")
            weights = iter_data.get("weights", {})
            beta = weights.get("beta", "N/A")

            status = "✅ improved" if improved else ("❌ decreased" if improved is False else "➖ first")
            summary_parts.append(f"Iter {iter_num}: {perf} | {status} | beta={beta}")

        # Overall stats
        consecutive_failures = self._memory_data.get("consecutive_failures", 0)
        best_iter = self._memory_data.get("best_iteration", 0)
        current_beta = self._memory_data.get("current_beta", 1.0)

        summary_parts.append(f"\n--- Stats ---")
        summary_parts.append(f"Best iteration: {best_iter}")
        summary_parts.append(f"Consecutive failures: {consecutive_failures}")
        summary_parts.append(f"Current beta: {current_beta}")

        return "\n".join(summary_parts)

    def create_agent(self):
        from langchain.chat_models import init_chat_model
        from langgraph.prebuilt import create_react_agent
        from configs.config import MODEL_NAME, MODEL_PROVIDER, MODEL_PARAMS, LANGGRAPH_CONFIG
        from agent_workflow.state import MARBLEState

        model = init_chat_model(
            MODEL_NAME,
            model_provider=MODEL_PROVIDER,
            **MODEL_PARAMS,
        )

        tools = self.get_additional_tools()
        prompt = self.get_prompt()

        agent_kwargs = {
            "model": model,
            "tools": tools,
            "prompt": prompt,
            "state_schema": MARBLEState,
        }

        if self.checkpointer is not None:
            agent_kwargs["checkpointer"] = self.checkpointer

        agent = create_react_agent(**agent_kwargs).with_config(
            recursion_limit=LANGGRAPH_CONFIG["recursion_limit"]
        )

        logger.info("[PaperAggregatorAgent] Agent created")
        return agent
