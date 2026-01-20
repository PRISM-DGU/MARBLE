"""OpenReview Researcher Agent for build_debate workflow.

Searches ICLR, NeurIPS, ICML papers from OpenReview with parallel keyword search.
Filters by: Accepted papers only, PDF availability, GitHub availability.

NOTE: PDF and GitHub are checked in ONE pass during search (no separate download step).
      This is more efficient than the previous two-step approach.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from .common_tools import (
    create_read_file_tool,
    create_write_file_tool,
    create_list_directory_tool,
)


class OpenReviewResearcherAgent(BaseAgentNode):
    """Agent that searches OpenReview (ICLR, NeurIPS, ICML) for ML papers.

    Uses parallel keyword search similar to PMC researcher.
    Filters: Top ML conferences (Accepted only), PDF + GitHub availability.

    NOTE: PDF and GitHub are checked in ONE pass (efficient single-step filtering).
    """

    # Target venues (Accepted papers from top ML conferences)
    TARGET_VENUES = {
        # ICLR
        "ICLR.cc/2024/Conference": "ICLR 2024",
        "ICLR.cc/2025/Conference": "ICLR 2025",
        # NeurIPS
        "NeurIPS.cc/2024/Conference": "NeurIPS 2024",
        "NeurIPS.cc/2023/Conference": "NeurIPS 2023",
        # ICML
        "ICML.cc/2024/Conference": "ICML 2024",
        "ICML.cc/2023/Conference": "ICML 2023",
    }

    def __init__(
        self,
        workspace_path: str = "",
        output_path: str = "",
        papers_dir: str = "",
        model_config: Optional[Dict[str, Any]] = None,
        checkpointer=None
    ):
        """Initialize OpenReview Researcher Agent.

        Args:
            workspace_path: Path to build workspace for model structure analysis
            output_path: Output file path for results JSON
            papers_dir: Directory for downloaded papers
            model_config: Model configuration dict for dynamic prompts
            checkpointer: Optional LangGraph checkpointer
        """
        super().__init__("openreview_researcher", checkpointer=checkpointer)

        self.workspace_path = workspace_path
        self.output_path = output_path
        self.papers_dir = papers_dir
        self.model_config = model_config

        # Filtering stats
        self.filter_stats = {
            "total_searched": 0,
            "github_passed": 0,
            "pdf_passed": 0,
            "candidates": [],
            "used_keywords": []
        }

        logger.debug(f"[OpenReviewResearcherAgent] Initialized with workspace={workspace_path}")

    def get_prompt(self):
        """Return system prompt for OpenReview paper search."""
        model_name = self.model_config.get("model_name", "Unknown") if self.model_config else "Unknown"
        domain = self.model_config.get("domain", "machine learning") if self.model_config else "machine learning"
        domain_description = self.model_config.get("domain_description", "") if self.model_config else ""

        return f"""You are an OpenReview Researcher Agent that finds ML architecture papers from top conferences.

## Your Task
Find relevant ML papers from ICLR, NeurIPS, ICML for improving {model_name} in the domain of {domain}.

## Domain Context
- Domain: {domain}
- Description: {domain_description}

## KEYWORD STRUCTURE
You will generate 6 keywords total:
- **Domain keywords (3)**: Based on the domain context above
- **Weakness keywords (3)**: Based on model weakness analysis

## WORKFLOW

### Step 1: Read Model Analysis Files
Use `read_file` to read these files in the workspace ({self.workspace_path}):
1. `{model_name.lower()}_summary.md` - Model architecture summary
2. `weakness_of_target_model.md` - Identified weaknesses

### Step 2: Generate 6 Search Keywords

**Domain Keywords (3)**: Based on the domain context above, generate 3 keywords (2-3 words each).
- Focus on the core concepts of the domain
- Use general ML terminology that appears frequently in top ML conferences
- Examples for drug response: "drug discovery", "molecular property", "biomedical prediction"

**Weakness Keywords (3)**: Based on the weakness analysis, generate 3 keywords (2-3 words each).
- Focus on architectures/methods that can address the model's identified weaknesses
- Examples: "graph neural network", "attention mechanism", "multi-modal learning"

### Step 3: Search OpenReview (ONE call with 6 keywords!)
```
search_openreview_papers(keywords=["domain_kw1", "domain_kw2", "domain_kw3", "weakness_kw1", "weakness_kw2", "weakness_kw3"])
```

The tool uses SINGLE-PASS filtering:
- Searches ICLR, NeurIPS, ICML with all 6 keywords IN PARALLEL
- For each paper, checks: keyword + Accepted + PDF + GitHub
- Only returns papers that meet ALL criteria

### Step 4: Done!
Results are saved to: {self.output_path}
These will be merged with PMC results by the PaperAggregatorAgent.
"""

    def get_additional_tools(self):
        """Return tools for OpenReview search."""
        return [
            create_read_file_tool(max_length=15000),
            create_write_file_tool(),
            create_list_directory_tool(),
            self._create_search_openreview_tool(),
        ]

    def _create_search_openreview_tool(self):
        """Create tool for OpenReview search with PDF + GitHub filtering in ONE pass."""

        papers_dir = self.papers_dir
        filter_stats = self.filter_stats
        target_venues = self.TARGET_VENUES
        output_path = self.output_path

        # GitHub URL pattern
        GITHUB_PATTERN = re.compile(r'github\.com/[a-zA-Z0-9\-_./]+', re.IGNORECASE)

        def _extract_github_from_content(content: dict, abstract: str) -> Optional[str]:
            """Extract GitHub URL from OpenReview content fields or abstract text."""
            # Check code-related fields first
            code_fields = ['code', 'software', 'program_website', 'supplementary_material', 'code_of_ethics']
            for field in code_fields:
                val = content.get(field, {})
                if isinstance(val, dict):
                    val = val.get('value', '')
                if val and 'github.com' in str(val).lower():
                    # Clean and return the URL
                    match = GITHUB_PATTERN.search(str(val))
                    if match:
                        return f"https://{match.group()}"

            # Search in abstract text
            if abstract:
                match = GITHUB_PATTERN.search(abstract)
                if match:
                    return f"https://{match.group()}"

            return None

        def _search_openreview_single(keyword: str, venue_id: str) -> List[Dict]:
            """Search OpenReview with a single keyword for a single venue.

            Filters in ONE pass: Keyword + Accepted + PDF + GitHub
            """
            try:
                import openreview
            except ImportError:
                logger.error("[OPENREVIEW] openreview-py not installed. Run: pip install openreview-py")
                return []

            try:
                # Use API v2 client (no auth required for public data)
                client = openreview.api.OpenReviewClient(
                    baseurl='https://api2.openreview.net'
                )

                # Get all submissions for this venue
                try:
                    notes = client.get_all_notes(invitation=f'{venue_id}/-/Submission')
                except Exception:
                    # Fallback to venueid-based search
                    notes = client.get_all_notes(content={"venueid": venue_id})

                keyword_lower = keyword.lower()
                matched_papers = []

                # Acceptance status keywords (lowercase for case-insensitive matching)
                ACCEPTED_STATUSES = ['accepted', 'oral', 'poster', 'spotlight', 'accept']
                # Rejection status keywords to explicitly exclude
                REJECTED_STATUSES = ['withdrawn', 'rejected', 'desk rejected', 'submitted to']

                for note in notes:
                    try:
                        content = note.content or {}

                        # Extract title and abstract (API v2 format)
                        title = content.get("title", {})
                        if isinstance(title, dict):
                            title = title.get("value", "")

                        abstract = content.get("abstract", {})
                        if isinstance(abstract, dict):
                            abstract = abstract.get("value", "")

                        # A. Keyword check
                        if keyword_lower not in title.lower() and keyword_lower not in abstract.lower():
                            continue

                        # B. Accepted check (venue field contains acceptance status)
                        venue_field = content.get("venue", {})
                        if isinstance(venue_field, dict):
                            venue_value = venue_field.get("value", "")
                        else:
                            venue_value = str(venue_field) if venue_field else ""

                        venue_lower = venue_value.lower()

                        # Check for rejection first
                        is_rejected = any(status in venue_lower for status in REJECTED_STATUSES)
                        if is_rejected:
                            continue

                        # Check for acceptance (case-insensitive)
                        is_accepted = any(status in venue_lower for status in ACCEPTED_STATUSES)
                        if not is_accepted:
                            continue  # Not accepted

                        # C. PDF check (field existence)
                        pdf_field = content.get("pdf", {})
                        if isinstance(pdf_field, dict):
                            pdf_value = pdf_field.get("value", "")
                        else:
                            pdf_value = pdf_field

                        if not pdf_value:
                            continue  # No PDF

                        # D. GitHub check (from fields or abstract)
                        github_url = _extract_github_from_content(content, abstract)
                        if not github_url:
                            continue  # No GitHub

                        # All 4 checks passed - add paper
                        pdf_url = f"https://openreview.net{pdf_value}" if pdf_value.startswith('/') else f"https://openreview.net/pdf?id={note.id}"

                        paper = {
                            "id": note.id,
                            "title": title,
                            "abstract": abstract,
                            "pdf_url": pdf_url,
                            "github_urls": [github_url],
                            "venue": target_venues.get(venue_id, venue_id),
                            "venue_id": venue_id,
                            "venue_status": venue_value,  # e.g., "ICLR 2024 Poster"
                            "year": int(venue_id.split("/")[1][:4]) if "/" in venue_id else None,
                            "pdf_available": True,
                            "source": "openreview",
                            "keyword": keyword
                        }

                        matched_papers.append(paper)

                    except Exception as e:
                        logger.debug(f"[OPENREVIEW] Error parsing note: {e}")
                        continue

                logger.info(f"[OPENREVIEW] '{keyword}' @ {target_venues.get(venue_id, venue_id)}: {len(matched_papers)} (Accepted+PDF+GitHub)")
                return matched_papers

            except Exception as e:
                logger.error(f"[OPENREVIEW] Search error for '{keyword}' @ {venue_id}: {e}")
                return []

        def _search_openreview_keyword(keyword: str) -> List[Dict]:
            """Search all venues for a single keyword."""
            all_papers = []
            for venue_id in target_venues.keys():
                papers = _search_openreview_single(keyword, venue_id)
                all_papers.extend(papers)
            return all_papers

        @tool
        def search_openreview_papers(
            keywords: List[str]
        ) -> Dict[str, Any]:
            """Search OpenReview (ICLR, NeurIPS, ICML) with 6 keywords IN PARALLEL.

            This tool filters Accepted + PDF + GitHub in ONE pass (efficient single-step filtering):
            1. Searches all target venues with ALL 6 keywords in parallel
            2. For each paper, checks: keyword match + Accepted + PDF field + GitHub (fields/abstract)
            3. Removes duplicates by paper ID
            4. Returns candidates that have ALL THREE: Accepted + PDF + GitHub

            No separate verification step needed - all filtering done during search!

            Args:
                keywords: List of 6 search keywords (3 domain + 3 architecture keywords)

            Returns:
                Dict with filtered candidates that have GitHub AND PDF available
            """
            if isinstance(keywords, str):
                keywords = [kw.strip() for kw in keywords.split(",")]

            # Clean keywords
            cleaned_keywords = []
            for kw in keywords:
                kw = re.sub(r'\bgithub\b', '', kw, flags=re.IGNORECASE).strip()
                kw = re.sub(r'\s+', ' ', kw)
                if kw:
                    cleaned_keywords.append(kw)

            if not cleaned_keywords:
                return {"success": False, "error": "No valid keywords provided"}

            logger.info(f"[OPENREVIEW] ========== Single-Pass Search (Accepted+PDF+GitHub) ==========")
            logger.info(f"[OPENREVIEW] Keywords: {cleaned_keywords}")
            logger.info(f"[OPENREVIEW] Target venues: {list(target_venues.values())}")

            # Step 1: Parallel search with all keywords across all venues
            # Each search already filters by PDF + GitHub in one pass
            all_papers = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(_search_openreview_keyword, kw): kw for kw in cleaned_keywords}
                for future in as_completed(futures):
                    kw = futures[future]
                    try:
                        papers = future.result()
                        all_papers.extend(papers)
                        logger.info(f"[OPENREVIEW] '{kw}': {len(papers)} papers (already PDF+GitHub filtered)")
                    except Exception as e:
                        logger.error(f"[OPENREVIEW] '{kw}' failed: {e}")

            logger.info(f"[OPENREVIEW] Total papers before dedup: {len(all_papers)}")

            # Step 2: Remove duplicates by paper ID
            unique_papers = {}
            for paper in all_papers:
                paper_id = paper.get("id")
                if paper_id and paper_id not in unique_papers:
                    unique_papers[paper_id] = paper

            candidates = list(unique_papers.values())
            logger.info(f"[OPENREVIEW] After dedup: {len(candidates)} unique candidates")

            # No Step 3 needed - already filtered during search!

            logger.info(f"[OPENREVIEW] ========== Search Complete ==========")
            logger.info(f"[OPENREVIEW] 📊 {len(candidates)} candidates (Accepted+PDF+GitHub)")

            # Update filter_stats
            filter_stats["candidates"] = candidates
            filter_stats["used_keywords"] = cleaned_keywords
            filter_stats["total_searched"] = len(all_papers)
            filter_stats["github_passed"] = len(candidates)
            filter_stats["pdf_passed"] = len(candidates)

            # Save candidates to output JSON for PaperAggregatorAgent (same format as PMC)
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                output_data = {
                    "candidates": candidates,  # Same key as PMC for consistency
                    "search_metadata": {
                        "source": "openreview",
                        "venues": list(target_venues.values()),
                        "total_searched": len(all_papers),
                        "unique_papers": len(candidates),
                        "candidates_count": len(candidates),
                        "keywords_used": cleaned_keywords,
                        "timestamp": datetime.now().isoformat(),
                        "filter_method": "single_pass"  # Indicates new efficient method
                    }
                }
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                logger.info(f"[OPENREVIEW] ✅ Saved {len(candidates)} candidates to: {output_path}")

            return {
                "success": True,
                "total_searched": len(all_papers),
                "unique_papers": len(candidates),
                "total_candidates": len(candidates),
                "keywords_used": cleaned_keywords,
                "venues_searched": list(target_venues.values()),
                "output_saved": output_path,
                "message": f"✅ Found {len(candidates)} candidates (Accepted+PDF+GitHub, single-pass). Ready for aggregation.",
                "candidates_preview": [
                    {
                        "title": c.get("title", "")[:80],
                        "venue": c.get("venue", ""),
                        "year": c.get("year"),
                        "github": c.get("github_urls", [""])[0][:50] if c.get("github_urls") else ""
                    }
                    for c in candidates[:5]
                ]
            }

        return search_openreview_papers

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

        logger.info("[OpenReviewResearcherAgent] Agent created")
        return agent
