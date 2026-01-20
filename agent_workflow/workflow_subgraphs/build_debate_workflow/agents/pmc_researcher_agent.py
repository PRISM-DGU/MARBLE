"""PMC Researcher Agent for build_debate workflow.

Searches PMC/PubMed papers with parallel keyword search.
Filters by: Q1 journals, GitHub availability, PDF availability.

NOTE: GitHub is checked via DOI page and GitHub API search.
      Only papers with both GitHub AND PDF available are returned.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool
from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from .common_tools import (
    create_read_file_tool,
    create_write_file_tool,
    create_list_directory_tool,
)


class PMCResearcherAgent(BaseAgentNode):
    """Agent that searches PMC/PubMed for biomedical papers.

    Uses parallel keyword search with Q1 journal filtering.
    Filters: Q1 journals, GitHub availability, PDF availability.

    NOTE: This agent only returns candidates.
          Scoring and final selection are done by PaperAggregatorAgent.
    """

    # Allowed journals (high-impact bioinformatics, drug discovery, ML journals)
    ALLOWED_JOURNALS = {
        # Bioinformatics & Genomics
        "nature methods", "nature genetics", "genome biology", "nature communications",
        "nucleic acids research", "briefings in bioinformatics", "bioinformatics",
        "plos computational biology", "genome research", "cell systems",
        "molecular systems biology", "cell reports methods", "gigascience",
        "computational and structural biotechnology journal", "database",
        "bmc bioinformatics", "bmc biology", "current opinion in biotechnology",
        "molecular biology and evolution", "genomics, proteomics & bioinformatics",
        # Drug Discovery & Pharmacology
        "nature reviews drug discovery", "journal of cheminformatics",
        "journal of chemical information and modeling", "drug discovery today",
        "molecular pharmaceutics", "journal of medicinal chemistry",
        "pharmacological research", "advanced drug delivery reviews",
        "acs central science", "chemical science", "trends in pharmacological sciences",
        "medicinal research reviews", "acta pharmaceutica sinica b",
        "bioorganic & medicinal chemistry",
        # AI/ML & Digital Health
        "nature machine intelligence", "npj digital medicine",
        "ieee journal of biomedical and health informatics",
        "patterns", "ieee transactions on pattern analysis and machine intelligence",
        "ieee transactions on medical imaging", "knowledge-based systems",
        "expert systems with applications", "journal of biomedical informatics",
        "journal of the american medical informatics association",
        "medical image analysis",
        # Top-tier General Science
        "nature", "science", "cell", "nature biotechnology", "elife", "iscience",
        "cell reports", "pnas", "proceedings of the national academy of sciences",
        "science advances", "nature medicine",
    }

    def __init__(
        self,
        workspace_path: str = "",
        output_path: str = "",
        papers_dir: str = "",
        start_year: int = 2022,
        model_config: Optional[Dict[str, Any]] = None,
        checkpointer=None
    ):
        """Initialize PMC Researcher Agent.

        Args:
            workspace_path: Path to build workspace for model structure analysis
            output_path: Output file path for results JSON
            papers_dir: Directory for paper metadata
            start_year: Start year for paper search (default: 2022)
            model_config: Model configuration dict for dynamic prompts
            checkpointer: Optional LangGraph checkpointer
        """
        super().__init__("pmc_researcher", checkpointer=checkpointer)

        self.workspace_path = workspace_path
        self.output_path = output_path
        self.papers_dir = papers_dir
        self.start_year = start_year
        self.model_config = model_config

        # Filtering stats
        self.filter_stats = {
            "total_searched": 0,
            "github_passed": 0,
            "pdf_passed": 0,
            "candidates": [],
            "used_keywords": []
        }

        logger.debug(f"[PMCResearcherAgent] Initialized with workspace={workspace_path}")

    def get_prompt(self):
        """Return system prompt for PMC paper search."""
        model_name = self.model_config.get("model_name", "Unknown") if self.model_config else "Unknown"
        domain = self.model_config.get("domain", "machine learning") if self.model_config else "machine learning"
        domain_description = self.model_config.get("domain_description", "") if self.model_config else ""

        return f"""You are a PMC Researcher Agent that finds relevant biomedical papers from PubMed/PMC.

## Your Task
Find relevant papers for improving {model_name} in the domain of {domain}.

## Domain Context
- Domain: {domain}
- Description: {domain_description}

## KEYWORD STRUCTURE
You will generate 6 keywords total:
- **Domain keywords (3)**: Based on the domain context above
- **Weakness keywords (3)**: Based on model weakness analysis

## WORKFLOW

### Step 1: Analyze Model Structure
Call: `analyze_model_structure(path="{self.workspace_path}")`

This will show you:
- Code files in the model directory
- Model summary ({model_name}_summary.md)
- Weakness analysis (weakness_of_target_model.md)

### Step 2: Generate 6 Search Keywords

**Domain Keywords (3)**: Based on the domain context above, generate 3 keywords (2-4 words each).
- Focus on the core concepts of the domain
- Examples for drug response: "drug sensitivity prediction", "IC50 deep learning", "cancer cell response"

**Weakness Keywords (3)**: Based on the weakness analysis, generate 3 keywords (2-4 words each).
- Focus on architectures/methods that can address the model's identified weaknesses
- Examples: "graph attention mechanism", "multi-scale encoder", "cross-modal fusion"

### Step 3: Search & Filter (ONE call with 6 keywords!)
```
search_and_filter_papers(keywords=["domain_kw1", "domain_kw2", "domain_kw3", "weakness_kw1", "weakness_kw2", "weakness_kw3"])
```

The tool automatically:
- Searches PMC with all 6 keywords SEQUENTIALLY (1s delay)
- Filters by Q1 journals only
- Checks GitHub availability for EACH paper
- Checks PDF download availability (Europe PMC)
- Only adds candidates that pass BOTH GitHub AND PDF checks

### Step 4: Done!
After search_and_filter_papers completes, your job is done.
The PaperAggregatorAgent will handle scoring and final selection.

## Output
The tool saves candidates to: {self.output_path}
"""

    def get_additional_tools(self):
        """Return tools for PMC search."""
        return [
            create_read_file_tool(max_length=15000),
            create_write_file_tool(),
            create_list_directory_tool(),
            self._create_analyze_model_structure_tool(),
            self._create_search_and_filter_tool(),
        ]

    def _create_analyze_model_structure_tool(self):
        """Create tool to analyze model structure from workspace."""

        workspace_path = self.workspace_path
        model_config = self.model_config

        @tool
        def analyze_model_structure(path: str = "") -> Dict[str, Any]:
            """Analyze the model structure from the build workspace.

            Reads Python source files and components to understand the model architecture.

            Args:
                path: Path to workspace (default: from agent config)

            Returns:
                Dict with model files, key files, domain info, and weakness analysis
            """
            target_path = path or workspace_path
            if not target_path:
                return {"success": False, "error": "No workspace path provided"}

            workspace = Path(target_path)
            if not workspace.exists():
                return {"success": False, "error": f"Workspace not found: {target_path}"}

            result = {
                "success": True,
                "workspace": str(workspace),
                "model_name": model_config.get("model_name", "Unknown") if model_config else "Unknown",
                "domain": model_config.get("domain", "machine learning") if model_config else "machine learning",
                "all_files": [],
                "key_files": [],
                "weakness_analysis": None,
            }

            # Scan for Python files
            py_files = list(workspace.rglob("*.py"))

            for py_file in py_files:
                file_name = py_file.name.lower()
                rel_path = str(py_file.relative_to(workspace))

                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    if len(content) > 8000:
                        content = content[:8000] + "\n... (truncated)"

                    file_info = {"path": rel_path, "content": content}
                    result["all_files"].append(file_info)

                    if "encoder" in file_name or "decoder" in file_name:
                        result["key_files"].append(rel_path)

                except Exception as e:
                    logger.debug(f"Error reading {py_file}: {e}")

            # Read weakness analysis if available
            weakness_path = workspace / "weakness.md"
            if weakness_path.exists():
                try:
                    result["weakness_analysis"] = weakness_path.read_text(encoding='utf-8')[:5000]
                except Exception as e:
                    logger.debug(f"Error reading weakness analysis: {e}")

            logger.info(f"[ANALYZE_MODEL] Found: {len(result['all_files'])} files, {len(result['key_files'])} key files")
            return result

        return analyze_model_structure

    def _create_search_and_filter_tool(self):
        """Create tool for parallel PMC search with 6 keywords."""

        papers_dir = self.papers_dir
        filter_stats = self.filter_stats
        allowed_journals = self.ALLOWED_JOURNALS
        output_path = self.output_path

        def _search_pmc_single(keyword: str) -> List[Dict]:
            """Search PMC with a single keyword."""
            try:
                from pymed import PubMed
            except ImportError:
                logger.error("[PMC_SEARCH] pymed not installed")
                return []

            current_year = datetime.now().year
            query = f"({keyword}) AND 2022[PDAT]:{current_year}[PDAT]"

            try:
                pubmed = PubMed(tool="MARBLE", email="drp-agent@example.com")
                articles = list(pubmed.query(query, max_results=500))
                logger.info(f"[PMC_SEARCH] Keyword '{keyword}': found {len(articles)} articles")
            except Exception as e:
                logger.error(f"[PMC_SEARCH] Error for '{keyword}': {e}")
                return []

            papers = []
            for article in articles:
                try:
                    journal = (getattr(article, 'journal', '') or "").lower()
                    if not any(allowed in journal for allowed in allowed_journals):
                        continue

                    pub_date = article.publication_date
                    year = None
                    if pub_date:
                        if hasattr(pub_date, 'year'):
                            year = pub_date.year
                        elif isinstance(pub_date, str) and len(pub_date) >= 4:
                            year = int(pub_date[:4])

                    paper = {
                        "pmid": str(article.pubmed_id).split('\n')[0] if article.pubmed_id else None,
                        "title": article.title or "",
                        "abstract": article.abstract or "",
                        "doi": article.doi or "",
                        "journal": journal,
                        "year": year,
                        "source": "pmc",
                        "keyword": keyword
                    }
                    if paper["title"] and paper["abstract"] and paper["pmid"]:
                        papers.append(paper)
                except Exception:
                    continue

            return papers

        def _check_pdf_available(pmid: str) -> tuple:
            """Check if PDF is available via Europe PMC."""
            if not pmid:
                return False, None

            try:
                pmc_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
                resp = requests.get(pmc_api, timeout=10)

                if resp.status_code != 200:
                    return False, None

                records = resp.json().get("records", [])
                if not records or not records[0].get("pmcid"):
                    return False, None

                pmcid = records[0]["pmcid"]

                pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                resp = requests.get(pdf_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}, stream=True)

                if resp.status_code == 200:
                    content_type = resp.headers.get('Content-Type', '')
                    first_chunk = next(resp.iter_content(chunk_size=1024), b'')
                    if 'pdf' in content_type.lower() or first_chunk[:4] == b'%PDF':
                        return True, pmcid

                return False, None
            except Exception as e:
                logger.debug(f"[PDF_CHECK] Error for PMID {pmid}: {e}")
                return False, None

        def _check_github_available(paper: Dict) -> List[str]:
            """Check if paper has accessible GitHub repository.

            Simplified: just check if DOI page contains github.com links.
            No GitHub API validation (to avoid rate limits).
            """
            doi = paper.get("doi", "")
            github_urls = []

            if doi:
                try:
                    resp = requests.get(f"https://doi.org/{doi}", timeout=10, allow_redirects=True)
                    if resp.status_code == 200:
                        # Find all GitHub URLs in the page
                        matches = re.findall(r'https?://github\.com/[\w\-]+/[\w\-\.]+', resp.text, re.IGNORECASE)
                        for url in matches[:3]:
                            clean_url = url.rstrip('/').rstrip('.git')
                            github_urls.append(clean_url)
                except Exception:
                    pass

            return list(set(github_urls))

        def _verify_paper(paper: Dict) -> Optional[Dict]:
            """Verify single paper: GitHub + PDF check."""
            github_urls = _check_github_available(paper)
            if not github_urls:
                return None

            pdf_available, pmcid = _check_pdf_available(paper.get("pmid", ""))
            if not pdf_available:
                return None

            return {
                "pmid": paper["pmid"],
                "pmcid": pmcid,
                "title": paper["title"],
                "abstract": paper["abstract"],
                "doi": paper["doi"],
                "journal": paper.get("journal", ""),
                "year": paper["year"],
                "github_urls": github_urls,
                "pdf_available": True,
                "source": "pmc",
                "keyword": paper.get("keyword", "")
            }

        @tool
        def search_and_filter_papers(
            keywords: List[str]
        ) -> Dict[str, Any]:
            """Search PMC with 6 keywords SEQUENTIALLY (1s delay) and filter by GitHub + PDF availability.

            This tool:
            1. Searches PMC with 6 keywords sequentially (500 papers each, 1s delay between keywords)
            2. Removes duplicates by PMID
            3. Filters by Q1 journals
            4. Checks GitHub availability via DOI page and GitHub API search
            5. Checks PDF availability via Europe PMC (parallel)
            6. Returns candidates that have BOTH GitHub AND PDF available
            7. Saves all candidates to output JSON for PaperAggregatorAgent

            Args:
                keywords: List of 6 search keywords (3 domain + 3 architecture keywords)

            Returns:
                Dict with filtered candidates that have GitHub + PDF available
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

            logger.info(f"[PMC_SEARCH] ========== Sequential Search with {len(cleaned_keywords)} keywords ==========")
            logger.info(f"[PMC_SEARCH] Keywords: {cleaned_keywords}")

            # Step 1: Sequential search with 1 second delay between keywords
            all_papers = []
            for i, kw in enumerate(cleaned_keywords):
                try:
                    papers = _search_pmc_single(kw)
                    all_papers.extend(papers)
                    logger.info(f"[PMC_SEARCH] '{kw}': {len(papers)} papers from Q1 journals")
                except Exception as e:
                    logger.error(f"[PMC_SEARCH] '{kw}' failed: {e}")

                # 1 second delay between keywords (except after last one)
                if i < len(cleaned_keywords) - 1:
                    import time
                    time.sleep(1.0)
                    logger.debug(f"[PMC_SEARCH] Waiting 1s before next keyword...")

            logger.info(f"[PMC_SEARCH] Total papers before dedup: {len(all_papers)}")

            # Step 2: Remove duplicates
            unique_papers = {}
            for paper in all_papers:
                pmid = paper.get("pmid")
                if pmid and pmid not in unique_papers:
                    unique_papers[pmid] = paper

            papers_list = list(unique_papers.values())
            logger.info(f"[PMC_SEARCH] After dedup: {len(papers_list)} unique papers")

            # Step 3: Parallel verification
            logger.info(f"[PMC_SEARCH] Verifying GitHub + PDF for {len(papers_list)} papers...")
            candidates = []

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(_verify_paper, p): p for p in papers_list}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            candidates.append(result)
                            logger.info(f"[PMC_SEARCH] ✅ {result['title'][:50]}...")
                    except Exception as e:
                        logger.debug(f"[PMC_SEARCH] Verification error: {e}")

            logger.info(f"[PMC_SEARCH] ========== Search Complete ==========")
            logger.info(f"[PMC_SEARCH] 📊 {len(papers_list)} searched → {len(candidates)} passed (GitHub+PDF)")

            # Update filter_stats
            filter_stats["candidates"] = candidates
            filter_stats["used_keywords"] = cleaned_keywords
            filter_stats["total_searched"] = len(all_papers)
            filter_stats["github_passed"] = len(candidates)
            filter_stats["pdf_passed"] = len(candidates)

            # Save candidates to output JSON for PaperAggregatorAgent
            if output_path:
                output_data = {
                    "candidates": candidates,
                    "search_metadata": {
                        "source": "pmc",
                        "total_searched": len(all_papers),
                        "unique_papers": len(papers_list),
                        "candidates_count": len(candidates),
                        "keywords_used": cleaned_keywords,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                logger.info(f"[PMC_SEARCH] ✅ Saved {len(candidates)} candidates to: {output_path}")

            # Return summary for agent
            return {
                "success": True,
                "total_searched": len(all_papers),
                "unique_papers": len(papers_list),
                "total_candidates": len(candidates),
                "keywords_used": cleaned_keywords,
                "output_saved": output_path,
                "message": f"✅ Found {len(candidates)} candidates with GitHub+PDF. Saved to {output_path}. PaperAggregatorAgent will handle scoring.",
                "candidates_preview": [
                    {
                        "title": c.get("title", "")[:80],
                        "journal": c.get("journal", ""),
                        "year": c.get("year")
                    }
                    for c in candidates[:5]
                ]
            }

        return search_and_filter_papers

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

        logger.info("[PMCResearcherAgent] Agent created")
        return agent
