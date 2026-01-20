"""
Paper Download Script for 4 Domains

Downloads papers from PMC and OpenReview for each domain.
Filters: PDF available + GitHub available
Target: ~200 papers per domain

Domains:
1. spatial_transcriptomics 
2. drug_response_prediction 
3. drug_target_interaction
4. drug_repurposing 
"""

import json
import os
import re
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv


def log(msg: str):
    """Print with flush for real-time output."""
    print(msg, flush=True)

# =============================================================================
# Configuration
# =============================================================================

# Load .env file from project root
ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/data/project/atwoddl/DRAgent")
BASE_DIR = Path(PROJECT_ROOT) / "experiments" / "pdf"
MAX_PAPERS_PER_DOMAIN = 500
PMC_MAX_PER_KEYWORD = 800
OPENREVIEW_VENUES = {
    "ICLR.cc/2024/Conference": "ICLR 2024",
    "ICLR.cc/2025/Conference": "ICLR 2025",
    "NeurIPS.cc/2024/Conference": "NeurIPS 2024",
    "NeurIPS.cc/2023/Conference": "NeurIPS 2023",
    "ICML.cc/2024/Conference": "ICML 2024",
    "ICML.cc/2023/Conference": "ICML 2023",
}

# Q1 Journals for PMC filtering
ALLOWED_JOURNALS = {
    "nature methods", "nature genetics", "genome biology", "nature communications",
    "nucleic acids research", "briefings in bioinformatics", "bioinformatics",
    "plos computational biology", "genome research", "cell systems",
    "molecular systems biology", "cell reports methods", "gigascience",
    "computational and structural biotechnology journal", "database",
    "bmc bioinformatics", "bmc biology", "current opinion in biotechnology",
    "nature reviews drug discovery", "journal of cheminformatics",
    "journal of chemical information and modeling", "drug discovery today",
    "molecular pharmaceutics", "journal of medicinal chemistry",
    "pharmacological research", "advanced drug delivery reviews",
    "acs central science", "chemical science", "trends in pharmacological sciences",
    "nature machine intelligence", "npj digital medicine",
    "ieee journal of biomedical and health informatics",
    "patterns", "ieee transactions on pattern analysis and machine intelligence",
    "knowledge-based systems", "expert systems with applications",
    "journal of biomedical informatics", "medical image analysis",
    "nature", "science", "cell", "nature biotechnology", "elife", "iscience",
    "cell reports", "pnas", "proceedings of the national academy of sciences",
    "science advances", "nature medicine",
}

# =============================================================================
# Domain Configuration
# =============================================================================

@dataclass
class DomainConfig:
    name: str
    folder_name: str
    keywords: List[str] = field(default_factory=list)
    description: str = ""


DOMAINS = {
    "spatial_transcriptomics": DomainConfig(
        name="Spatial Transcriptomics",
        folder_name="spatial_transcriptomics",
        description="Analyzing spatial gene expression patterns and tissue domains",
        keywords=[
            # Domain keywords
            "spatial transcriptomics deep learning",
            "spatial gene expression clustering",
            "single cell spatial analysis",
            "tissue domain identification",
            "spatial omics neural network",
            "cell type annotation spatial",
            "spatial proteomics machine learning",
            "tissue segmentation deep learning",
            "spatial resolved transcriptomics",
            "multiplexed imaging analysis",
            # Architecture keywords
            "graph attention network gene expression",
            "variational autoencoder single cell",
            "deep embedded clustering",
            "graph neural network spatial",
            "self-supervised learning single cell",
            "contrastive learning gene expression",
            "transformer single cell",
            "autoencoder gene expression",
            "graph convolutional network biology",
            "attention mechanism bioinformatics",
        ]
    ),
    "drug_response_prediction": DomainConfig(
        name="Drug Response Prediction",
        folder_name="drug_response_prediction",
        description="Predicting drug sensitivity/response using molecular and cell features",
        keywords=[
            # Domain keywords
            "drug response prediction deep learning",
            "drug sensitivity prediction",
            "IC50 prediction neural network",
            "cancer drug response",
            "pharmacogenomics deep learning",
            "drug efficacy prediction",
            "cell line drug response",
            "anti-cancer drug prediction",
            "chemosensitivity prediction",
            "drug synergy prediction",
            # Architecture keywords
            "transformer molecular representation",
            "graph isomorphism network molecule",
            "SMILES representation learning",
            "gene expression encoder neural network",
            "molecular graph neural network",
            "multi-omics integration deep learning",
            "attention mechanism drug",
            "graph transformer molecule",
            "message passing neural network",
            "molecular fingerprint deep learning",
        ]
    ),
    "drug_target_interaction": DomainConfig(
        name="Drug-Target Interaction",
        folder_name="drug_target_interaction",
        description="Predicting drug-target interactions and drug discovery",
        keywords=[
            # Domain keywords
            "drug target interaction prediction",
            "drug discovery deep learning",
            "DTI prediction neural network",
            "compound protein interaction",
            "molecular docking deep learning",
            "binding affinity prediction",
            "ligand protein interaction",
            "virtual screening machine learning",
            "drug target affinity",
            "compound target prediction",
            # Architecture keywords
            "heterogeneous network embedding",
            "knowledge graph drug discovery",
            "pathway neural network",
            "random walk graph embedding",
            "graph neural network protein",
            "protein language model",
            "molecular representation learning",
            "multi-task learning drug",
            "transfer learning drug discovery",
            "self-supervised molecular",
        ]
    ),
    "drug_repurposing": DomainConfig(
        name="Drug Repurposing",
        folder_name="drug_repurposing",
        description="Predicting drug-target interactions for drug repurposing",
        keywords=[
            # Domain keywords
            "drug repurposing deep learning",
            "drug repositioning neural network",
            "drug target binding prediction",
            "virtual screening deep learning",
            "drug indication prediction",
            "off-target prediction",
            "polypharmacology prediction",
            "drug side effect prediction",
            "drug drug interaction prediction",
            "adverse drug reaction prediction",
            # Architecture keywords
            "bilinear attention network",
            "graph convolutional network molecule",
            "cross attention drug protein",
            "CNN protein sequence encoding",
            "recurrent neural network drug",
            "sequence to sequence drug",
            "BERT molecular",
            "pretrained molecular model",
            "few-shot learning drug",
            "meta-learning drug discovery",
        ]
    ),
}

# =============================================================================
# PMC Search Functions
# =============================================================================

def search_pmc_single(keyword: str, max_results: int = PMC_MAX_PER_KEYWORD) -> List[Dict]:
    """Search PMC with a single keyword."""
    try:
        from pymed import PubMed
    except ImportError:
        log("[PMC] ERROR: pymed not installed. Run: pip install pymed")
        return []

    current_year = datetime.now().year
    query = f"({keyword}) AND 2022[PDAT]:{current_year}[PDAT]"

    log(f"[PMC] Searching: '{keyword}'...")
    try:
        pubmed = PubMed(tool="DRAgent", email="drp-agent@example.com")
        articles = list(pubmed.query(query, max_results=max_results))
        log(f"[PMC] '{keyword}': {len(articles)} raw articles found")
    except Exception as e:
        log(f"[PMC] ERROR for '{keyword}': {e}")
        return []

    papers = []
    for article in articles:
        try:
            journal = (getattr(article, 'journal', '') or "").lower()
            if not any(allowed in journal for allowed in ALLOWED_JOURNALS):
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


def check_pmc_pdf_available(pmid: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if PDF is available via Europe PMC. Returns (available, pmcid, pdf_url)."""
    if not pmid:
        return False, None, None

    try:
        pmc_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
        resp = requests.get(pmc_api, timeout=10)

        if resp.status_code != 200:
            return False, None, None

        records = resp.json().get("records", [])
        if not records or not records[0].get("pmcid"):
            return False, None, None

        pmcid = records[0]["pmcid"]
        pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"

        # Quick check if PDF is accessible
        resp = requests.get(pdf_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        if resp.status_code == 200:
            content_type = resp.headers.get('Content-Type', '')
            first_chunk = next(resp.iter_content(chunk_size=1024), b'')
            if 'pdf' in content_type.lower() or first_chunk[:4] == b'%PDF':
                return True, pmcid, pdf_url

        return False, None, None
    except Exception as e:
        return False, None, None


def check_github_available(paper: Dict) -> List[str]:
    """Check if paper has accessible GitHub repository via DOI page."""
    doi = paper.get("doi", "")
    github_urls = []

    if doi:
        try:
            resp = requests.get(f"https://doi.org/{doi}", timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                matches = re.findall(r'https?://github\.com/[\w\-]+/[\w\-\.]+', resp.text, re.IGNORECASE)
                for url in matches[:3]:
                    clean_url = url.rstrip('/').rstrip('.git')
                    github_urls.append(clean_url)
        except Exception:
            pass

    return list(set(github_urls))


def verify_pmc_paper(paper: Dict) -> Optional[Dict]:
    """Verify single PMC paper: GitHub + PDF check (both required)."""
    title_short = paper.get("title", "")[:40]

    # GitHub check first (required)
    github_urls = check_github_available(paper)
    if not github_urls:
        return None

    log(f"  [VERIFY] GitHub found: {title_short}...")

    # PDF check (required)
    pdf_available, pmcid, pdf_url = check_pmc_pdf_available(paper.get("pmid", ""))
    if not pdf_available:
        return None

    log(f"  [VERIFY] PDF+GitHub OK: {title_short}...")

    return {
        "pmid": paper["pmid"],
        "pmcid": pmcid,
        "title": paper["title"],
        "abstract": paper["abstract"],
        "doi": paper["doi"],
        "journal": paper.get("journal", ""),
        "year": paper["year"],
        "github_urls": github_urls,
        "pdf_url": pdf_url,
        "pdf_available": True,
        "source": "pmc",
        "keyword": paper.get("keyword", "")
    }


def search_pmc_domain(keywords: List[str]) -> List[Dict]:
    """Search PMC for all keywords and verify papers."""
    log(f"\n[PMC] {'='*50}")
    log(f"[PMC] PHASE 1: Searching with {len(keywords)} keywords")
    log(f"[PMC] {'='*50}")

    # Sequential search with delay
    all_papers = []
    for i, kw in enumerate(keywords):
        log(f"\n[PMC] Keyword {i+1}/{len(keywords)}")
        papers = search_pmc_single(kw)
        q1_count = len(papers)
        all_papers.extend(papers)
        log(f"[PMC] After Q1 filter: {q1_count} papers")
        if i < len(keywords) - 1:
            log(f"[PMC] Waiting 1s before next keyword...")
            time.sleep(1.0)

    log(f"\n[PMC] {'='*50}")
    log(f"[PMC] PHASE 2: Deduplication")
    log(f"[PMC] Total raw papers: {len(all_papers)}")

    # Deduplicate by PMID
    unique_papers = {}
    for paper in all_papers:
        pmid = paper.get("pmid")
        if pmid and pmid not in unique_papers:
            unique_papers[pmid] = paper

    papers_list = list(unique_papers.values())
    log(f"[PMC] After dedup: {len(papers_list)} unique papers")

    log(f"\n[PMC] {'='*50}")
    log(f"[PMC] PHASE 3: Verifying GitHub + PDF")
    log(f"[PMC] Checking {len(papers_list)} papers (parallel, 10 workers)...")
    log(f"[PMC] {'='*50}")

    candidates = []
    verified_count = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(verify_pmc_paper, p): p for p in papers_list}
        for future in as_completed(futures):
            verified_count += 1
            if verified_count % 50 == 0:
                log(f"[PMC] Progress: {verified_count}/{len(papers_list)} checked, {len(candidates)} passed")
            try:
                result = future.result()
                if result:
                    candidates.append(result)
            except Exception:
                pass

    log(f"\n[PMC] {'='*50}")
    log(f"[PMC] RESULT: {len(candidates)} papers with GitHub+PDF")
    log(f"[PMC] {'='*50}")
    return candidates


# =============================================================================
# OpenReview Search Functions
# =============================================================================

GITHUB_PATTERN = re.compile(r'github\.com/[a-zA-Z0-9\-_./]+', re.IGNORECASE)


def extract_github_from_content(content: dict, abstract: str) -> Optional[str]:
    """Extract GitHub URL from OpenReview content fields or abstract."""
    code_fields = ['code', 'software', 'program_website', 'supplementary_material', 'code_of_ethics']
    for field in code_fields:
        val = content.get(field, {})
        if isinstance(val, dict):
            val = val.get('value', '')
        if val and 'github.com' in str(val).lower():
            match = GITHUB_PATTERN.search(str(val))
            if match:
                return f"https://{match.group()}"

    if abstract:
        match = GITHUB_PATTERN.search(abstract)
        if match:
            return f"https://{match.group()}"

    return None


def search_openreview_single(keyword: str, venue_id: str) -> List[Dict]:
    """Search OpenReview with a single keyword for a single venue."""
    try:
        import openreview
    except ImportError:
        log("[OPENREVIEW] ERROR: openreview-py not installed. Run: pip install openreview-py")
        return []

    venue_name = OPENREVIEW_VENUES.get(venue_id, venue_id)
    log(f"  [OPENREVIEW] Searching '{keyword}' @ {venue_name}...")

    try:
        client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

        try:
            notes = client.get_all_notes(invitation=f'{venue_id}/-/Submission')
        except Exception:
            notes = client.get_all_notes(content={"venueid": venue_id})

        log(f"  [OPENREVIEW] '{keyword}' @ {venue_name}: {len(notes)} submissions to scan")

        keyword_lower = keyword.lower()
        matched_papers = []

        ACCEPTED_STATUSES = ['accepted', 'oral', 'poster', 'spotlight', 'accept']
        REJECTED_STATUSES = ['withdrawn', 'rejected', 'desk rejected', 'submitted to']

        for note in notes:
            try:
                content = note.content or {}

                title = content.get("title", {})
                if isinstance(title, dict):
                    title = title.get("value", "")

                abstract = content.get("abstract", {})
                if isinstance(abstract, dict):
                    abstract = abstract.get("value", "")

                # Keyword check
                if keyword_lower not in title.lower() and keyword_lower not in abstract.lower():
                    continue

                # Accepted check
                venue_field = content.get("venue", {})
                if isinstance(venue_field, dict):
                    venue_value = venue_field.get("value", "")
                else:
                    venue_value = str(venue_field) if venue_field else ""

                venue_lower = venue_value.lower()

                if any(status in venue_lower for status in REJECTED_STATUSES):
                    continue

                if not any(status in venue_lower for status in ACCEPTED_STATUSES):
                    continue

                # PDF check
                pdf_field = content.get("pdf", {})
                if isinstance(pdf_field, dict):
                    pdf_value = pdf_field.get("value", "")
                else:
                    pdf_value = pdf_field

                if not pdf_value:
                    continue

                # GitHub check
                github_url = extract_github_from_content(content, abstract)
                if not github_url:
                    continue

                pdf_url = f"https://openreview.net{pdf_value}" if pdf_value.startswith('/') else f"https://openreview.net/pdf?id={note.id}"

                paper = {
                    "id": note.id,
                    "title": title,
                    "abstract": abstract,
                    "pdf_url": pdf_url,
                    "github_urls": [github_url],
                    "venue": OPENREVIEW_VENUES.get(venue_id, venue_id),
                    "venue_id": venue_id,
                    "year": int(venue_id.split("/")[1][:4]) if "/" in venue_id else None,
                    "pdf_available": True,
                    "source": "openreview",
                    "keyword": keyword
                }

                matched_papers.append(paper)

            except Exception:
                continue

        return matched_papers

    except Exception as e:
        log(f"[OPENREVIEW] Error for '{keyword}' @ {venue_id}: {e}")
        return []


def search_openreview_keyword(keyword: str) -> List[Dict]:
    """Search all venues for a single keyword."""
    all_papers = []
    for venue_id in OPENREVIEW_VENUES.keys():
        papers = search_openreview_single(keyword, venue_id)
        all_papers.extend(papers)
    return all_papers


def search_openreview_domain(keywords: List[str]) -> List[Dict]:
    """Search OpenReview for all keywords."""
    log(f"\n[OPENREVIEW] {'='*50}")
    log(f"[OPENREVIEW] Searching {len(keywords)} keywords x {len(OPENREVIEW_VENUES)} venues")
    log(f"[OPENREVIEW] Venues: {list(OPENREVIEW_VENUES.values())}")
    log(f"[OPENREVIEW] {'='*50}")

    all_papers = []
    completed = 0
    total_keywords = len(keywords)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(search_openreview_keyword, kw): kw for kw in keywords}
        for future in as_completed(futures):
            kw = futures[future]
            completed += 1
            try:
                papers = future.result()
                all_papers.extend(papers)
                log(f"\n[OPENREVIEW] Keyword {completed}/{total_keywords} done: '{kw}' -> {len(papers)} papers")
            except Exception as e:
                log(f"\n[OPENREVIEW] Keyword {completed}/{total_keywords} FAILED: '{kw}' -> {e}")

    log(f"\n[OPENREVIEW] {'='*50}")
    log(f"[OPENREVIEW] Total papers before dedup: {len(all_papers)}")

    # Deduplicate by paper ID
    unique_papers = {}
    for paper in all_papers:
        paper_id = paper.get("id")
        if paper_id and paper_id not in unique_papers:
            unique_papers[paper_id] = paper

    candidates = list(unique_papers.values())
    log(f"[OPENREVIEW] After dedup: {len(candidates)} unique candidates")
    log(f"[OPENREVIEW] {'='*50}")

    return candidates


# =============================================================================
# PDF Download Functions
# =============================================================================

def download_pdf(paper: Dict, output_dir: Path, index: int, total: int) -> Optional[str]:
    """Download PDF for a single paper. Returns saved path or None."""
    pdf_url = paper.get("pdf_url", "")
    title = paper.get("title", f"paper_{index}")[:50]
    source = paper.get("source", "unknown")

    if not pdf_url:
        log(f"  [{index+1}/{total}] SKIP (no URL): {title}...")
        return None

    try:
        resp = requests.get(pdf_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and (resp.content[:4] == b'%PDF' or len(resp.content) > 5000):
            safe_title = re.sub(r'[^\w\s\-]', '', paper.get("title", f"paper_{index}"))
            safe_title = re.sub(r'\s+', '_', safe_title.strip())[:60]

            pdf_path = output_dir / f"{source}_{index:04d}_{safe_title}.pdf"

            with open(pdf_path, 'wb') as f:
                f.write(resp.content)

            size_kb = len(resp.content) // 1024
            log(f"  [{index+1}/{total}] DOWNLOADED ({size_kb}KB): {title}...")
            return str(pdf_path)
        else:
            log(f"  [{index+1}/{total}] FAILED (status={resp.status_code}): {title}...")
    except Exception as e:
        log(f"  [{index+1}/{total}] ERROR ({e}): {title}...")

    return None


def download_papers_for_domain(papers: List[Dict], output_dir: Path, max_papers: int = MAX_PAPERS_PER_DOMAIN) -> Dict:
    """Download PDFs for a domain. Returns stats."""
    # Create output directory before downloading
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        log(f"[DOWNLOAD] Created directory: {output_dir}")
    else:
        log(f"[DOWNLOAD] Directory exists: {output_dir}")

    stats = {"downloaded": 0, "failed": 0, "papers": []}

    # Limit to max_papers
    papers_to_download = papers[:max_papers]
    total = len(papers_to_download)

    log(f"\n[DOWNLOAD] Starting download of {total} papers to {output_dir}...")
    log(f"[DOWNLOAD] {'='*50}")

    for i, paper in enumerate(papers_to_download):
        pdf_path = download_pdf(paper, output_dir, i, total)

        if pdf_path:
            paper["pdf_path"] = pdf_path
            stats["papers"].append(paper)
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1

        # Rate limit
        time.sleep(0.3)

    log(f"[DOWNLOAD] {'='*50}")
    log(f"[DOWNLOAD] COMPLETE: {stats['downloaded']} downloaded, {stats['failed']} failed")
    return stats


# =============================================================================
# Main Functions
# =============================================================================

def process_domain(domain_key: str, config: DomainConfig) -> Dict:
    """Process a single domain: search + download."""
    log(f"\n{'#'*70}")
    log(f"#  DOMAIN: {config.name}")
    log(f"#  Folder: {config.folder_name}")
    log(f"#  Keywords: {len(config.keywords)}")
    log(f"{'#'*70}")

    for i, kw in enumerate(config.keywords):
        log(f"  [{i+1}] {kw}")

    output_dir = BASE_DIR / config.folder_name

    # Search PMC
    log(f"\n{'*'*60}")
    log(f"* STEP 1: PMC Search")
    log(f"{'*'*60}")
    pmc_papers = search_pmc_domain(config.keywords)

    # Search OpenReview
    log(f"\n{'*'*60}")
    log(f"* STEP 2: OpenReview Search")
    log(f"{'*'*60}")
    openreview_papers = search_openreview_domain(config.keywords)

    # Combine and deduplicate
    log(f"\n{'*'*60}")
    log(f"* STEP 3: Combine & Deduplicate")
    log(f"{'*'*60}")

    all_papers = pmc_papers + openreview_papers
    log(f"[COMBINE] PMC: {len(pmc_papers)} + OpenReview: {len(openreview_papers)} = {len(all_papers)} total")

    # Deduplicate by title
    unique_papers = []
    seen_titles = set()
    for paper in all_papers:
        title_key = paper.get("title", "").lower().strip()[:100]
        if title_key and title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_papers.append(paper)

    log(f"[COMBINE] After title dedup: {len(unique_papers)} unique papers")

    # Save all candidates JSON (before download)
    log(f"\n{'*'*60}")
    log(f"* STEP 4: Save All Candidates JSON")
    log(f"{'*'*60}")

    # Create output directory first
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[DIR] Created: {output_dir}")

    # Save PMC candidates
    pmc_json_path = output_dir / "pmc_candidates.json"
    pmc_data = {
        "source": "pmc",
        "domain": config.name,
        "count": len(pmc_papers),
        "papers": pmc_papers,
        "timestamp": datetime.now().isoformat()
    }
    with open(pmc_json_path, 'w', encoding='utf-8') as f:
        json.dump(pmc_data, f, indent=2, ensure_ascii=False)
    log(f"[JSON] PMC candidates saved: {pmc_json_path} ({len(pmc_papers)} papers)")

    # Save OpenReview candidates
    openreview_json_path = output_dir / "openreview_candidates.json"
    openreview_data = {
        "source": "openreview",
        "domain": config.name,
        "count": len(openreview_papers),
        "papers": openreview_papers,
        "timestamp": datetime.now().isoformat()
    }
    with open(openreview_json_path, 'w', encoding='utf-8') as f:
        json.dump(openreview_data, f, indent=2, ensure_ascii=False)
    log(f"[JSON] OpenReview candidates saved: {openreview_json_path} ({len(openreview_papers)} papers)")

    # Save all unique candidates (combined)
    all_candidates_path = output_dir / "all_candidates.json"
    all_candidates_data = {
        "domain": config.name,
        "total_count": len(unique_papers),
        "pmc_count": len(pmc_papers),
        "openreview_count": len(openreview_papers),
        "papers": unique_papers,
        "timestamp": datetime.now().isoformat()
    }
    with open(all_candidates_path, 'w', encoding='utf-8') as f:
        json.dump(all_candidates_data, f, indent=2, ensure_ascii=False)
    log(f"[JSON] All candidates saved: {all_candidates_path} ({len(unique_papers)} papers)")

    # Download PDFs
    log(f"\n{'*'*60}")
    log(f"* STEP 5: Download PDFs")
    log(f"{'*'*60}")
    download_stats = download_papers_for_domain(unique_papers, output_dir)

    # Save downloaded papers metadata
    metadata = {
        "domain": config.name,
        "keywords": config.keywords,
        "search_stats": {
            "pmc_found": len(pmc_papers),
            "openreview_found": len(openreview_papers),
            "total_unique": len(unique_papers),
            "downloaded": download_stats["downloaded"],
            "failed": download_stats["failed"],
        },
        "downloaded_papers": download_stats["papers"],
        "timestamp": datetime.now().isoformat()
    }

    metadata_path = output_dir / "downloaded_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log(f"\n[JSON] Download metadata saved: {metadata_path}")

    log(f"\n{'#'*70}")
    log(f"#  DOMAIN COMPLETE: {config.name}")
    log(f"#  Downloaded: {download_stats['downloaded']} papers")
    log(f"#  Failed: {download_stats['failed']} papers")
    log(f"{'#'*70}")

    return metadata


def main():
    """Main entry point."""
    # Create all directories immediately
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[INIT] Created base directory: {BASE_DIR}")
    for config in DOMAINS.values():
        domain_dir = BASE_DIR / config.folder_name
        domain_dir.mkdir(parents=True, exist_ok=True)
        log(f"[INIT] Created domain directory: {domain_dir}")

    # Unbuffered output for real-time logging
    log("="*70)
    log("  PAPER DOWNLOAD SCRIPT FOR 4 DOMAINS")
    log("="*70)
    log(f"  Output directory: {BASE_DIR}")
    log(f"  Max papers per domain: {MAX_PAPERS_PER_DOMAIN}")
    log(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*70)

    log("\nDomains to process:")
    for i, (key, config) in enumerate(DOMAINS.items()):
        log(f"  [{i+1}] {config.name} ({config.folder_name})")

    results = {}
    total_domains = len(DOMAINS)

    for i, (domain_key, config) in enumerate(DOMAINS.items()):
        log(f"\n\n{'@'*70}")
        log(f"@  Processing domain {i+1}/{total_domains}: {domain_key}")
        log(f"{'@'*70}")

        try:
            results[domain_key] = process_domain(domain_key, config)
        except Exception as e:
            log(f"[ERROR] Failed to process {domain_key}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    summary = {
        "total_domains": len(results),
        "domains": {k: v["search_stats"] for k, v in results.items()},
        "timestamp": datetime.now().isoformat()
    }

    summary_path = BASE_DIR / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log("\n\n" + "="*70)
    log("  FINAL SUMMARY")
    log("="*70)
    total_downloaded = 0
    for domain_key, stats in summary["domains"].items():
        downloaded = stats['downloaded']
        total_downloaded += downloaded
        log(f"  {domain_key}: {downloaded} papers downloaded")
    log("-"*70)
    log(f"  TOTAL: {total_downloaded} papers downloaded")
    log(f"\n  Summary saved to: {summary_path}")
    log(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*70)


if __name__ == "__main__":
    main()
