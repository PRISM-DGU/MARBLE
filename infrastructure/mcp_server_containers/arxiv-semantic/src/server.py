#!/usr/bin/env python
"""
Enhanced ArXiv MCP Server with Semantic Search
Combines ArXiv paper search/download with semantic search and deep analysis capabilities.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from fastmcp import FastMCP
from fastmcp import Context

from semantic_search import SemanticSearchEngine
from paper_analyzer import PaperAnalyzer
from embeddings import EmbeddingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="arxiv-semantic",
    version="1.0.0"
)

# Runtime component cache keyed by resolved context
_RUNTIME_CACHE: Dict[str, Dict[str, Any]] = {}


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.abspath(path)


def _ensure_directory(path: Optional[str]):
    if path:
        os.makedirs(path, exist_ok=True)


def _setup_cache_environment(cache_root: Optional[str]) -> None:
    if not cache_root:
        return

    cache_root = _normalize_path(cache_root)
    if not cache_root:
        return

    os.environ['HF_HOME'] = cache_root
    hub_dir = os.path.join(cache_root, 'hub')
    transformers_dir = os.path.join(cache_root, 'transformers')
    os.environ['HF_HUB_CACHE'] = hub_dir
    os.environ['TRANSFORMERS_CACHE'] = transformers_dir

    _ensure_directory(cache_root)
    _ensure_directory(hub_dir)
    _ensure_directory(transformers_dir)


def _resolve_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ctx = context.copy() if context else {}

    resolved = {
        'model': ctx.get('model'),
        'papers_dir': _normalize_path(ctx.get('papers_dir') or os.environ.get('PAPERS_DIR', '/workspace/papers')),
        'origin_dir': _normalize_path(ctx.get('origin_dir') or os.environ.get('PAPERS_ORIGIN_DIR')),
        'chromadb_path': _normalize_path(ctx.get('chromadb_path') or os.environ.get('CHROMA_DB_PATH', '/workspace/chromadb')),
        'embeddings_dir': _normalize_path(ctx.get('embeddings_dir') or os.environ.get('EMBEDDINGS_DIR', '/workspace/embeddings')),
        'cache_dir': _normalize_path(ctx.get('cache_dir') or os.environ.get('HF_HOME', '/workspace/.cache/huggingface'))
    }

    return resolved


def _get_runtime_components(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    resolved = _resolve_context(context)
    cache_key = json.dumps(resolved, sort_keys=True)

    if cache_key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[cache_key]

    _ensure_directory(resolved['papers_dir'])
    _ensure_directory(resolved['chromadb_path'])
    _ensure_directory(resolved['embeddings_dir'])
    _setup_cache_environment(resolved['cache_dir'])

    if resolved['embeddings_dir']:
        os.environ['EMBEDDINGS_DIR'] = resolved['embeddings_dir']

    embedding_mgr = EmbeddingManager()
    search_eng = SemanticSearchEngine(db_path=resolved['chromadb_path'], embedding_manager=embedding_mgr)
    paper_mgr = PaperAnalyzer(resolved['papers_dir'])

    runtime = {
        'context': resolved,
        'embedding_manager': embedding_mgr,
        'search_engine': search_eng,
        'paper_analyzer': paper_mgr
    }

    _RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _resolve_pdf_path(paper_path: str, context: Dict[str, Any]) -> str:
    candidates = []
    if os.path.isabs(paper_path):
        candidates.append(paper_path)
    else:
        for base in [context.get('origin_dir'), context.get('papers_dir'), os.getcwd()]:
            if base:
                candidates.append(os.path.join(base, paper_path))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)

    # Fallback to provided path
    return os.path.abspath(paper_path)


def _finalize_analysis(
    runtime: Dict[str, Any],
    paper_id: str,
    analysis: Dict[str, Any],
    extract_figures: bool,
    extract_equations: bool,
    extract_references: bool,
    index_for_search: bool,
    target_sections: Optional[List[str]] = None
) -> Dict[str, Any]:
    if not extract_figures:
        analysis.pop('figures', None)
    if not extract_equations:
        analysis.pop('equations', None)
    if not extract_references:
        analysis.pop('references', None)
    
    # Filter sections if target_sections is provided
    if target_sections:
        all_sections = analysis.get('sections', {})
        filtered_sections = {
            section_name: section_content
            for section_name, section_content in all_sections.items()
            if any(target.lower() in section_name.lower() for target in target_sections)
        }
        analysis['sections'] = filtered_sections

    if index_for_search and analysis.get('full_text'):
        search_engine = runtime['search_engine']
        embedding_manager = runtime['embedding_manager']

        chunks = embedding_manager.chunk_text(
            analysis['full_text'],
            chunk_size=512,
            overlap=50
        )

        metadata = {
            'authors': analysis.get('authors', []),
            'published': analysis.get('published'),
            'categories': analysis.get('categories', []),
            'model': runtime['context'].get('model')
        }

        # Remove keys whose values are None to satisfy Chroma metadata requirements
        metadata = {key: value for key, value in metadata.items() if value is not None}

        search_engine.add_paper(
            paper_id=paper_id,
            title=analysis.get('title') or paper_id,
            abstract=analysis.get('abstract') or '',
            content_chunks=chunks,
            metadata=metadata
        )

        analysis['indexed'] = True
        analysis['num_chunks_indexed'] = len(chunks)
    else:
        analysis['indexed'] = False

    analysis.pop('full_text', None)

    return {
        'success': True,
        'paper_id': paper_id,
        'analysis': analysis
    }


# ============================================================================
# Tool Definitions
# ============================================================================


@mcp.tool()
async def search_arxiv(
    query: str,
    max_results: int = 10,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Search ArXiv for papers matching the query.
    Returns paper metadata including title, authors, abstract, and ArXiv ID.

    Args:
        query: Search query for ArXiv papers
        max_results: Maximum number of results to return
    """
    try:
        import arxiv

        logger.info(f"Searching ArXiv for: {query}")

        # Search ArXiv
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in search.results():
            results.append({
                'paper_id': paper.entry_id.split('/')[-1],  # Extract ID from URL
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'abstract': paper.summary,
                'published': paper.published.isoformat(),
                'updated': paper.updated.isoformat() if paper.updated else None,
                'categories': paper.categories,
                'pdf_url': paper.pdf_url,
                'comment': paper.comment
            })

        logger.info(f"Found {len(results)} papers")
        return {
            'success': True,
            'query': query,
            'num_results': len(results),
            'papers': results
        }

    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def semantic_search(
    query: str,
    n_results: int = 10,
    filter_categories: Optional[List[str]] = None,
    filter_year: Optional[int] = None,
    context: Optional[Dict[str, Any]] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Perform semantic search across indexed papers.
    Uses embeddings to find papers semantically similar to the query.

    Args:
        query: Semantic search query
        n_results: Number of results to return
        filter_categories: Filter by ArXiv categories
        filter_year: Filter by publication year
        context: Optional runtime context overrides (directories, model)
    """
    try:
        logger.info(f"Performing semantic search: {query}")

        # Build metadata filter
        filter_metadata = {}
        if filter_categories:
            filter_metadata['categories'] = {'$in': filter_categories}
        if filter_year:
            filter_metadata['year'] = filter_year

        runtime = _get_runtime_components(context)
        search_engine = runtime['search_engine']

        # Perform semantic search
        results = search_engine.search(
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata if filter_metadata else None
        )

        # Group results by paper
        papers_dict = {}
        for result in results:
            paper_id = result['metadata'].get('paper_id')
            if paper_id not in papers_dict:
                papers_dict[paper_id] = {
                    'paper_id': paper_id,
                    'title': result['metadata'].get('title'),
                    'best_score': result['score'],
                    'relevant_chunks': []
                }

            papers_dict[paper_id]['relevant_chunks'].append({
                'text': result['document'][:500],  # First 500 chars
                'score': result['score'],
                'type': result['metadata'].get('type')
            })

        # Convert to list and sort by best score
        papers = list(papers_dict.values())
        papers.sort(key=lambda x: x['best_score'], reverse=True)

        logger.info(f"Found {len(papers)} relevant papers")
        return {
            'success': True,
            'query': query,
            'num_papers': len(papers),
            'papers': papers,
            'index_stats': search_engine.get_statistics()
        }

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def analyze_paper(
    paper_id: str,
    extract_figures: bool = True,
    extract_equations: bool = True,
    extract_references: bool = True,
    index_for_search: bool = True,
    context: Optional[Dict[str, Any]] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Perform deep analysis of an ArXiv paper.
    Extracts sections, figures, equations, references, and key findings.
    Optionally indexes the paper for semantic search.

    Args:
        paper_id: ArXiv paper ID to analyze
        extract_figures: Extract figures from paper
        extract_equations: Extract equations from paper
        extract_references: Extract references from paper
        index_for_search: Index paper for semantic search
        context: Optional runtime context overrides (directories, model)
    """
    try:
        logger.info(f"Analyzing paper: {paper_id}")
        runtime = _get_runtime_components(context)
        analyzer = runtime['paper_analyzer']

        analysis = analyzer.analyze_paper(paper_id)

        logger.info(f"Analysis complete for paper {paper_id}")
        return _finalize_analysis(
            runtime,
            paper_id,
            analysis,
            extract_figures=extract_figures,
            extract_equations=extract_equations,
            extract_references=extract_references,
            index_for_search=index_for_search,
            target_sections=None
        )

    except Exception as e:
        logger.error(f"Paper analysis failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def analyze_local_paper(
    paper_id: str,
    paper_path: str,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    abstract: Optional[str] = None,
    published: Optional[str] = None,
    categories: Optional[List[str]] = None,
    extract_figures: bool = True,
    extract_equations: bool = True,
    extract_references: bool = True,
    index_for_search: bool = True,
    target_sections: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Analyze a locally stored PDF without downloading from ArXiv.

    Args:
        paper_id: Identifier to use for the paper
        paper_path: Path to the local PDF (absolute or relative)
        title: Paper title for metadata
        authors: List of authors
        abstract: Abstract text if available
        published: Publication date
        categories: Paper categories/tags
        extract_figures: Extract figures from paper
        extract_equations: Extract equations from paper
        extract_references: Extract references from paper
        index_for_search: Index paper for semantic search
        target_sections: Specific sections to include in analysis (e.g., ['Methods', 'Results']). If None, returns all sections.
        context: Optional runtime context overrides (directories, model)
    """
    try:
        logger.info(f"Analyzing local paper: {paper_id}")

        runtime = _get_runtime_components(context)
        analyzer = runtime['paper_analyzer']

        pdf_path = _resolve_pdf_path(paper_path, runtime['context'])
        metadata = {
            'title': title,
            'authors': authors or [],
            'abstract': abstract,
            'published': published,
            'categories': categories or []
        }

        analysis = analyzer.analyze_pdf(paper_id, pdf_path, metadata)
        analysis['source'] = 'local'

        return _finalize_analysis(
            runtime,
            paper_id,
            analysis,
            extract_figures=extract_figures,
            extract_equations=extract_equations,
            extract_references=extract_references,
            index_for_search=index_for_search,
            target_sections=target_sections
        )

    except Exception as e:
        logger.error(f"Local paper analysis failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def find_similar_papers(
    paper_id: str,
    n_results: int = 10,
    context: Optional[Dict[str, Any]] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Find papers similar to a given paper based on semantic similarity.
    The paper must be indexed first using analyze_paper.

    Args:
        paper_id: ArXiv paper ID to find similar papers for
        n_results: Number of similar papers to return
        context: Optional runtime context overrides (directories, model)
    """
    try:
        logger.info(f"Finding papers similar to: {paper_id}")

        runtime = _get_runtime_components(context)
        search_engine = runtime['search_engine']

        similar_papers = search_engine.search_similar_papers(
            paper_id=paper_id,
            n_results=n_results
        )

        logger.info(f"Found {len(similar_papers)} similar papers")
        return {
            'success': True,
            'source_paper_id': paper_id,
            'num_results': len(similar_papers),
            'similar_papers': similar_papers
        }

    except Exception as e:
        logger.error(f"Similar paper search failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def extract_section(
    paper_id: str,
    section_name: str,
    context: Optional[Dict[str, Any]] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Extract a specific section from a paper.
    Common sections: Abstract, Introduction, Methods, Results, Discussion, Conclusion.

    Args:
        paper_id: ArXiv paper ID
        section_name: Name of section to extract (e.g., 'Methods', 'Results')
        context: Optional runtime context overrides (directories, model)
    """
    try:
        logger.info(f"Extracting section '{section_name}' from paper {paper_id}")

        runtime = _get_runtime_components(context)
        analyzer = runtime['paper_analyzer']

        # Download and extract paper text
        paper, pdf_path = analyzer.download_paper(paper_id)
        full_text = analyzer.extract_full_text(pdf_path)

        # Extract sections
        sections = analyzer.extract_sections(full_text)

        # Find matching section
        matching_section = None
        matching_key = None

        for key, content in sections.items():
            if section_name.lower() in key.lower():
                matching_section = content
                matching_key = key
                break

        if matching_section:
            logger.info(f"Found section: {matching_key}")
            return {
                'success': True,
                'paper_id': paper_id,
                'section_name': matching_key,
                'content': matching_section,
                'available_sections': list(sections.keys())
            }
        else:
            return {
                'success': False,
                'paper_id': paper_id,
                'error': f"Section '{section_name}' not found",
                'available_sections': list(sections.keys())
            }

    except Exception as e:
        logger.error(f"Section extraction failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def search_paper_content(
    paper_id: str,
    query: str,
    n_results: int = 5,
    context: Optional[Dict[str, Any]] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Search within a specific paper's content using semantic search.
    The paper must be indexed first using analyze_paper.

    Args:
        paper_id: ArXiv paper ID to search within
        query: Search query
        n_results: Number of relevant chunks to return
        context: Optional runtime context overrides (directories, model)
    """
    try:
        logger.info(f"Searching within paper {paper_id} for: {query}")

        runtime = _get_runtime_components(context)
        search_engine = runtime['search_engine']
        embedding_manager = runtime['embedding_manager']

        # Get all chunks for the paper
        all_chunks = search_engine.get_paper_chunks(
            paper_id=paper_id,
            chunk_type='content'
        )

        if not all_chunks:
            return {
                'success': False,
                'error': f"Paper {paper_id} not indexed. Use analyze_paper first."
            }

        # Generate query embedding
        query_embedding = embedding_manager.generate_query_embedding(query)

        # Get chunk texts and generate embeddings if needed
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        chunk_embeddings = embedding_manager.generate_embeddings(
            chunk_texts,
            show_progress=False
        )

        # Compute similarities
        similarities = embedding_manager.compute_similarity(
            query_embedding,
            chunk_embeddings
        )

        # Get top results
        top_indices = similarities.argsort()[-n_results:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Similarity threshold
                results.append({
                    'chunk_id': all_chunks[idx]['metadata'].get('chunk_id'),
                    'text': all_chunks[idx]['text'][:500],  # First 500 chars
                    'score': float(similarities[idx]),
                    'position': {
                        'start': all_chunks[idx]['metadata'].get('chunk_start'),
                        'end': all_chunks[idx]['metadata'].get('chunk_end')
                    }
                })

        logger.info(f"Found {len(results)} relevant chunks in paper")
        return {
            'success': True,
            'paper_id': paper_id,
            'query': query,
            'num_results': len(results),
            'results': results
        }

    except Exception as e:
        logger.error(f"Paper content search failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def get_index_statistics(ctx: Context) -> Dict[str, Any]:
    """
    Get statistics about the semantic search index.
    Shows number of indexed papers, chunks, and storage information.
    """
    try:
        runtime = _get_runtime_components(None)
        stats = runtime['search_engine'].get_statistics()
        return {
            'success': True,
            'statistics': stats
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@mcp.tool()
async def clear_index(ctx: Context) -> Dict[str, Any]:
    """
    Clear the entire semantic search index.
    WARNING: This will remove all indexed papers and cannot be undone.
    """
    try:
        logger.warning("Clearing semantic search index")

        # Get current statistics before clearing
        runtime = _get_runtime_components(None)
        search_engine = runtime['search_engine']

        stats_before = search_engine.get_statistics()

        # Clear the collection
        search_engine.client.delete_collection("arxiv_papers")

        # Reinitialize for continued usage
        search_engine._initialize_db()

        stats_after = search_engine.get_statistics()

        return {
            'success': True,
            'message': 'Index cleared successfully',
            'stats_before': stats_before,
            'stats_after': stats_after
        }
    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        return {
            'success': False,
            'error': str(e)
        }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the server."""
    try:
        logger.info("Starting Enhanced ArXiv MCP Server with Semantic Search")
        logger.info(f"Papers directory: {os.environ.get('PAPERS_DIR', '/workspace/papers')}")
        logger.info(f"ChromaDB path: {os.environ.get('CHROMA_DB_PATH', '/workspace/chromadb')}")

        # Run the server
        mcp.run(transport="stdio")

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
