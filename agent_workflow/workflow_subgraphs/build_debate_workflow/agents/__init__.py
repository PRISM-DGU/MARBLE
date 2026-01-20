"""Build Debate Workflow Agents"""

# Paper search agents (parallel execution)
from .pmc_researcher_agent import PMCResearcherAgent
from .openreview_researcher_agent import OpenReviewResearcherAgent
from .paper_aggregator_agent import PaperAggregatorAgent

# Core agents
from .model_researcher_agent import ModelResearcherAgent
from .article_researcher_agent import ArticleResearcherAgent  # paper_summary, generate_proposal modes only
from .critic_agent import CriticAgent

# Legacy agents (for backwards compatibility and specific tasks)
from .paper_reader_agent import PaperReaderAgent
from .proposal_agent import ProposalAgent
from .validation_agent import ValidationAgent

__all__ = [
    # Paper search agents
    "PMCResearcherAgent",
    "OpenReviewResearcherAgent",
    "PaperAggregatorAgent",
    # Core agents
    "ModelResearcherAgent",
    "ArticleResearcherAgent",
    "CriticAgent",
    # Legacy agents
    "PaperReaderAgent",
    "ProposalAgent",
    "ValidationAgent",
]
