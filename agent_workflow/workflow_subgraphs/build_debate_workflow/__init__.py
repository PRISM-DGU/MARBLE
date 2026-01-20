"""Build Debate Workflow - Paper-based Architecture Design

This workflow reads the target model PDF, searches for relevant papers, and generates
architecture improvement proposals through multi-agent debate.

Workflow:
1. Paper Reader (Target): Read target model PDF, extract methods, write summary
2. Model Problem Agent: Identify model's limitations
3. Article Researcher: Search papers via PMC, filter by GitHub+PDF, download 2 PDFs
4. Paper Readers (x2): Read downloaded PDFs, write summaries
5. Proposal Generation: Generate proposals based on paper summaries
6. Critique & Ranking: Evaluate and rank proposals
7. Proposal Agent: Synthesize into implementation plan

Output: Proposal MD file for build_development_workflow
"""

from .graph import create_build_debate_subgraph, get_build_debate_subgraph

__all__ = [
    "create_build_debate_subgraph",
    "get_build_debate_subgraph",
]
