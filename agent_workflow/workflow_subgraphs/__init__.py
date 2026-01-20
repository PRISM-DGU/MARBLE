"""Workflow Subgraphs Package.

This package contains all workflow subgraphs for MARBLE:
- analysis_workflow: Result analysis and visualization
"""

from .analysis_workflow.graph import get_analysis_subgraph

__all__ = [
    "get_analysis_subgraph",
]
