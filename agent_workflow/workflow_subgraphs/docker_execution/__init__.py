"""Docker Execution Subgraph.

Runs Docker-based testing for build workflow outputs.
"""

from .graph import get_docker_execution_subgraph

__all__ = ["get_docker_execution_subgraph"]
