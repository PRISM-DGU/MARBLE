"""Visualizer Agent for MARBLE Research Subgraph.

This agent creates publication-quality visualizations from baseline/refined execution results
using DRP-VIS MCP tools.
"""

from typing import Dict
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from agent_workflow.state import MARBLEState
from agent_workflow.logger import logger
from ..prompts.analysis_prompts import VISUALIZER_SYSTEM_PROMPT

class VisualizerAgent(BaseAgentNode):
    """Creates visualizations from execution results using DRP-VIS MCP."""

    def __init__(self):
        super().__init__("visualizer")
    
    def get_prompt(self):
        return VISUALIZER_SYSTEM_PROMPT


    def update_domain_state(self, state: MARBLEState, agent_response: str) -> Dict:
        """Extract visualization results from agent response."""
        viz_results = self.extract_json_from_response(agent_response)

        if viz_results and viz_results.get("status") == "success":
            generated = viz_results.get("generated_files", [])
            pairwise = viz_results.get("pairwise", []) or []
            logger.info(f"📊 [VISUALIZER] Generated {len(generated)} files; pairwise plots: {len(pairwise)}")

            return {
                "visualization_results": viz_results,
                "processing_logs": state.get("processing_logs", []) + [
                    f"[VISUALIZER] Generated {len(generated)} visualization files",
                    f"[VISUALIZER] Pairwise plots: {len(pairwise)}",
                    "[VISUALIZER] Visualization complete"
                ]
            }

        return {
            "visualization_results": viz_results or {"status": "failed"},
            "processing_logs": state.get("processing_logs", []) + [
                "[VISUALIZER] Visualization failed or incomplete"
            ]
        }