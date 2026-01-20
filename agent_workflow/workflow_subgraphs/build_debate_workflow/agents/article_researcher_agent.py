"""Article Researcher Agent for build_debate workflow.

Supports multiple modes:
- paper_summary: Read paper and generate summary
- generate_proposal: Generate improvement proposal based on weakness analysis

NOTE: PMC paper search has been moved to PMCResearcherAgent.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from ..prompts.build_debate_prompts import (
    PAPER_READER_OTHER_PROMPT,
    ARTICLE_RESEARCHER_PROPOSAL_PROMPT,
)
from ..prompts.prompt_factory import PromptFactory
from .common_tools import (
    create_read_file_tool,
    create_write_file_tool,
    create_read_pdf_tool,
    create_extract_github_urls_tool,
    create_clone_github_repo_tool,
    create_read_github_file_tool,
    create_list_repo_structure_tool,
)


class ArticleResearcherAgent(BaseAgentNode):
    """Agent that handles article research tasks based on mode.

    Modes:
    - paper_summary: Read PDF and generate summary
    - generate_proposal: Generate improvement proposal

    NOTE: For PMC paper search, use PMCResearcherAgent instead.

    Uses PromptFactory for dynamic prompt generation based on model configuration.
    """

    def __init__(
        self,
        mode: str = "paper_summary",
        # Common
        output_path: str = "",
        # For paper_summary mode
        paper_name: str = "",
        pdf_path: str = "",
        repos_dir: str = "",
        # For generate_proposal mode
        weakness_path: str = "",
        paper_summary_path: str = "",
        target_summary_path: str = "",
        # Model configuration for dynamic prompts
        model_config: Optional[Dict[str, Any]] = None,
        # Iteration context for prompt injection
        iteration_context: str = "",
        checkpointer=None
    ):
        """Initialize Article Researcher Agent.

        Args:
            mode: Operation mode - "paper_summary", "generate_proposal"
            output_path: Output file path
            paper_name: Name of paper being read (paper_summary mode)
            pdf_path: Path to PDF file (paper_summary mode)
            repos_dir: Directory for cloned repositories
            weakness_path: Path to weakness analysis (generate_proposal mode)
            paper_summary_path: Path to reference paper summary (generate_proposal mode)
            target_summary_path: Path to target model summary (generate_proposal mode)
            model_config: Model configuration dict for dynamic prompts (from MODEL_WORKFLOW_CONFIG)
            checkpointer: Optional LangGraph checkpointer
        """
        super().__init__(f"article_researcher_{mode}", checkpointer=checkpointer)
        self.mode = mode

        # Common parameters
        self.output_path = output_path

        # Paper summary mode parameters
        self.paper_name = paper_name
        self.pdf_path = pdf_path
        self.repos_dir = repos_dir

        # Generate proposal mode parameters
        self.weakness_path = weakness_path
        self.paper_summary_path = paper_summary_path
        self.target_summary_path = target_summary_path

        # Model configuration for dynamic prompts
        self.model_config = model_config
        self.iteration_context = iteration_context
        self.prompt_factory = PromptFactory(model_config, iteration_context=iteration_context) if model_config else None

        logger.debug(f"[ArticleResearcherAgent] Initialized with mode={mode}")

    def get_prompt(self):
        """Return system prompt based on mode.

        Uses PromptFactory for dynamic prompts if model_config is provided,
        otherwise falls back to legacy static prompts.
        """
        if self.prompt_factory:
            return self._get_dynamic_prompt()
        else:
            return self._get_legacy_prompt()

    def _get_dynamic_prompt(self):
        """Generate prompt using PromptFactory (model-aware)."""
        if self.mode == "paper_summary":
            return self.prompt_factory.get_other_paper_reader_prompt(
                pdf_path=self.pdf_path,
                output_path=self.output_path,
                repos_dir=self.repos_dir
            )
        elif self.mode == "generate_proposal":
            return self.prompt_factory.get_proposal_generation_prompt(
                paper_name=self.paper_name or "Reference Paper",
                weakness_path=self.weakness_path,
                paper_summary_path=self.paper_summary_path,
                target_summary_path=self.target_summary_path,
                output_path=self.output_path,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_legacy_prompt(self):
        """Generate prompt using legacy static templates (for backwards compatibility)."""
        if self.mode == "paper_summary":
            return PAPER_READER_OTHER_PROMPT.format(
                pdf_path=self.pdf_path,
                output_path=self.output_path,
                repos_dir=self.repos_dir
            )
        elif self.mode == "generate_proposal":
            return ARTICLE_RESEARCHER_PROPOSAL_PROMPT.format(
                weakness_path=self.weakness_path,
                paper_summary_path=self.paper_summary_path,
                target_summary_path=self.target_summary_path,
                output_path=self.output_path,
                paper_name=self.paper_name or "Reference Model"
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_additional_tools(self):
        """Return tools based on mode."""
        if self.mode == "paper_summary":
            # Tools for reading PDFs and generating summaries
            return [
                create_read_pdf_tool(max_length=50000),
                create_write_file_tool(),
                create_read_file_tool(max_length=10000),
                create_extract_github_urls_tool(),
                create_clone_github_repo_tool(),
                create_read_github_file_tool(),
                create_list_repo_structure_tool(),
            ]
        elif self.mode == "generate_proposal":
            # Tools for reading files and generating proposals
            return [
                create_read_file_tool(max_length=30000),
                create_write_file_tool(),
                create_read_github_file_tool(),
                create_list_repo_structure_tool(),
            ]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

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

        logger.info(f"[ArticleResearcherAgent] Agent created with mode={self.mode}")
        return agent
