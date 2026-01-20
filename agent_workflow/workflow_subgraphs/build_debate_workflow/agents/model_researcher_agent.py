"""Model Researcher Agent for build_debate workflow.

Supports multiple modes:
- read_summary: Read target model paper and generate summary
- rank_proposals: Rank improvement proposals based on critique
- final_ranking: Make final proposal selection after critic feedback

Supports dynamic prompts via PromptFactory for different bioinformatics models.
"""

from typing import Any, Dict, List, Optional
from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from ..prompts.build_debate_prompts import (
    PAPER_READER_DEEPTTA_PROMPT,
    MODEL_RESEARCHER_RANKING_PROMPT,
    MODEL_RESEARCHER_FINAL_RANKING_PROMPT,
)
from ..prompts.prompt_factory import PromptFactory
from .common_tools import (
    create_read_file_tool,
    create_write_file_tool,
    create_read_pdf_tool,
)


class ModelResearcherAgent(BaseAgentNode):
    """Agent that reads target model paper and performs proposal ranking.

    Modes:
    - read_summary: Read target model paper and generate summary
    - rank_proposals: Rank improvement proposals
    - final_ranking: Make final proposal selection

    Uses PromptFactory for dynamic prompt generation based on model configuration.
    """

    def __init__(
        self,
        mode: str = "read_summary",
        # For read_summary mode
        pdf_path: str = "",
        output_path: str = "",
        # For rank_proposals mode
        critique_path: str = "",
        proposal1_path: str = "",
        proposal2_path: str = "",
        target_summary_path: str = "",
        # For final_ranking mode
        critique_ranked_path: str = "",
        ranking_r1_path: str = "",
        # Model configuration for dynamic prompts
        model_config: Optional[Dict[str, Any]] = None,
        # Iteration context for prompt injection
        iteration_context: str = "",
        checkpointer=None
    ):
        """Initialize Model Researcher Agent.

        Args:
            mode: Operation mode - "read_summary", "rank_proposals", "final_ranking"
            pdf_path: Path to the target model PDF file (read_summary mode)
            output_path: Path where output will be saved
            critique_path: Path to critique of proposals (rank_proposals mode)
            proposal1_path: Path to first proposal (ranking modes)
            proposal2_path: Path to second proposal (ranking modes)
            target_summary_path: Path to target model summary (rank_proposals mode)
            critique_ranked_path: Path to critique of ranked proposals (final_ranking mode)
            ranking_r1_path: Path to initial ranking (final_ranking mode)
            model_config: Model configuration dict for dynamic prompts (from MODEL_WORKFLOW_CONFIG)
            checkpointer: Optional LangGraph checkpointer
        """
        super().__init__(f"model_researcher_{mode}", checkpointer=checkpointer)
        self.mode = mode

        # read_summary mode parameters
        self.pdf_path = pdf_path
        self.output_path = output_path

        # rank_proposals mode parameters
        self.critique_path = critique_path
        self.proposal1_path = proposal1_path
        self.proposal2_path = proposal2_path
        self.target_summary_path = target_summary_path

        # final_ranking mode parameters
        self.critique_ranked_path = critique_ranked_path
        self.ranking_r1_path = ranking_r1_path

        # Model configuration for dynamic prompts
        self.model_config = model_config
        self.iteration_context = iteration_context
        self.prompt_factory = PromptFactory(model_config, iteration_context=iteration_context) if model_config else None

        logger.debug(f"[ModelResearcherAgent] Initialized with mode={mode}, model_config={'provided' if model_config else 'None'}")

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
        if self.mode == "read_summary":
            return self.prompt_factory.get_paper_reader_prompt(
                pdf_path=self.pdf_path,
                output_path=self.output_path
            )
        elif self.mode == "rank_proposals":
            return self.prompt_factory.get_ranking_prompt(
                critique_path=self.critique_path,
                proposal1_path=self.proposal1_path,
                proposal2_path=self.proposal2_path,
                target_summary_path=self.target_summary_path,
                output_path=self.output_path
            )
        elif self.mode == "final_ranking":
            return self.prompt_factory.get_final_ranking_prompt(
                critique_ranked_path=self.critique_ranked_path,
                ranking_r1_path=self.ranking_r1_path,
                proposal1_path=self.proposal1_path,
                proposal2_path=self.proposal2_path,
                output_path=self.output_path
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_legacy_prompt(self):
        """Generate prompt using legacy static templates (for backwards compatibility)."""
        if self.mode == "read_summary":
            return PAPER_READER_DEEPTTA_PROMPT.format(
                pdf_path=self.pdf_path,
                output_path=self.output_path
            )
        elif self.mode == "rank_proposals":
            return MODEL_RESEARCHER_RANKING_PROMPT.format(
                critique_path=self.critique_path,
                proposal1_path=self.proposal1_path,
                proposal2_path=self.proposal2_path,
                target_summary_path=self.target_summary_path,
                output_path=self.output_path
            )
        elif self.mode == "final_ranking":
            return MODEL_RESEARCHER_FINAL_RANKING_PROMPT.format(
                critique_ranked_path=self.critique_ranked_path,
                ranking_r1_path=self.ranking_r1_path,
                proposal1_path=self.proposal1_path,
                proposal2_path=self.proposal2_path,
                output_path=self.output_path
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_additional_tools(self):
        """Provide tools based on mode."""
        if self.mode == "read_summary":
            # PDF reading and writing for summary generation
            return [
                create_read_pdf_tool(max_length=50000),
                create_write_file_tool(),
                create_read_file_tool(max_length=10000),
            ]
        elif self.mode in ["rank_proposals", "final_ranking"]:
            # File reading and writing for ranking tasks
            return [
                create_read_file_tool(max_length=30000),
                create_write_file_tool(),
            ]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def create_agent(self):
        """Create the LLM agent with tools."""
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

        logger.info(f"[ModelResearcherAgent] Agent created with mode={self.mode}")
        return agent
