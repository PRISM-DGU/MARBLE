from typing import Dict, List, Any, Optional
from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from ..prompts.build_debate_prompts import (
    CRITIC_ROUND1_PROMPT,
    CRITIC_ROUND2_PROMPT,
    CRITIC_WEAKNESS_PROMPT,
    CRITIC_FINAL_PROMPT,
    CRITIC_RANKED_CRITIQUE_PROMPT,
)
from ..prompts.prompt_factory import PromptFactory
from .common_tools import create_read_file_tool, create_write_file_tool


class CriticAgent(BaseAgentNode):
    def __init__(
        self,
        mode: str = "critique_proposals",
        target_summary_path: str = "",
        output_path: str = "",
        weakness_path: str = "",
        proposal_paths: List[str] = None,
        # For critique_ranked mode
        ranking_r1_path: str = "",
        proposal1_path: str = "",
        proposal2_path: str = "",
        # Model configuration for dynamic prompts
        model_config: Optional[Dict[str, Any]] = None,
        # Iteration context for prompt injection
        iteration_context: str = "",
        # Legacy parameters for backwards compatibility
        deeptta_summary_path: str = "",
        model_problem_path: str = "",
        paper_summary_paths: List[str] = None,
        debate_round: int = 1,
        debate_history: str = "",
        checkpointer=None
    ):

        super().__init__(f"critic_agent_{mode}", checkpointer=checkpointer)
        self.mode = mode
        self.target_summary_path = target_summary_path or deeptta_summary_path
        self.output_path = output_path
        self.weakness_path = weakness_path or model_problem_path
        self.proposal_paths = proposal_paths or []

        # critique_ranked mode parameters
        self.ranking_r1_path = ranking_r1_path
        self.proposal1_path = proposal1_path
        self.proposal2_path = proposal2_path

        # Model configuration for dynamic prompts
        self.model_config = model_config
        self.iteration_context = iteration_context
        self.prompt_factory = PromptFactory(model_config, iteration_context=iteration_context) if model_config else None

        # Legacy parameters
        self.deeptta_summary_path = deeptta_summary_path
        self.model_problem_path = model_problem_path
        self.paper_summary_paths = paper_summary_paths or []
        self.debate_round = debate_round
        self.debate_history = debate_history

        logger.debug(f"[CriticAgent] Initialized with mode={mode}, model_config={'provided' if model_config else 'None'}")

    def get_prompt(self):
        """Return system prompt based on mode.

        Uses PromptFactory for dynamic prompts if model_config is provided,
        otherwise falls back to legacy static prompts.
        """
        # Use PromptFactory for dynamic prompts if available
        if self.prompt_factory:
            return self._get_dynamic_prompt()
        else:
            return self._get_legacy_prompt()

    def _get_dynamic_prompt(self):
        """Generate prompt using PromptFactory (model-aware)."""
        if self.mode == "analyze_weakness":
            return self.prompt_factory.get_weakness_analysis_prompt(
                target_summary_path=self.target_summary_path,
                output_path=self.output_path
            )
        elif self.mode == "critique_proposals_initial":
            return self.prompt_factory.get_critique_proposals_prompt(
                weakness_path=self.weakness_path,
                proposal_paths=self.proposal_paths
            )
        elif self.mode == "critique_ranked":
            return self.prompt_factory.get_critique_ranked_prompt(
                ranking_r1_path=self.ranking_r1_path,
                proposal1_path=self.proposal1_path,
                proposal2_path=self.proposal2_path,
                weakness_path=self.weakness_path
            )
        elif self.mode == "final_critique":
            # Deprecated - redirects to critique_proposals_initial
            return self.prompt_factory.get_critique_proposals_prompt(
                weakness_path=self.weakness_path,
                proposal_paths=self.proposal_paths
            )
        else:
            # For legacy modes, fall back to static prompts
            return self._get_legacy_prompt()

    def _get_legacy_prompt(self):
        """Generate prompt using legacy static templates (for backwards compatibility)."""
        if self.mode == "analyze_weakness":
            # Use static prompt with dynamic variables from model_config if available
            model_name = self.model_config.get("model_name", "DeepTTA") if self.model_config else "DeepTTA"
            domain = self.model_config.get("domain", "drug response prediction") if self.model_config else "drug response prediction"
            components = self.model_config.get("components", ["drug_encoder", "cell_encoder", "decoder"]) if self.model_config else ["drug_encoder", "cell_encoder", "decoder"]
            components_display = " / ".join(components) + " / overall"

            return CRITIC_WEAKNESS_PROMPT.format(
                target_summary_path=self.target_summary_path,
                output_path=self.output_path,
                model_name=model_name,
                domain=domain,
                components_display=components_display
            )
        elif self.mode == "critique_proposals_initial":
            return CRITIC_FINAL_PROMPT.format(
                weakness_path=self.weakness_path,
                proposal_paths=", ".join(self.proposal_paths)
            )
        elif self.mode == "critique_ranked":
            return CRITIC_RANKED_CRITIQUE_PROMPT.format(
                ranking_r1_path=self.ranking_r1_path,
                proposal1_path=self.proposal1_path,
                proposal2_path=self.proposal2_path,
                weakness_path=self.weakness_path
            )
        elif self.mode == "final_critique":
            return CRITIC_FINAL_PROMPT.format(
                weakness_path=self.weakness_path,
                proposal_paths=", ".join(self.proposal_paths)
            )
        elif self.mode == "critique_proposals":
            # Legacy mode - use existing round-based logic
            if self.debate_round == 1:
                paper_path = self.paper_summary_paths[0] if self.paper_summary_paths else "N/A"
                return CRITIC_ROUND1_PROMPT.format(
                    deeptta_summary_path=self.deeptta_summary_path,
                    vision_summary_path=paper_path
                )
            else:
                paper1 = self.paper_summary_paths[0] if len(self.paper_summary_paths) > 0 else "N/A"
                paper2 = self.paper_summary_paths[1] if len(self.paper_summary_paths) > 1 else "N/A"
                return CRITIC_ROUND2_PROMPT.format(
                    model_problem_path=self.model_problem_path or "N/A",
                    deeptta_summary_path=self.deeptta_summary_path,
                    paper1_summary_path=paper1,
                    paper2_summary_path=paper2,
                    debate_history=self.debate_history or "Use the conversation history in messages."
                )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_additional_tools(self):
        """Provide file reading and writing tools based on mode."""
        tools = [create_read_file_tool(max_length=15000)]

        # Add write_file tool for modes that generate output files
        if self.mode in ["analyze_weakness"]:
            tools.append(create_write_file_tool())

        return tools

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

        logger.info(f"[CriticAgent] Agent created for round {self.debate_round}")
        return agent
