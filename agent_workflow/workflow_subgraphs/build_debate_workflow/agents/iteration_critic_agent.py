"""Iteration Critic Agent for analyzing best iteration results and generating new weakness analysis."""

from typing import Dict, List, Any, Optional
import os
from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from ..prompts.build_debate_prompts import ITERATION_CRITIC_PROMPT
from .common_tools import create_read_file_tool, create_write_file_tool


class IterationCriticAgent(BaseAgentNode):
    """Agent that analyzes best iteration results and generates new weakness analysis.

    This agent is used in iteration 2+ to:
    1. Read memory.json for performance changes (iter 0 baseline vs best iteration)
    2. Read best iteration's weakness and implementation proposal
    3. Read all code in components/ and src/ folders (focusing on *_other.py files)
    4. Read config.yaml to understand which component types were applied
    5. Analyze what worked/failed and why
    6. Generate new weakness_of_target_model.md for the current iteration

    NOTE: t-1이 아닌 best iteration을 분석 대상으로 함.
    """

    def __init__(
        self,
        current_iteration: int,
        best_iteration: int,
        memory_json_path: str,
        best_build_path: str,
        output_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        checkpointer=None
    ):
        super().__init__("iteration_critic_agent", checkpointer=checkpointer)

        self.current_iteration = current_iteration
        self.best_iteration = best_iteration
        self.memory_json_path = memory_json_path
        self.best_build_path = best_build_path
        self.output_path = output_path
        self.model_config = model_config or {}

        # Derive paths from best_build_path
        self.best_weakness_path = os.path.join(
            best_build_path, "build_debate_outputs", "weakness_of_target_model.md"
        )
        self.best_proposal_path = os.path.join(
            best_build_path, "build_debate_outputs", "implementation_proposal.md"
        )
        self.best_components_path = os.path.join(best_build_path, "components")
        self.best_src_path = os.path.join(best_build_path, "src")
        self.best_config_path = os.path.join(best_build_path, "config.yaml")

        logger.debug(
            f"[IterationCriticAgent] Initialized for iteration {current_iteration}, "
            f"best_iteration={best_iteration}, best_build={best_build_path}"
        )

    def _list_files_in_dir(self, dir_path: str, extensions: tuple = (".py",)) -> List[str]:
        """List all files with given extensions in a directory."""
        files = []
        if os.path.exists(dir_path):
            for filename in sorted(os.listdir(dir_path)):
                if filename.endswith(extensions) and not filename.startswith("__"):
                    files.append(os.path.join(dir_path, filename))
        return files

    def _get_components_files(self) -> List[str]:
        """Get all Python files in components/ folder."""
        return self._list_files_in_dir(self.best_components_path)

    def _get_src_files(self) -> List[str]:
        """Get all Python files in src/ folder."""
        return self._list_files_in_dir(self.best_src_path)

    def _get_other_files(self) -> List[str]:
        """Get all *_other.py files (the modified components)."""
        other_files = []
        if os.path.exists(self.best_components_path):
            for filename in sorted(os.listdir(self.best_components_path)):
                if filename.endswith("_other.py"):
                    other_files.append(os.path.join(self.best_components_path, filename))
        return other_files

    def _format_file_list(self, files: List[str], indent: str = "  ") -> str:
        """Format a list of files for display in prompt."""
        if not files:
            return f"{indent}(none found)"
        return "\n".join([f"{indent}- {f}" for f in files])

    def get_prompt(self) -> str:
        """Generate the system prompt for iteration critic analysis."""
        model_name = self.model_config.get("model_name", "Target Model")
        domain = self.model_config.get("domain", "drug response prediction")
        components = self.model_config.get("components", ["encoder", "decoder"])

        # Get file lists
        components_files = self._get_components_files()
        src_files = self._get_src_files()
        other_files = self._get_other_files()

        return ITERATION_CRITIC_PROMPT.format(
            current_iteration=self.current_iteration,
            best_iteration=self.best_iteration,
            memory_json_path=self.memory_json_path,
            best_weakness_path=self.best_weakness_path,
            best_proposal_path=self.best_proposal_path,
            best_config_path=self.best_config_path,
            components_files=self._format_file_list(components_files),
            src_files=self._format_file_list(src_files),
            other_files=self._format_file_list(other_files),
            output_path=self.output_path,
            model_name=model_name,
            domain=domain,
            components=", ".join(components)
        )

    def get_additional_tools(self) -> List:
        """Provide file reading and writing tools."""
        return [
            create_read_file_tool(max_length=20000),
            create_write_file_tool()
        ]

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

        logger.info(f"[IterationCriticAgent] Agent created for iteration {self.current_iteration}")
        return agent
