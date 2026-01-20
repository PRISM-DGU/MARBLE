"""Paper Reader Agent for build_debate workflow.

This agent reads a PDF paper and extracts method/architecture information,
saving a structured summary markdown file.
"""

from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from ..prompts.build_debate_prompts import (
    PAPER_READER_DEEPTTA_PROMPT,
    PAPER_READER_VISION_PROMPT,
    PAPER_READER_OTHER_PROMPT,
)
from .common_tools import (
    create_read_file_tool,
    create_write_file_tool,
    create_read_pdf_tool,
    create_extract_github_urls_tool,
    create_clone_github_repo_tool,
    create_read_github_file_tool,
    create_list_repo_structure_tool,
)


class PaperReaderAgent(BaseAgentNode):
    """Agent that reads a PDF paper and produces a method summary."""

    def __init__(self, paper_name: str, pdf_path: str, output_path: str, checkpointer=None):
        """Initialize Paper Reader Agent.

        Args:
            paper_name: Name identifier for the paper (e.g., "deeptta", "vision_model")
            pdf_path: Path to the PDF file
            output_path: Path where the summary MD will be saved
            checkpointer: Optional LangGraph checkpointer
        """
        super().__init__(f"paper_reader_{paper_name}", checkpointer=checkpointer)
        self.paper_name = paper_name
        self.pdf_path = pdf_path
        self.output_path = output_path
        logger.debug(f"[PaperReaderAgent:{paper_name}] Initialized")

    def get_prompt(self):
        """Return system prompt for paper reading from prompts file."""
        if self.paper_name == "deeptta":
            return PAPER_READER_DEEPTTA_PROMPT.format(
                pdf_path=self.pdf_path,
                output_path=self.output_path
            )
        elif self.paper_name == "vision_model":
            return PAPER_READER_VISION_PROMPT.format(
                pdf_path=self.pdf_path,
                output_path=self.output_path
            )
        elif self.paper_name.startswith("other_paper"):
            # Reference papers from article researcher
            import os
            repos_dir = os.path.join(os.path.dirname(self.output_path), "repos")
            return PAPER_READER_OTHER_PROMPT.format(
                pdf_path=self.pdf_path,
                output_path=self.output_path,
                repos_dir=repos_dir
            )
        else:
            # For response agents (deeptta_response, vision_response, etc.)
            # Use a generic prompt
            return f"""You are a {self.paper_name.upper()} Paper Expert Agent.

## Your Task
Read the paper PDF and respond to the discussion based on your paper's content.

## PDF Location
{self.pdf_path}

## Instructions
1. Use the read_pdf tool to read the paper content if needed
2. Provide specific responses based on the paper's methodology
3. Be concrete about architectural details and implementation

## Output Location
{self.output_path}
"""

    def get_additional_tools(self):
        """Provide PDF reading, file writing, and GitHub tools."""
        return [
            create_read_pdf_tool(max_length=50000),
            create_write_file_tool(),
            create_read_file_tool(max_length=10000),
            create_extract_github_urls_tool(),
            create_clone_github_repo_tool(),
            create_read_github_file_tool(),
            create_list_repo_structure_tool(),
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

        logger.info(f"[PaperReaderAgent:{self.paper_name}] Agent created with PDF tools")
        return agent
