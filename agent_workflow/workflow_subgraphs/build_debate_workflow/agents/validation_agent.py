"""Validation Agent for build_debate workflow.

Validates that the code proposed by Proposal Agent is correct.
Checks if the code matches the source repository.
"""

from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from .common_tools import (
    create_read_file_tool,
    create_read_github_file_tool,
    create_list_repo_structure_tool,
)


VALIDATION_AGENT_PROMPT = """You are a Code Validation Agent.

## Your Task
Validate that the code proposed by the Proposal Agent is correct and complete.

## Context
- Proposal Agent wants to use code from: {source_repo_path}
- Target file: {target_file_path}
- Target class/function: {target_code_name}

## Validation Process
1. Use read_github_file to read the target file
2. Check if the proposed code matches the actual code in the repository
3. Verify all necessary imports and dependencies are identified

## Validation Criteria
- PASS: The proposed code correctly identifies the target class/function
- FAIL: The code is incomplete, wrong file, or missing dependencies

## Output Format
You MUST respond with EXACTLY one of these formats:

If valid:
```
VALIDATION: PASS
REASON: [brief explanation]
CODE_VERIFIED: [class/function name]
FILE_PATH: [full path]
```

If invalid:
```
VALIDATION: FAIL
REASON: [what's wrong or missing]
SUGGESTION: [what should be done instead]
```
"""


class ValidationAgent(BaseAgentNode):
    """Agent that validates code proposals."""

    def __init__(
        self,
        source_repo_path: str,
        target_file_path: str,
        target_code_name: str,
        checkpointer=None
    ):
        super().__init__("validation_agent", checkpointer=checkpointer)
        self.source_repo_path = source_repo_path
        self.target_file_path = target_file_path
        self.target_code_name = target_code_name
        logger.debug("[ValidationAgent] Initialized")

    def get_prompt(self):
        return VALIDATION_AGENT_PROMPT.format(
            source_repo_path=self.source_repo_path,
            target_file_path=self.target_file_path,
            target_code_name=self.target_code_name,
        )

    def get_additional_tools(self):
        return [
            create_read_file_tool(max_length=30000),
            create_read_github_file_tool(),
            create_list_repo_structure_tool(),
        ]

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

        logger.info("[ValidationAgent] Agent created")
        return agent
