"""Proposal Agent for build_debate workflow.

This agent synthesizes the final ranking into a clear implementation proposal
that the Code Expert can use to implement the new architecture.
It also explores GitHub code and works with Validation Agent to verify code.

Supports dynamic prompts via PromptFactory for different bioinformatics models.
"""

from typing import Any, Dict, List, Optional
from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from ..prompts.prompt_factory import PromptFactory
from .common_tools import (
    create_read_file_tool,
    create_write_file_tool,
    create_read_github_file_tool,
    create_list_repo_structure_tool,
)


class ProposalAgent(BaseAgentNode):
    """Agent that synthesizes ranking into implementation proposal.

    Uses PromptFactory for dynamic prompt generation based on model configuration.
    """

    def __init__(
        self,
        final_proposal_path: str,
        other_summary_paths: List[str],
        repos_dir: str,
        output_path: str,
        # Model configuration for dynamic prompts
        model_config: Optional[Dict[str, Any]] = None,
        # Iteration context for prompt injection
        iteration_context: str = "",
        # Legacy parameters for backwards compatibility
        debate_transcript_path: str = "",
        cell_encoder_template: str = "",
        drug_encoder_template: str = "",
        decoder_template: str = "",
        checkpointer=None
    ):
        """Initialize Proposal Agent.

        Args:
            final_proposal_path: Path to final rank #1 proposal MD
            other_summary_paths: List of paths to reference paper summaries
            repos_dir: Path to cloned GitHub repositories
            output_path: Path where proposal will be saved
            model_config: Model configuration dict for dynamic prompts (from MODEL_WORKFLOW_CONFIG)
            debate_transcript_path: Legacy - Path to debate transcript MD
            cell_encoder_template: Legacy - Path to cell encoder template
            drug_encoder_template: Legacy - Path to drug encoder template
            decoder_template: Legacy - Path to decoder template
            checkpointer: Optional LangGraph checkpointer
        """
        super().__init__("proposal_agent", checkpointer=checkpointer)

        # Primary parameters
        self.final_proposal_path = final_proposal_path or debate_transcript_path
        self.other_summary_paths = other_summary_paths or []
        self.repos_dir = repos_dir
        self.output_path = output_path

        # Model configuration
        self.model_config = model_config
        self.iteration_context = iteration_context
        self.prompt_factory = PromptFactory(model_config, iteration_context=iteration_context) if model_config else None

        # Extract model-specific settings
        if model_config:
            self.model_name = model_config.get("model_name", "Unknown")
            self.domain = model_config.get("domain", "machine learning")
            self.components = model_config.get("components", ["encoder", "decoder"])
            self.component_templates = model_config.get("component_templates", {})
        else:
            # Legacy defaults
            self.model_name = "DeepTTA"
            self.domain = "drug response prediction"
            self.components = ["drug_encoder", "cell_encoder", "decoder"]
            self.component_templates = {
                "drug_encoder": drug_encoder_template,
                "cell_encoder": cell_encoder_template,
                "decoder": decoder_template,
            }

        logger.debug(f"[ProposalAgent] Initialized for {self.model_name}")

    def get_prompt(self):
        """Return system prompt for proposal generation.

        Uses dynamic model configuration if available.
        """
        # Build component list for display
        components_display = " / ".join(self.components)

        # Build template files section
        template_lines = []
        for comp, path in self.component_templates.items():
            comp_display = comp.replace("_", " ").title()
            template_lines.append(f"- {comp_display}: {path}")
        templates_section = "\n".join(template_lines) if template_lines else "- (No templates configured)"

        return f"""You are a Proposal Agent for {self.model_name} architecture improvement.

## Domain Context
- Model: {self.model_name}
- Domain: {self.domain}
- Components: {components_display}

## Your Task
1. Read the final rank #1 proposal and reference paper summaries
2. Decide which {self.model_name} component to modify ({components_display})
3. Identify which model's code to use
4. Create an implementation proposal with verified code

## Input Files
- Final Rank #1 Proposal: {self.final_proposal_path}
- Reference Paper Summaries: {', '.join(self.other_summary_paths)}
- Cloned Repos Directory: {self.repos_dir}

## Template Files (for reference)
{templates_section}

## Output
Save proposal to: {self.output_path}

## Process
STEP 1: Read the final proposal and paper summaries
STEP 2: Write initial proposal (target component, source model, rationale)
STEP 3: Explore the cloned repository to find the actual code
STEP 4: For each code file needed, output a CODE_REQUEST block

## CODE_REQUEST Format
When you need to include code from the cloned repo, output:
```
CODE_REQUEST:
  SOURCE_REPO: [repo name]
  FILE_PATH: [full path to file]
  TARGET_CLASS: [class name to extract]
  REASON: [why this code is needed]
```

After validation, you will receive the verified code to add to the proposal.

## Final Proposal Format
```markdown
# Implementation Proposal for {self.model_name}

## 1. Decision Summary
- Target Component: [{components_display}]
- Source Model: [model name from reference paper]
- Rationale: [why this improves {self.model_name}]

## 2. Architecture Overview
[High-level description of the new architecture]

## 3. Code to Implement

### Auxiliary Modules
[Code for helper classes - to be added after validation]

### Main Encoder/Decoder
[Code for main class - to be added after validation]

## 4. Integration Notes
- Input format: [expected input]
- Output format: [expected output]
- Config parameters: [key hyperparameters]
- Integration with {self.model_name}: [how to integrate]
```

IMPORTANT: Use write_file tool to save the proposal to {self.output_path}
"""

    def get_additional_tools(self):
        """Provide file reading, writing, and GitHub exploration tools."""
        return [
            create_read_file_tool(max_length=30000),
            create_write_file_tool(),
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

        logger.info(f"[ProposalAgent] Agent created for {self.model_name}")
        return agent
