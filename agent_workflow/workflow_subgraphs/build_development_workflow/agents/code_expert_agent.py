"""Code Expert Agent for build_development workflow.

This agent implements the architecture specified in the proposal
by modifying the template files with actual PyTorch code.
"""

import os
import yaml
from langchain_core.tools import tool
from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from ..prompts.build_development_prompts import (
    CODE_EXPERT_SYSTEM_PROMPT,
    CODE_EXPERT_IMPLEMENTATION_PROMPT,
    CODE_EXPERT_FIX_PROMPT,
)


class CodeExpertAgent(BaseAgentNode):
    """Agent that implements code based on the proposal."""

    def __init__(
        self,
        proposal_path: str,
        target_component: str,
        template_path: str,
        config_path: str = "",
        validator_feedback: str = "",
        is_fix_iteration: bool = False,
        checkpointer=None
    ):
        """Initialize Code Expert Agent.

        Args:
            proposal_path: Path to the implementation proposal MD
            target_component: Which component to implement (drug_encoder/cell_encoder/decoder)
            template_path: Path to the template file to modify
            config_path: Path to the config.yaml file to update
            validator_feedback: Feedback from validator (for fix iterations)
            is_fix_iteration: Whether this is a fix iteration
            checkpointer: Optional LangGraph checkpointer
        """
        super().__init__("code_expert_agent", checkpointer=checkpointer)
        self.proposal_path = proposal_path
        self.target_component = target_component
        self.template_path = template_path
        self.config_path = config_path
        self.validator_feedback = validator_feedback
        self.is_fix_iteration = is_fix_iteration
        logger.debug(f"[CodeExpertAgent] Initialized for {target_component}")

    def get_prompt(self):
        """Return system prompt for code implementation from prompts file."""
        # Use prompts from the prompts file
        system_prompt = CODE_EXPERT_SYSTEM_PROMPT

        if self.is_fix_iteration and self.validator_feedback:
            # Fix iteration prompt
            task_prompt = CODE_EXPERT_FIX_PROMPT.format(
                validator_feedback=self.validator_feedback
            )
        else:
            # Initial implementation prompt
            task_prompt = CODE_EXPERT_IMPLEMENTATION_PROMPT.format(
                component=self.target_component,
                template_path=self.template_path,
                proposal_path=self.proposal_path,
                config_path=self.config_path
            )

        return f"{system_prompt}\n\n{task_prompt}"

    def get_additional_tools(self):
        """Provide file manipulation tools."""
        return [
            self._create_read_file_tool(),
            self._create_write_file_tool(),
            self._create_replace_in_file_tool(),
            self._create_list_directory_tool(),
            self._create_update_yaml_config_tool(),
        ]

    def _create_read_file_tool(self):
        """Create a tool for reading files."""
        @tool
        def read_file(file_path: str) -> str:
            """Read contents of a file.

            Args:
                file_path: Path to the file to read

            Returns:
                File contents as string
            """
            try:
                if os.path.isdir(file_path):
                    return f"ERROR: '{file_path}' is a directory, not a file."
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except Exception as e:
                return f"Error reading file: {e}"

        return read_file

    def _create_write_file_tool(self):
        """Create a tool for writing files."""
        @tool
        def write_file(file_path: str, content: str) -> str:
            """Write content to a file (overwrites existing).

            Args:
                file_path: Path to write to
                content: Complete file content to write

            Returns:
                Success message or error
            """
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully wrote to {file_path}"
            except Exception as e:
                return f"Error writing file: {e}"

        return write_file

    def _create_replace_in_file_tool(self):
        """Create a tool for replacing content in files."""
        @tool
        def replace_in_file(file_path: str, old_string: str, new_string: str) -> str:
            """Replace a string in a file.

            Args:
                file_path: Path to the file
                old_string: Text to find and replace
                new_string: Text to replace with

            Returns:
                Success message or error
            """
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if old_string not in content:
                    return f"ERROR: String not found in file. Make sure the old_string matches exactly (including whitespace)."

                new_content = content.replace(old_string, new_string, 1)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                return f"Successfully replaced in {file_path}"
            except Exception as e:
                return f"Error replacing in file: {e}"

        return replace_in_file

    def _create_list_directory_tool(self):
        """Create a tool for listing directory contents."""
        @tool
        def list_directory(dir_path: str) -> str:
            """List files and subdirectories in a directory.

            Args:
                dir_path: Path to the directory

            Returns:
                List of files and directories
            """
            try:
                if not os.path.exists(dir_path):
                    return f"Directory not found: {dir_path}"
                if not os.path.isdir(dir_path):
                    return f"Not a directory: {dir_path}"
                items = []
                for item in sorted(os.listdir(dir_path)):
                    full_path = os.path.join(dir_path, item)
                    prefix = "[DIR]" if os.path.isdir(full_path) else "[FILE]"
                    items.append(f"{prefix} {item}")
                return "\n".join(items) if items else "Empty directory"
            except Exception as e:
                return f"Error listing directory: {e}"

        return list_directory

    def _create_update_yaml_config_tool(self):
        """Create a tool for updating YAML config files."""
        @tool
        def update_yaml_config(config_path: str, param_path: str, new_value: str) -> str:
            """Update a parameter in a YAML config file.

            This tool modifies specific parameters in YAML config files
            while preserving the file structure and comments where possible.

            Args:
                config_path: Path to the YAML config file (e.g., /path/to/config.yaml)
                param_path: Dot-separated path to the parameter (e.g., "encoder.architecture.hidden_dim")
                new_value: New value for the parameter. Will be auto-converted to appropriate type
                          (int, float, bool, or string)

            Returns:
                Success message or error description

            Examples:
                update_yaml_config("/path/config.yaml", "encoder.architecture.hidden_dim", "256")
                update_yaml_config("/path/config.yaml", "training.learning_rate", "0.0005")
                update_yaml_config("/path/config.yaml", "training.epochs", "200")
            """
            try:
                # Read existing config
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                if config is None:
                    return f"Error: Config file {config_path} is empty or invalid"

                # Parse the parameter path
                keys = param_path.split('.')

                # Navigate to the parent of the target key
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

                final_key = keys[-1]
                old_value = current.get(final_key, "NOT_SET")

                # Auto-convert value type
                converted_value = self._convert_value(new_value)

                # Update the value
                current[final_key] = converted_value

                # Write back to file
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

                return f"Successfully updated {param_path}: {old_value} → {converted_value}"

            except FileNotFoundError:
                return f"Error: Config file not found: {config_path}"
            except yaml.YAMLError as e:
                return f"Error parsing YAML: {e}"
            except Exception as e:
                return f"Error updating config: {e}"

        return update_yaml_config

    def _convert_value(self, value_str: str):
        """Convert string value to appropriate Python type."""
        # Try boolean
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False

        # Try integer
        try:
            return int(value_str)
        except ValueError:
            pass

        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Keep as string
        return value_str

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

        logger.info(f"[CodeExpertAgent] Agent created for {self.target_component}")
        return agent
