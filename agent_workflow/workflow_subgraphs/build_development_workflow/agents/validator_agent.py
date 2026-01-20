"""Validator Agent for build_development workflow.

This agent validates that the Code Expert's implementation
matches the proposal and is correct PyTorch code.
"""

import os
from langchain_core.tools import tool
from agent_workflow.logger import logger
from agent_workflow.routing_logic.base_agent_node import BaseAgentNode
from ..prompts.build_development_prompts import (
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_CHECK_PROMPT,
)


class ValidatorAgent(BaseAgentNode):
    """Agent that validates code implementation against proposal."""

    def __init__(
        self,
        proposal_path: str,
        code_path: str,
        target_component: str,
        checkpointer=None
    ):
        """Initialize Validator Agent.

        Args:
            proposal_path: Path to the implementation proposal MD
            code_path: Path to the implemented code file
            target_component: Which component was implemented
            checkpointer: Optional LangGraph checkpointer
        """
        super().__init__("validator_agent", checkpointer=checkpointer)
        self.proposal_path = proposal_path
        self.code_path = code_path
        self.target_component = target_component
        logger.debug(f"[ValidatorAgent] Initialized for {target_component}")

    def get_prompt(self):
        """Return system prompt for validation from prompts file."""
        # Use prompts from the prompts file
        system_prompt = VALIDATOR_SYSTEM_PROMPT
        task_prompt = VALIDATOR_CHECK_PROMPT.format(
            proposal_path=self.proposal_path,
            code_path=self.code_path,
            component=self.target_component
        )

        return f"{system_prompt}\n\n{task_prompt}"

    def get_additional_tools(self):
        """Provide file reading and code checking tools."""
        return [
            self._create_read_file_tool(),
            self._create_check_syntax_tool(),
            self._create_check_imports_tool(),
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

    def _create_check_syntax_tool(self):
        """Create a tool for checking Python syntax."""
        @tool
        def check_python_syntax(file_path: str) -> str:
            """Check if a Python file has valid syntax.

            Args:
                file_path: Path to the Python file

            Returns:
                "SYNTAX OK" or error message with details
            """
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                compile(code, file_path, 'exec')
                return "SYNTAX OK: No syntax errors found."

            except SyntaxError as e:
                return f"SYNTAX ERROR at line {e.lineno}: {e.msg}\n  {e.text}"
            except Exception as e:
                return f"Error checking syntax: {e}"

        return check_python_syntax

    def _create_check_imports_tool(self):
        """Create a tool for checking imports."""
        @tool
        def check_imports(file_path: str) -> str:
            """Check if all imports in a Python file are valid.

            Also detects symbols that are USED but NOT imported/defined.

            Args:
                file_path: Path to the Python file

            Returns:
                Report of import status and missing imports
            """
            import ast

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                tree = ast.parse(code)

                # Collect imported names
                imported_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.asname if alias.asname else alias.name
                            imported_names.add(name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            name = alias.asname if alias.asname else alias.name
                            imported_names.add(name)

                # Collect defined names (classes, functions, variables, parameters)
                defined_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        defined_names.add(node.name)
                    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        defined_names.add(node.name)
                        # Add function parameters
                        for arg in node.args.args:
                            defined_names.add(arg.arg)
                        for arg in node.args.kwonlyargs:
                            defined_names.add(arg.arg)
                        if node.args.vararg:
                            defined_names.add(node.args.vararg.arg)
                        if node.args.kwarg:
                            defined_names.add(node.args.kwarg.arg)
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        defined_names.add(node.id)
                    # Add comprehension variables
                    elif isinstance(node, ast.comprehension):
                        if isinstance(node.target, ast.Name):
                            defined_names.add(node.target.id)

                # Collect used names (function calls, class instantiations)
                used_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        used_names.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        # Get the root name (e.g., 'nn' from 'nn.Module')
                        current = node
                        while isinstance(current, ast.Attribute):
                            current = current.value
                        if isinstance(current, ast.Name):
                            used_names.add(current.id)

                # Python builtins to ignore
                builtins = {'True', 'False', 'None', 'print', 'len', 'range', 'int', 'str',
                           'float', 'list', 'dict', 'set', 'tuple', 'type', 'isinstance',
                           'hasattr', 'getattr', 'setattr', 'super', 'self', 'cls',
                           'Exception', 'KeyError', 'ValueError', 'TypeError', 'open',
                           'any', 'all', 'zip', 'enumerate', 'sorted', 'reversed', 'map', 'filter',
                           'next', 'iter', 'min', 'max', 'sum', 'abs', 'round', 'pow',
                           'id', 'repr', 'hash', 'callable', 'dir', 'vars', 'globals', 'locals',
                           'object', 'property', 'staticmethod', 'classmethod',
                           'NotImplementedError', 'RuntimeError', 'AttributeError', 'IndexError',
                           'StopIteration', 'AssertionError', 'ImportError', 'FileNotFoundError',
                           'bool', 'bytes', 'bytearray', 'memoryview', 'complex', 'slice',
                           'frozenset', 'input', 'format', 'chr', 'ord', 'bin', 'hex', 'oct',
                           'breakpoint', 'compile', 'eval', 'exec', 'delattr', 'divmod', 'issubclass'}

                # Find missing imports
                all_defined = imported_names | defined_names | builtins
                missing = used_names - all_defined

                # Check common PyTorch imports
                has_torch = 'torch' in imported_names
                has_nn = 'nn' in imported_names

                report = f"=== Import Analysis ===\n"
                report += f"Imported: {len(imported_names)} names\n"
                report += f"Defined: {len(defined_names)} names\n"
                report += f"Used: {len(used_names)} names\n"

                report += f"\n=== PyTorch Status ===\n"
                report += f"  - torch imported: {'YES' if has_torch else 'NO'}\n"
                report += f"  - nn module imported: {'YES' if has_nn else 'NO'}\n"

                if missing:
                    report += f"\n=== MISSING IMPORTS (CRITICAL!) ===\n"
                    report += f"The following symbols are USED but NOT imported/defined:\n"
                    for name in sorted(missing):
                        report += f"  - {name}\n"
                    report += f"\nYou MUST add imports for these symbols!"
                else:
                    report += f"\n=== All imports OK ===\n"
                    report += "No missing imports detected."

                return report

            except SyntaxError as e:
                return f"Cannot check imports - syntax error: {e}"
            except Exception as e:
                return f"Error checking imports: {e}"

        return check_imports

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

        logger.info(f"[ValidatorAgent] Agent created for {self.target_component}")
        return agent
