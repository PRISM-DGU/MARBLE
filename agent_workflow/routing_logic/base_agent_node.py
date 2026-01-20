"""Global base agent node class for all MARBLE agent nodes."""

import os
import sys
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from datetime import datetime

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from configs.config import MODEL_NAME, MODEL_PROVIDER, NODE_MCP_MAPPING, MODEL_PARAMS
from agent_workflow.logger import logger
from agent_workflow.state import MARBLEState
from agent_workflow.utils import GlobalStateManager

DEBUG_LOGS = os.getenv("AUTODRP_DEBUG_LOGS", "false").lower() == "true"
COMPILE_MODE = "compile" in " ".join(sys.argv)
_AGENT_READY_LOGGED = set()


def debug_print(*args, **kwargs):
    """Print only if debug logging is enabled."""
    if DEBUG_LOGS and not COMPILE_MODE:
        print(*args, **kwargs)


class BaseAgentNode(ABC):
    """Base class for all agent nodes in MARBLE system."""

    def __init__(self, node_name: str, checkpointer=None):
        self.node_name = node_name
        self.checkpointer = checkpointer
        self._cached_agent = None
        self._agent_initialized = False
        self.compiled_agent = None

    def get_mcp_manager(self) -> Any:
        """Get MCP manager instance from global state."""
        manager = GlobalStateManager.get_mcp_manager_instance()
        if manager is None:
            raise RuntimeError("[BaseAgentNode] MCPManager not found in GlobalStateManager.")
        return manager

    def get_mcp_tools(self, server_name: str):
        """Get tools from MCP server."""
        try:
            mcp_manager = self.get_mcp_manager()

            if not mcp_manager.mcp_enabled:
                debug_print(f"[{self.node_name}] MCP disabled - returning empty tools for {server_name}")
                return []

            if not mcp_manager._initialized:
                debug_print(f"[{self.node_name}] MCP Manager not initialized for {server_name}")
                return []

            return mcp_manager.get_tools_from_server(server_name)
        except Exception as e:
            debug_print(f"[{self.node_name}] Error getting tools from {server_name}: {e}")
            return []


    @abstractmethod
    def get_prompt(self):
        """Get the prompt for this node."""
        pass

    def get_additional_tools(self) -> List[Any]:
        """Get additional tools specific to this node."""
        return []

    def get_or_create_agent(self):
        """Get cached agent or create new one."""
        if self.compiled_agent is not None:
            return self.compiled_agent

        mcp_manager = self.get_mcp_manager()
        if not mcp_manager or not mcp_manager._initialized:
            debug_print(f"[{self.node_name}] MCP Manager not ready, creating agent without tools")

        if self._cached_agent is None or not self._agent_initialized:
            self._cached_agent = self.create_agent()
            self._agent_initialized = True
            debug_print(f"[{self.node_name}] Agent cached successfully")
        return self._cached_agent

    def initialize_agent(self):
        """Initialize agent at graph build time for visualization."""
        logger.debug(f"[{self.node_name}] Initializing agent for graph visualization...")

        try:
            mcp_manager = self.get_mcp_manager()
            if not mcp_manager or not mcp_manager._initialized:
                logger.warning(f"[{self.node_name}] MCP Manager not initialized")
        except RuntimeError as e:
            logger.warning(f"[{self.node_name}] GlobalStateManager not initialized yet: {e}")

        self.compiled_agent = self.create_agent()
        self._agent_initialized = True

        return self.compiled_agent


    def create_agent(self):
        """Create agent with MCP tools."""
        model = init_chat_model(
            MODEL_NAME,
            model_provider=MODEL_PROVIDER,
            **MODEL_PARAMS,
        )

        mcp_names = NODE_MCP_MAPPING.get(self.node_name, [])
        tools = []
        tool_summary = {}

        for mcp_name in mcp_names:
            server_tools = self.get_mcp_tools(mcp_name)

            if server_tools:
                tools.extend(server_tools)
                tool_summary[mcp_name] = len(server_tools)
                tool_names = [getattr(tool, 'name', str(tool)[:50]) for tool in server_tools[:3]]
                debug_print(f"[{self.node_name}] Sample tools: {tool_names}")
            else:
                debug_print(f"[{self.node_name}] No tools received from {mcp_name}")
                tool_summary[mcp_name] = 0

        additional_tools = self.get_additional_tools()
        if additional_tools:
            tools.extend(additional_tools)
            tool_summary["additional"] = len(additional_tools)

        total_tools = len(tools)
        debug_print(f"[{self.node_name}] Tool loading complete: {total_tools} total, by server: {tool_summary}")

        global _AGENT_READY_LOGGED
        if self.node_name not in _AGENT_READY_LOGGED:
            logger.info(f"[{self.node_name}] Ready ({total_tools} tools)")
            _AGENT_READY_LOGGED.add(self.node_name)
        else:
            logger.debug(f"[{self.node_name}] Ready ({total_tools} tools) [cached]")

        from configs.config import LANGGRAPH_CONFIG
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
        import re

        prompt_string = self.get_prompt()

        # Valid state variables for template injection
        VALID_STATE_VARIABLES = {
            'target_model', 'target_component', 'current_phase', 'turn_count', 'current_topic',
            'rep_conclusion', 'ml_conclusion', 'critic_conclusion',
            'debate_session_id', 'agenda_report_path',
            'debate_config', 'research_config', 'development_config'
        }

        template_vars = set(re.findall(r'\{(\w+)\}', prompt_string))
        has_valid_template_vars = bool(template_vars & VALID_STATE_VARIABLES)

        if has_valid_template_vars:
            system_template = SystemMessagePromptTemplate.from_template(prompt_string)
            prompt = ChatPromptTemplate.from_messages([
                system_template,
                ("placeholder", "{messages}")
            ])
            logger.debug(f"[{self.node_name}] Using ChatPromptTemplate with state variables: {template_vars & VALID_STATE_VARIABLES}")
        else:
            prompt = prompt_string
            logger.debug(f"[{self.node_name}] Using static string prompt")

        agent_kwargs = {
            "model": model,
            "tools": tools,
            "prompt": prompt,
            "state_schema": MARBLEState
        }

        if self.checkpointer is not None:
            agent_kwargs["checkpointer"] = self.checkpointer

        agent = create_react_agent(**agent_kwargs).with_config(
            recursion_limit=LANGGRAPH_CONFIG["recursion_limit"]
        )

        return agent

    def update_domain_state(self, state: MARBLEState, agent_response: str) -> dict:
        """Hook for nodes to update domain-specific state fields."""
        return {}

    def extract_json_from_response(self, agent_response: str, expected_key: str = None) -> dict:
        """Extract JSON structure from agent response."""
        import json
        import re

        try:
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, agent_response, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                parsed_data = json.loads(json_str)

                if expected_key:
                    return parsed_data.get(expected_key, {})
                else:
                    return parsed_data

        except json.JSONDecodeError as e:
            print(f"[{self.node_name}] JSON parsing failed: {e}")
        except Exception as e:
            print(f"[{self.node_name}] JSON extraction error: {e}")

        return {}

    def validate_required_state(self, state: MARBLEState) -> List[str]:
        """Validate that required state fields are present."""
        errors = []
        if not state.get("messages"):
            state["messages"] = []
        return errors

    async def execute_node(self, state: MARBLEState) -> MARBLEState:
        """Execute the agent node."""
        try:
            if self.compiled_agent is None:
                import asyncio
                max_retries = 20
                for i in range(max_retries):
                    mcp_manager = self.get_mcp_manager()
                    if mcp_manager and mcp_manager._initialized:
                        break
                    await asyncio.sleep(0.5)
                    if i == 0:
                        debug_print(f"[{self.node_name}] Waiting for MCP Manager initialization...")

            validation_errors = self.validate_required_state(state)
            if validation_errors:
                error_msg = f"{self.node_name} validation failed: {'; '.join(validation_errors)}"
                print(f"[ERROR] {error_msg}")
                return {
                    'processing_logs': [f"{self.node_name.replace('_', ' ').title()} failed: {error_msg}"]
                }

            agent = self.get_or_create_agent()
            result = await agent.ainvoke(state)

            updated_state = {**state, "messages": result["messages"]}
            last_message = result["messages"][-1].content
            domain_updates = self.update_domain_state(updated_state, last_message)

            GlobalStateManager.update_state({
                f"{self.node_name}_results": last_message,
                "current_node": self.node_name
            }, self.node_name.upper(), self.node_name)

            node_logs = [
                f"{self.node_name.replace('_', ' ').title()} started",
                *domain_updates.get('processing_logs', []),
                f"{self.node_name.replace('_', ' ').title()} completed"
            ]

            final_updates = {
                **domain_updates,
                'processing_logs': node_logs
            }

            return final_updates

        except Exception as e:
            error_msg = f"{self.node_name.replace('_', ' ').title()} error: {e}"
            print(f"[ERROR] {error_msg}")

            return {
                'processing_logs': [f"{self.node_name.replace('_', ' ').title()} failed: {error_msg}"]
            }

    def generate_report(self, state: Dict, response: str) -> Optional[str]:
        """Generate report for this node."""
        return response

    def save_report(self, content: str, report_type: str = None) -> str:
        """Save report to file system."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if report_type:
                filename = f"{self.node_name}_{report_type}_{timestamp}.md"
            else:
                filename = f"{self.node_name}_report_{timestamp}.md"

            import os
            report_dir = "/workspace/reports"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, filename)

            mcp_manager = self.get_mcp_manager()
            if mcp_manager:
                try:
                    mcp_manager.call_tool(
                        "create_text_file",
                        file_path=report_path,
                        content=content
                    )
                    print(f"[{self.node_name}] Report saved: {report_path}")
                except:
                    mcp_manager.call_tool(
                        "write_file",
                        path=report_path,
                        content=content
                    )
                    print(f"[{self.node_name}] Report saved (fallback): {report_path}")
            else:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"[{self.node_name}] Report saved (native): {report_path}")

            return report_path

        except Exception as e:
            print(f"[{self.node_name}] Failed to save report: {e}")
            return ""

    def create_route_function(self, success_route: str):
        """Create a routing function for this node."""
        def route_function(state: MARBLEState) -> str:
            try:
                last_message = state.get("messages", [])[-1].content if state.get("messages") else ""

                error_keywords = ["error", "failed", "exception", "cannot", "unable"]
                if any(keyword in last_message.lower() for keyword in error_keywords):
                    print(f"[ROUTE] {self.node_name.replace('_', ' ').title()} failed, ending")
                    return "END"

                print(f"[ROUTE] {self.node_name.replace('_', ' ').title()} successful, proceeding to {success_route}")
                return success_route

            except Exception as e:
                print(f"[ROUTE] Error in {self.node_name} routing: {e}")
                return "END"

        return route_function
