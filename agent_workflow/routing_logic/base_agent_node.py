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

# Environment variable for debug logging
DEBUG_LOGS = os.getenv("AUTODRP_DEBUG_LOGS", "false").lower() == "true"
COMPILE_MODE = "compile" in " ".join(sys.argv)

# Global tracking for agent initialization logging (prevents duplicate INFO logs)
_AGENT_READY_LOGGED = set()

# Helper function for conditional logging
def debug_print(*args, **kwargs):
    """Print only if debug logging is enabled and not in compile mode."""
    if DEBUG_LOGS and not COMPILE_MODE:
        print(*args, **kwargs)


class BaseAgentNode(ABC):
    """Base class for all agent nodes in MARBLE system."""

    def __init__(self, node_name: str, checkpointer=None):
        """Initialize base agent node.

        Args:
            node_name: Name of the node for logging and identification
            checkpointer: Optional checkpointer for agent-level memory persistence
                         (e.g., InMemorySaver for conversation history sharing)
        """
        self.node_name = node_name
        self.checkpointer = checkpointer  # Agent-level checkpointer for memory sharing
        self._cached_agent = None  # Agent caching for performance
        self._agent_initialized = False
        self.compiled_agent = None  # For direct graph registration
    
    def get_mcp_manager(self) -> Any: # Type hint could be MCPManager if imported
        """Get or create MCP manager instance from global state."""
        # Directly retrieve the globally initialized MCPManager instance
        manager = GlobalStateManager.get_mcp_manager_instance()
        if manager is None:
            # This indicates a serious problem: initialize_system was not called or failed.
            raise RuntimeError("[BaseAgentNode] MCPManager not found in GlobalStateManager. Ensure system initialization.")
        return manager
    
    def get_mcp_tools(self, server_name: str):
        """Get tools from MCP server directly through MCP manager."""
        try:
            # Get MCP manager and retrieve tools directly
            mcp_manager = self.get_mcp_manager()

            # Return empty if MCP is disabled
            if not mcp_manager.mcp_enabled:
                debug_print(f"ℹ️ [{self.node_name}] MCP disabled - returning empty tools for {server_name}")
                return []

            if not mcp_manager._initialized:
                debug_print(f"⚠️ [{self.node_name}] MCP Manager not initialized for {server_name}")
                return []

            tools = mcp_manager.get_tools_from_server(server_name)
            return tools
        except Exception as e:
            debug_print(f"❌ [{self.node_name}] Error getting tools from {server_name}: {e}")
            return []

   
    @abstractmethod
    def get_prompt(self):
        """Get the prompt for this node. Must be implemented by subclasses."""
        pass
    
    def get_additional_tools(self) -> List[Any]:
        """Get additional tools specific to this node. Override in subclasses if needed."""
        return []
    
    def get_or_create_agent(self):
        """Get cached agent or create new one if not exists."""
        # If we have a compiled agent from initialize_agent(), use it
        if self.compiled_agent is not None:
            return self.compiled_agent

        # Otherwise, fall back to runtime creation (backward compatibility)
        # Check MCP Manager availability
        mcp_manager = self.get_mcp_manager()
        if not mcp_manager or not mcp_manager._initialized:
            debug_print(f"⚠️ [{self.node_name}] MCP Manager not ready, creating agent without tools")

        if self._cached_agent is None or not self._agent_initialized:
            self._cached_agent = self.create_agent()
            self._agent_initialized = True
            debug_print(f"[{self.node_name}] Agent cached successfully")
        return self._cached_agent

    def initialize_agent(self):
        """Initialize agent at graph build time for visualization.

        This method should be called during graph construction to create
        the agent with all tools, making them visible in LangGraph visualization.

        Returns:
            The compiled create_react_agent instance
        """
        logger.debug(f"🚀 [{self.node_name}] Initializing agent for graph visualization...")

        # Check MCP Manager is ready (안전한 예외 처리)
        try:
            mcp_manager = self.get_mcp_manager()
            if not mcp_manager or not mcp_manager._initialized:
                logger.warning(f"⚠️ [{self.node_name}] MCP Manager not initialized, agent will have no tools")
        except RuntimeError as e:
            # System not yet initialized - agent will work without MCP tools initially
            logger.warning(f"⚠️ [{self.node_name}] GlobalStateManager not initialized yet: {e}")
            mcp_manager = None

        # Create the agent
        self.compiled_agent = self.create_agent()
        self._agent_initialized = True

        return self.compiled_agent

    
    def create_agent(self):
        """Create agent with MCP tools."""
        # Initialize model at runtime with parameters for reproducibility
        model = init_chat_model(
            MODEL_NAME,
            model_provider=MODEL_PROVIDER,
            **MODEL_PARAMS,  # All parameters unpacked directly
        )

        # Get MCP assignments from configuration
        mcp_names = NODE_MCP_MAPPING.get(self.node_name, [])

        tools = []
        tool_summary = {}

        # Collect tools from all assigned MCP servers
        for mcp_name in mcp_names:
            server_tools = self.get_mcp_tools(mcp_name)

            if server_tools:
                tools.extend(server_tools)
                tool_summary[mcp_name] = len(server_tools)

                # Log first few tool names for verification (DEBUG only)
                tool_names = [getattr(tool, 'name', str(tool)[:50]) for tool in server_tools[:3]]
                debug_print(f"   📝 [{self.node_name}] Sample tools: {tool_names}")
            else:
                debug_print(f"⚠️  [{self.node_name}] No tools received from {mcp_name}")
                tool_summary[mcp_name] = 0

        # Add node-specific additional tools
        additional_tools = self.get_additional_tools()
        if additional_tools:
            tools.extend(additional_tools)
            tool_summary["additional"] = len(additional_tools)

        # Tool loading complete - Log summary (DEBUG only)
        total_tools = len(tools)
        debug_print(f"📊 [{self.node_name}] Tool loading complete:")
        debug_print(f"   Total tools: {total_tools}")
        debug_print(f"   By server: {tool_summary}")
        if total_tools == 0:
            debug_print(f"   ⚠️ WARNING: No tools loaded for agent {self.node_name}!")

        # Log agent ready with tool count (log once at INFO, then DEBUG)
        global _AGENT_READY_LOGGED
        if self.node_name not in _AGENT_READY_LOGGED:
            logger.info(f"✅ [{self.node_name}] Ready ({total_tools} tools)")
            _AGENT_READY_LOGGED.add(self.node_name)
        else:
            logger.debug(f"✅ [{self.node_name}] Ready ({total_tools} tools) [cached]")

        # Create agent with collected tools
        from configs.config import LANGGRAPH_CONFIG
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
        import re

        # Get static prompt
        prompt_string = self.get_prompt()

        # Define valid state variables that can be used in templates
        # These are the actual MARBLEState fields that can be injected
        # Note: Some legacy fields kept for backward compatibility (debate_config, research_config, development_config)
        VALID_STATE_VARIABLES = {
            'target_model', 'target_component', 'current_phase', 'turn_count', 'current_topic',
            'rep_conclusion', 'ml_conclusion', 'critic_conclusion',
            'debate_session_id', 'agenda_report_path',
            'debate_config', 'research_config', 'development_config'  # Legacy - not used in state
        }

        # Extract template variables using regex to find {variable_name} patterns
        template_vars = set(re.findall(r'\{(\w+)\}', prompt_string))

        # Check if any valid state variables are present
        has_valid_template_vars = bool(template_vars & VALID_STATE_VARIABLES)

        # Convert to ChatPromptTemplate only if valid state variables exist
        if has_valid_template_vars:
            # Create template that accepts state variables (current_phase, turn_count, etc.)
            system_template = SystemMessagePromptTemplate.from_template(prompt_string)
            prompt = ChatPromptTemplate.from_messages([
                system_template,
                ("placeholder", "{messages}")  # For conversation history
            ])
            logger.debug(f"📝 [{self.node_name}] Using ChatPromptTemplate with state variables: {template_vars & VALID_STATE_VARIABLES}")
        else:
            # No valid template variables, use string as-is (treats { } as literal text)
            prompt = prompt_string
            logger.debug(f"📝 [{self.node_name}] Using static string prompt")

        # Create agent with optional checkpointer for memory persistence
        agent_kwargs = {
            "model": model,
            "tools": tools,
            "prompt": prompt,
            "state_schema": MARBLEState
        }

        # Add checkpointer if provided (enables conversation memory sharing)
        if self.checkpointer is not None:
            agent_kwargs["checkpointer"] = self.checkpointer

        agent = create_react_agent(**agent_kwargs).with_config(
            recursion_limit=LANGGRAPH_CONFIG["recursion_limit"]
        )

        return agent
    
    def update_domain_state(self, state: MARBLEState, agent_response: str) -> dict:
        """Hook for nodes to update domain-specific state fields.
        
        This method provides a common JSON extraction utility for all nodes.
        Override this method in specialized nodes to extract information from 
        agent responses and return reducer-compatible dictionaries.
        
        Args:
            state: Current state
            agent_response: The agent's response content
            
        Returns:
            Dictionary with state updates that will be processed by reducers
        """
        # Default implementation - no domain-specific updates
        return {}
    
    def extract_json_from_response(self, agent_response: str, expected_key: str = None) -> dict:
        """Common utility to extract JSON structure from agent response.
        
        Args:
            agent_response: The agent's response content
            expected_key: Optional key to look for in the JSON structure
            
        Returns:
            Parsed JSON data or empty dict if extraction fails
        """
        import json
        import re
        
        try:
            # Try to extract JSON structure from agent response
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
        """Validate that required state fields for this node are present.
        
        Override in specialized nodes to add specific validation requirements.
        
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        
        # Basic validation for all nodes
        if not state.get("messages"):
            state["messages"] = []  # Initialize if missing
        
        return errors
    
    async def execute_node(self, state: MARBLEState) -> MARBLEState:
        """Execute the agent node.

        This method can be used as a fallback when the agent needs to be
        wrapped with additional logic. For direct agent usage, use
        initialize_agent() and register the agent directly.
        """
        try:
            # If we have a compiled agent, we shouldn't need to wait
            if self.compiled_agent is None:
                # Wait for MCP Manager to be ready (backward compatibility)
                import asyncio
                max_retries = 20
                for i in range(max_retries):
                    mcp_manager = self.get_mcp_manager()
                    if mcp_manager and mcp_manager._initialized:
                        break
                    await asyncio.sleep(0.5)
                    if i == 0:
                        debug_print(f"⏳ [{self.node_name}] Waiting for MCP Manager initialization...")

            # Validate required state fields
            validation_errors = self.validate_required_state(state)
            if validation_errors:
                error_msg = f"{self.node_name} validation failed: {'; '.join(validation_errors)}"
                print(f"[ERROR] {error_msg}")
                return {
                    'processing_logs': [f"{self.node_name.replace('_', ' ').title()} failed: {error_msg}"]
                }

            # Get or create cached agent
            agent = self.get_or_create_agent()
            result = await agent.ainvoke(state)
            
            # Update state with agent response
            updated_state = {**state, "messages": result["messages"]}
            
            # Extract agent response content
            last_message = result["messages"][-1].content
            
            # Call domain-specific state update hook and get reducer-compatible updates
            domain_updates = self.update_domain_state(updated_state, last_message)

            # Store analysis results in global state
            GlobalStateManager.update_state({
                f"{self.node_name}_results": last_message,
                "current_node": self.node_name
            }, self.node_name.upper(), self.node_name)
            
            # Merge domain updates with start and completion logs and return for reducer processing
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
        """Generate free-form report for this node.
        
        Override in subclasses to customize report generation.
        Default implementation returns the agent's natural language response.
        
        Args:
            state: Current state dictionary
            response: Agent's response content
            
        Returns:
            Report content as string or None
        """
        # Default: return agent's response as is (already natural language)
        return response
    
    def save_report(self, content: str, report_type: str = None) -> str:
        """Save report to file system.
        
        Args:
            content: Report content to save
            report_type: Optional report type suffix
            
        Returns:
            Path to saved report file
        """
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Build report filename
            if report_type:
                filename = f"{self.node_name}_{report_type}_{timestamp}.md"
            else:
                filename = f"{self.node_name}_report_{timestamp}.md"
            
            # Build full path
            import os
            report_dir = "/workspace/reports"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, filename)
            
            # Get MCP manager and save file
            mcp_manager = self.get_mcp_manager()
            if mcp_manager:
                # Try to use Serena MCP for file creation
                try:
                    mcp_manager.call_tool(
                        "create_text_file",
                        file_path=report_path,
                        content=content
                    )
                    print(f"📄 [{self.node_name}] Report saved: {report_path}")
                except:
                    # Fallback to desktop-commander if Serena not available
                    mcp_manager.call_tool(
                        "write_file",
                        path=report_path,
                        content=content
                    )
                    print(f"📄 [{self.node_name}] Report saved (fallback): {report_path}")
            else:
                # Fallback to native Python if MCP not available
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"📄 [{self.node_name}] Report saved (native): {report_path}")
            
            return report_path
            
        except Exception as e:
            print(f"⚠️ [{self.node_name}] Failed to save report: {e}")
            return ""
    
    def create_route_function(self, success_route: str):
        """Create a routing function for this node."""
        def route_function(state: MARBLEState) -> str:
            try:
                # Check if analysis was successful
                last_message = state.get("messages", [])[-1].content if state.get("messages") else ""
                
                # Simple success check - if no error keywords found, proceed
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
