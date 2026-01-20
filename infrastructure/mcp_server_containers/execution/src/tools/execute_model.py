"""
Execute Model Tool
Unified MCP tool for drug response model execution workflow

Features:
- Config-driven execution with validation
- Automatic command building from config files
- Override support for quick experimentation
- GPU management
"""

import sys
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Add workspace to path to import existing execution code
sys.path.insert(0, '/workspace')

from agent_workflow.logger import logger

# Import existing execution components
from agent_workflow.workflow_subgraphs.execution_workflow.agents.environment_builder_agent import build_and_run_container
from agent_workflow.workflow_subgraphs.execution_workflow.agents.model_executor_agent import execute_in_container

# Import new config/command tools
from tools.config_validator import validate_config, ConfigValidationError
from tools.command_builder import build_execution_command


class ExecuteModelTool:
    """
    Unified model execution tool for env building and execution.

    This tool orchestrates the execution workflow:
    1. Build isolated Docker environment
    2. Execute model training/evaluation
    3. Return log file path for reporter node analysis
    """

    def __init__(self):
        """Initialize execution tool."""
        self.logger = logger

    async def apply(
        self,
        model_name: str,
        model_source_path: Optional[str] = None,
        config_path: Optional[str] = "config.yaml",
        execution_command: Optional[str] = None,
        mode: str = "train",
        overrides: Optional[Dict[str, Any]] = None,
        gpu_id: Optional[int] = None,
        dataset_type: Optional[str] = None,
        auto_report: bool = True
    ) -> str:
        """
        Execute drug response prediction model in isolated container.

        NEW: Config-driven execution with automatic command building!

        Args:
            model_name: Model identifier (deeptta, deepdr, dgdrp, deeptta)
            model_source_path: Path to model source code (default: {PROJECT_ROOT}/experiments/experiments/{model_name})
            config_path: Path to config.yaml (default: "config.yaml")
            execution_command: Custom command to execute (if None, auto-built from config)
            mode: Execution mode (train, test, evaluate, predict)
            overrides: Dict of config overrides (e.g., {"epochs": 2, "batch_size": 64})
            gpu_id: GPU ID to use (sets CUDA_VISIBLE_DEVICES)
            dataset_type: Cold split dataset type (e.g., dataset_1_random, dataset_2_drug_blind, etc.)
            auto_report: Whether to automatically generate execution report (reserved for future use)

        Returns:
            JSON string with execution results:
            {
                "success": bool,
                "model_name": str,
                "container_id": str,
                "log_file_path": str,
                "config_summary": Dict,
                "command_used": str,
                "message": str
            }
        """
        try:
            # Get PROJECT_ROOT from environment
            import os
            project_root = os.getenv("PROJECT_ROOT", "/workspace")

            # Set default model source path with correct pattern
            if model_source_path is None:
                model_source_path = f"{project_root}/experiments/experiments/{model_name}"
            else:
                # '가상 주소'(/workspace)를 '실제 주소'(PROJECT_ROOT)로 변환
                if model_source_path.startswith("/workspace"):
                    model_source_path = model_source_path.replace("/workspace", project_root)
                # 이미 절대 경로면 그대로 사용

            self.logger.info(f"🚀 [EXECUTE_MODEL] Starting execution for {model_name}")
            self.logger.info(f"📁 [EXECUTE_MODEL] Model source path: {model_source_path}")

            # NEW: Step 0 - Config validation and command building
            config_summary = None
            command_used = execution_command

            if execution_command is None and config_path:
                # Auto-build command from config
                self.logger.info(f"📝 [EXECUTE_MODEL] Auto-building command from config: {config_path}")

                # Build full config path
                full_config_path = Path(model_source_path) / config_path

                # Validate config
                try:
                    validation_result = validate_config(str(full_config_path), project_root)
                    config_summary = validation_result["summary"]

                    self.logger.info(f"✅ [EXECUTE_MODEL] Config validation passed")
                    self.logger.info(f"   Model type: {config_summary.get('model_type', 'unknown')}")
                    self.logger.info(f"   Batch size: {config_summary.get('batch_size')}")
                    self.logger.info(f"   Epochs: {config_summary.get('epochs')}")

                    if validation_result["warnings"]:
                        self.logger.warning(f"⚠️  Config has {len(validation_result['warnings'])} warnings")

                except ConfigValidationError as e:
                    self.logger.error(f"❌ [EXECUTE_MODEL] Config validation failed: {e}")
                    return json.dumps({
                        "success": False,
                        "model_name": model_name,
                        "error": f"Config validation failed: {e}",
                        "error_type": "ConfigValidationError"
                    }, indent=2)

                # Build command
                command_used = build_execution_command(
                    model_name=model_name,
                    config_path=config_path,  # Relative path inside container
                    mode=mode,
                    overrides=overrides,
                    gpu_id=gpu_id,
                    dataset_type=dataset_type
                )

                self.logger.info(f"📜 [EXECUTE_MODEL] Built command: {command_used}")

            elif execution_command:
                self.logger.info(f"📜 [EXECUTE_MODEL] Using provided command: {execution_command}")
                command_used = execution_command

            # Step 1: Build environment and run container
            self.logger.info(f"🏗️  [EXECUTE_MODEL] Building environment for {model_name}...")
            container_id = build_and_run_container(model_source_path, model_name, gpu_id)
            self.logger.info(f"✅ [EXECUTE_MODEL] Container created: {container_id[:12]}")

            # Step 2: Execute model in container
            self.logger.info(f"🏃‍♂️ [EXECUTE_MODEL] Executing in container...")
            log_file_path = execute_in_container(container_id, model_name, command_used)
            self.logger.info(f"✅ [EXECUTE_MODEL] Execution completed. Log: {log_file_path}")

            # Return execution info with log file path for reporter node
            result = {
                "success": True,
                "model_name": model_name,
                "container_id": container_id,
                "log_file_path": log_file_path,
                "command_used": command_used,
                "message": "Execution completed successfully. Reporter node will analyze the log."
            }

            if config_summary:
                result["config_summary"] = config_summary

            return json.dumps(result, indent=2)

        except Exception as e:
            self.logger.error(f"❌ [EXECUTE_MODEL] Execution failed: {e}")

            return json.dumps({
                "success": False,
                "model_name": model_name,
                "error": str(e),
                "error_type": type(e).__name__
            }, indent=2)
