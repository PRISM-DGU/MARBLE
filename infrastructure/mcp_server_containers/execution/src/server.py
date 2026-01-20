#!/usr/bin/env python3
"""
Execution MCP Server
Unified MCP server for drug response prediction model execution
"""

import asyncio
import logging
import sys

# Add workspace to Python path for imports
sys.path.insert(0, '/workspace')

try:
    from fastmcp import FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from tools.execute_model import ExecuteModelTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Execution Server")

# Initialize tool instance
execute_tool = ExecuteModelTool()


@mcp.tool()
async def execute_model(
    model_name: str,
    model_source_path: str | None = None,
    config_path: str | None = "config.yaml",
    execution_command: str | None = None,
    mode: str = "train",
    overrides: dict | None = None,
    gpu_id: int | None = None,
    auto_report: bool = True
) -> str:
    """
    Execute drug response prediction model in isolated Docker container.

    This tool orchestrates the complete execution workflow:
    1. Builds isolated Docker environment for the model
    2. Executes training/evaluation command in container
    3. Generates structured execution report with Pydantic validation

    Args:
        model_name: Model identifier. Supported models:
            - "deeptta": DeepTTA model
            - "deepdr": DeepDR model
            - "dgdrp": DG-DRP model
            - "deeptta": DeepTTA model
        model_source_path: Absolute path to model source code directory.
            Default: /workspace/experiments/experiments/{model_name}
        config_path: Path to config.yaml file (relative to model_source_path).
            Default: "config.yaml"
        execution_command: Shell command to execute inside container.
            If None, command is auto-built from config.
            Examples:
            - "python train.py --epochs 100 --batch-size 128"
            - "python main.py --mode train --dataset 2369joint"
            - "python evaluate.py --checkpoint best_model.pth"
        mode: Execution mode (train, test, evaluate, predict).
            Default: "train"
        overrides: Dict of config overrides (e.g., {"epochs": 2, "batch_size": 64}).
            Default: None
        gpu_id: GPU ID to use for training (sets CUDA_VISIBLE_DEVICES).
            If None, runs on CPU.
            Examples: 0, 1, 2, 3
        auto_report: Whether to automatically generate execution report.
            If True: Returns full execution report with metrics and analysis
            If False: Returns basic execution info (container ID, log path)
            Default: True

    Returns:
        JSON string containing execution results:

        Success case (auto_report=True):
        {
            "success": true,
            "model_name": "deeptta",
            "container_id": "abc123def456",
            "log_file_path": "/workspace/execution_logs/deeptta_20250124_143022.log",
            "execution_report": {
                "execution_success": true,
                "metrics": {
                    "accuracy": 0.85,
                    "mse": 0.12,
                    "r2": 0.78
                },
                "summary": "Model training completed successfully with 85% accuracy",
                "error_message": null
            }
        }

        Success case (auto_report=False):
        {
            "success": true,
            "model_name": "deeptta",
            "container_id": "abc123def456",
            "log_file_path": "/workspace/execution_logs/deeptta_20250124_143022.log",
            "message": "Execution completed successfully. Use reporter tool for detailed analysis."
        }

        Failure case:
        {
            "success": false,
            "model_name": "deeptta",
            "error": "Container build failed: Dockerfile not found",
            "error_type": "FileNotFoundError"
        }

    Example Usage:
        # Basic execution with automatic reporting
        result = execute_model(
            model_name="deeptta",
            execution_command="python main.py --mode train"
        )

        # Execution without automatic report
        result = execute_model(
            model_name="deepdr",
            model_source_path="/workspace/models/DeepDR",
            execution_command="python train.py --epochs 200",
            auto_report=False
        )

        # Evaluation mode
        result = execute_model(
            model_name="dgdrp",
            execution_command="python evaluate.py --checkpoint best.pth"
        )

    Notes:
        - Each execution creates an isolated Docker container
        - Containers are automatically removed after execution (--rm flag)
        - Execution logs are saved to /workspace/execution_logs/
        - Reporter uses Pydantic schema for structured output validation
        - GPU support is automatically enabled if available
    """
    try:
        logger.info(f"🔌 [MCP] Received execute_model request for {model_name}")
        if gpu_id is not None:
            logger.info(f"   GPU ID: {gpu_id}")

        result = await execute_tool.apply(
            model_name=model_name,
            model_source_path=model_source_path,
            config_path=config_path,
            execution_command=execution_command,
            mode=mode,
            overrides=overrides,
            gpu_id=gpu_id,
            auto_report=auto_report
        )

        logger.info(f"✅ [MCP] Execution completed for {model_name}")
        return result

    except Exception as e:
        logger.error(f"❌ [MCP] Execution tool error: {e}")
        import json
        return json.dumps({
            "success": False,
            "model_name": model_name,
            "error": str(e),
            "error_type": type(e).__name__
        }, indent=2)


def main():
    """Main server entry point."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Execution MCP Server")
    logger.info("=" * 60)
    logger.info("Server: Execution Server")
    logger.info("Tools: execute_model")
    logger.info("Transport: stdio")
    logger.info("=" * 60)

    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()
