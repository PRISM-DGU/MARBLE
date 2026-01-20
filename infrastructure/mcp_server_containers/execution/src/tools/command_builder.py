"""
Command Builder for DRP Framework
Generates execution commands from config files and overrides
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class CommandBuilder:
    """
    Builds execution commands for DRP framework models.

    Generates commands like:
    python main.py --config config.yaml --mode train --epochs 10 --batch-size 64
    """

    # Model-specific script patterns
    MODEL_SCRIPTS = {
        "deepdr": "main.py",
        "deeptta": "main.py",
        "dgdrp": "main.py",
        "deeptta": "main.py",
        "dipk": "main.py",
    }

    # Common override mappings (CLI arg → config field)
    OVERRIDE_MAPPINGS = {
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "learning_rate": "--lr",
        "device": "--device",
        "seed": "--seed",
    }

    def __init__(self, model_name: str):
        """
        Initialize command builder.

        Args:
            model_name: Model name (deepdr, deeptta, dgdrp, deeptta)
        """
        self.model_name = model_name.lower()
        if self.model_name not in self.MODEL_SCRIPTS:
            raise ValueError(f"Unknown model: {model_name}")

    def build_command(
        self,
        config_path: str,
        mode: str = "train",
        overrides: Optional[Dict[str, Any]] = None,
        gpu_id: Optional[int] = None,
        dataset_type: Optional[str] = None
    ) -> str:
        """
        Build execution command.

        Args:
            config_path: Path to config.yaml (relative to model directory)
            mode: Execution mode (train, test, evaluate, predict)
            overrides: Dict of config overrides (e.g., {"epochs": 10, "batch_size": 64})
            gpu_id: GPU ID to use (sets CUDA_VISIBLE_DEVICES)
            dataset_type: Cold split dataset type (e.g., dataset_1_random, dataset_2_drug_blind)

        Returns:
            Full command string ready for execution

        Example:
            >>> builder = CommandBuilder("deepdr")
            >>> cmd = builder.build_command("config.yaml", "train", {"epochs": 2})
            >>> print(cmd)
            "python main.py --config config.yaml --epochs 2"
        """
        overrides = overrides or {}

        # Start with base command
        script = self.MODEL_SCRIPTS[self.model_name]
        parts = ["python", script]

        # Add config path
        parts.extend(["--config", config_path])

        # Add dataset type if specified
        if dataset_type:
            parts.extend(["--dataset-type", dataset_type])

        # Add overrides
        for key, value in overrides.items():
            cli_arg = self.OVERRIDE_MAPPINGS.get(key, f"--{key.replace('_', '-')}")
            parts.append(cli_arg)
            parts.append(str(value))

        # Build full command
        command = " ".join(parts)

        # Prepend GPU environment variable if specified
        if gpu_id is not None:
            command = f"export CUDA_VISIBLE_DEVICES={gpu_id} && {command}"

        logger.info(f"📝 Built command: {command}")
        return command

    def build_from_config(
        self,
        config_dict: Dict[str, Any],
        config_path: str = "config.yaml",
        mode: str = "train",
        override_epochs: Optional[int] = None,
        override_batch_size: Optional[int] = None,
        gpu_id: Optional[int] = None
    ) -> str:
        """
        Build command with config-aware defaults.

        Args:
            config_dict: Parsed config dictionary
            config_path: Path to config file
            mode: Execution mode
            override_epochs: Override epochs (for quick testing)
            override_batch_size: Override batch size
            gpu_id: GPU ID

        Returns:
            Full command string
        """
        overrides = {}

        # Apply explicit overrides
        if override_epochs is not None:
            overrides["epochs"] = override_epochs
        if override_batch_size is not None:
            overrides["batch_size"] = override_batch_size

        return self.build_command(config_path, mode, overrides, gpu_id)

    @staticmethod
    def parse_override_string(override_str: str) -> Dict[str, Any]:
        """
        Parse override string into dictionary.

        Args:
            override_str: String like "epochs=10,batch_size=64,lr=0.001"

        Returns:
            Dictionary of overrides

        Example:
            >>> CommandBuilder.parse_override_string("epochs=10,batch_size=64")
            {"epochs": 10, "batch_size": 64}
        """
        overrides = {}

        if not override_str:
            return overrides

        for pair in override_str.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue

            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Try to convert to appropriate type
            try:
                # Try int
                overrides[key] = int(value)
            except ValueError:
                try:
                    # Try float
                    overrides[key] = float(value)
                except ValueError:
                    # Keep as string
                    overrides[key] = value

        return overrides

    def validate_mode(self, mode: str) -> bool:
        """
        Validate execution mode.

        Args:
            mode: Mode string

        Returns:
            True if valid mode
        """
        valid_modes = ["train", "test", "evaluate", "predict", "inference"]
        return mode.lower() in valid_modes

    def get_log_file_path(
        self,
        model_name: str,
        mode: str,
        timestamp: Optional[str] = None
    ) -> str:
        """
        Generate log file path for execution.

        Args:
            model_name: Model name
            mode: Execution mode
            timestamp: Optional timestamp string

        Returns:
            Log file path relative to model directory
        """
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"logs/{model_name}_{mode}_{timestamp}.log"


def build_execution_command(
    model_name: str,
    config_path: str = "config.yaml",
    mode: str = "train",
    overrides: Optional[Dict[str, Any]] = None,
    gpu_id: Optional[int] = None,
    dataset_type: Optional[str] = None
) -> str:
    """
    Convenience function to build execution command.

    Args:
        model_name: Model name (deepdr, deeptta, dgdrp, deeptta)
        config_path: Path to config.yaml
        mode: Execution mode (train, test, etc.)
        overrides: Dict of config overrides
        gpu_id: GPU ID to use
        dataset_type: Cold split dataset type (e.g., dataset_1_random, dataset_2_drug_blind)

    Returns:
        Full command string

    Example:
        >>> cmd = build_execution_command("deepdr", "config.yaml", "train", {"epochs": 2})
        >>> print(cmd)
        "python main.py --config config.yaml --epochs 2"
    """
    builder = CommandBuilder(model_name)
    return builder.build_command(config_path, mode, overrides, gpu_id, dataset_type)
