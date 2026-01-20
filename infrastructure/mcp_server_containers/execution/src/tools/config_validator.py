"""
Config Validator for DRP Framework
Validates YAML config files before model execution
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for config validation errors"""
    pass


class ConfigValidator:
    """
    Validates DRP framework config files.

    Checks:
    - File existence and YAML syntax
    - Required fields (data, model, training)
    - Data file paths
    - Dimension consistency (if dimension_validation present)
    """

    REQUIRED_SECTIONS = ["data", "model", "training"]
    REQUIRED_TRAINING_FIELDS = ["batch_size", "learning_rate", "epochs"]

    def __init__(self, project_root: str = "/workspace"):
        """
        Initialize validator.

        Args:
            project_root: Project root path for resolving relative paths
        """
        self.project_root = Path(project_root)

    def validate(self, config_path: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate a config file.

        Args:
            config_path: Path to config.yaml file

        Returns:
            Tuple of (is_valid, warnings, config_dict)
            - is_valid: True if config passes all checks
            - warnings: List of warning messages (non-critical issues)
            - config_dict: Parsed config dictionary

        Raises:
            ConfigValidationError: If config has critical errors
        """
        warnings = []

        # Check 1: File existence
        config_file = Path(config_path)
        if not config_file.exists():
            raise ConfigValidationError(f"Config file not found: {config_path}")

        # Check 2: Load YAML
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML syntax: {e}")

        if not isinstance(config, dict):
            raise ConfigValidationError(f"Config must be a dictionary, got {type(config)}")

        # Check 3: Required sections
        missing_sections = [sec for sec in self.REQUIRED_SECTIONS if sec not in config]
        if missing_sections:
            raise ConfigValidationError(f"Missing required sections: {missing_sections}")

        # Check 4: Training fields
        training = config.get("training", {})
        missing_training = [field for field in self.REQUIRED_TRAINING_FIELDS
                           if field not in training]
        if missing_training:
            raise ConfigValidationError(f"Missing training fields: {missing_training}")

        # Check 5: Data paths (warning only)
        data_warnings = self._validate_data_paths(config.get("data", {}))
        warnings.extend(data_warnings)

        # Check 6: Dimension validation (if present)
        if "dimension_validation" in config:
            dim_warnings = self._validate_dimensions(config)
            warnings.extend(dim_warnings)

        logger.info(f"✅ Config validation passed: {config_path}")
        if warnings:
            logger.warning(f"⚠️  {len(warnings)} warnings found")
            for w in warnings:
                logger.warning(f"   - {w}")

        return True, warnings, config

    def _validate_data_paths(self, data_config: Dict[str, Any]) -> List[str]:
        """
        Validate data file paths in config.

        Returns:
            List of warning messages
        """
        warnings = []

        # Common data path fields
        path_fields = [
            "drug_file", "cell_file", "response_file",
            "train_file", "test_file", "val_file",
            "gene_expression_file", "drug_features_file",
            "drug_smiles_file", "cell_features_file"
        ]

        for field in path_fields:
            if field in data_config:
                path = data_config[field]
                if isinstance(path, str):
                    full_path = self.project_root / path
                    if not full_path.exists():
                        warnings.append(f"Data file not found: {path}")

        return warnings

    def _validate_dimensions(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate dimension consistency if dimension_validation present.

        Returns:
            List of warning messages
        """
        warnings = []

        try:
            dim_val = config.get("dimension_validation", {})
            if not dim_val.get("enabled", False):
                return warnings

            rules = dim_val.get("rules", [])
            if not rules:
                warnings.append("Dimension validation enabled but no rules defined")

            # Basic check: data_dimensions existence
            if "data_dimensions" in config:
                data_dims = config["data_dimensions"]
                if "raw" not in data_dims and "processed" not in data_dims:
                    warnings.append("data_dimensions should have 'raw' or 'processed' fields")

        except Exception as e:
            warnings.append(f"Could not validate dimensions: {e}")

        return warnings

    def get_config_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key information from config for logging.

        Args:
            config: Parsed config dictionary

        Returns:
            Summary dictionary with key config details
        """
        summary = {
            "model_type": "unknown",
            "batch_size": config.get("training", {}).get("batch_size"),
            "learning_rate": config.get("training", {}).get("learning_rate"),
            "epochs": config.get("training", {}).get("epochs"),
            "device": config.get("training", {}).get("device", "cpu"),
            "has_dimension_validation": "dimension_validation" in config,
        }

        # Infer model type from architecture
        model_config = config.get("model", {})
        if "drug_encoder" in model_config and "cell_encoder" in model_config:
            if "pathway_encoder" in model_config:
                summary["model_type"] = "deeptta"
            elif model_config.get("drug_encoder", {}).get("type") == "transformer":
                summary["model_type"] = "deeptta"
            elif "drug_similarity_encoder" in model_config:
                summary["model_type"] = "dgdrp"
            else:
                summary["model_type"] = "deepdr"

        return summary


def validate_config(config_path: str, project_root: str = "/workspace") -> Dict[str, Any]:
    """
    Convenience function to validate a config file.

    Args:
        config_path: Path to config.yaml
        project_root: Project root for path resolution

    Returns:
        Dictionary with validation results:
        {
            "valid": bool,
            "warnings": List[str],
            "config": Dict,
            "summary": Dict
        }

    Raises:
        ConfigValidationError: If config has critical errors
    """
    validator = ConfigValidator(project_root)
    is_valid, warnings, config = validator.validate(config_path)
    summary = validator.get_config_summary(config)

    return {
        "valid": is_valid,
        "warnings": warnings,
        "config": config,
        "summary": summary
    }
