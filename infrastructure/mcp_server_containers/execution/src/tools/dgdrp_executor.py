"""
DGDRP-specific Execution Wrapper

Pre-validates DGDRP preprocessing before executing the model in Docker.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add workspace to path
sys.path.insert(0, '/workspace')

from validators.dgdrp_validator import DGDRPPreprocessingValidator


class DGDRPExecutor:
    """
    DGDRP execution wrapper with preprocessing validation.

    Ensures all required DGDRP preprocessing files exist before
    attempting to train the model.
    """

    def __init__(self):
        """Initialize DGDRP executor."""
        self.logger = logging.getLogger('DGDRP.Executor')
        self.validator = DGDRPPreprocessingValidator()

    def validate_preprocessing(self) -> tuple[bool, Optional[str]]:
        """
        Validate DGDRP preprocessing completion.

        Returns:
            Tuple of (is_valid, error_message)
        """
        self.logger.info("Validating DGDRP preprocessing...")
        is_valid, errors = self.validator.validate()

        if is_valid:
            self.logger.info("✓ DGDRP preprocessing validation passed")
            return True, None
        else:
            error_msg = "DGDRP preprocessing validation failed:\n"
            for error in errors:
                error_msg += f"  - {error}\n"
            error_msg += "\nTo complete preprocessing, run:\n"
            error_msg += "  bash /data1/project/20rak/MARBLE/scripts/preprocess_dgdrp.sh"
            self.logger.error(error_msg)
            return False, error_msg

    def prepare_execution_result(
        self,
        success: bool,
        message: str,
        container_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare execution result JSON.

        Args:
            success: Whether execution was successful
            message: Result message
            container_info: Container execution information

        Returns:
            JSON string with result
        """
        result = {
            "success": success,
            "model_name": "dgdrp",
            "message": message
        }

        if container_info:
            result.update(container_info)

        return json.dumps(result, indent=2)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    executor = DGDRPExecutor()
    is_valid, error = executor.validate_preprocessing()
    sys.exit(0 if is_valid else 1)
