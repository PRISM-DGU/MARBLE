"""
DGDRP Preprocessing Validator

Validates that DGDRP preprocessing has been completed before model training.
Checks for required files and provides guidance if preprocessing is missing.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List

logger = logging.getLogger('DGDRP.Validator')


class DGDRPPreprocessingValidator:
    """
    Validates DGDRP preprocessing completion.

    Checks for:
    1. STRING PPI network file
    2. DTI info file
    3. Drug networks (1-stage output)
    4. Graph PyG files (2-stage output)
    5. Completion flag
    """

    # Default paths
    DEFAULT_INPUT_DIR = "/data1/project/20rak/MARBLE/input_raw_data"
    DEFAULT_N_INDIRECT_TARGETS = 20

    def __init__(
        self,
        input_dir: str = None,
        n_indirect_targets: int = None
    ):
        """
        Initialize validator.

        Args:
            input_dir: Input data directory (default: /data1/project/20rak/MARBLE/input_raw_data)
            n_indirect_targets: Number of indirect targets (default: 20)
        """
        self.input_dir = Path(input_dir or self.DEFAULT_INPUT_DIR)
        self.n_indirect_targets = n_indirect_targets or self.DEFAULT_N_INDIRECT_TARGETS

        logger.info(f"DGDRP Validator initialized")
        logger.info(f"  Input dir: {self.input_dir}")
        logger.info(f"  Indirect targets: {self.n_indirect_targets}")

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Run full validation.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check base directory
        if not self.input_dir.exists():
            errors.append(f"Input directory not found: {self.input_dir}")
            return False, errors

        logger.info("Checking preprocessing files...")

        # Check required input files
        required_inputs = [
            ('response_data_total.tsv', 'Response data'),
            ('expression_10k_genes_data_total.pkl', 'Expression data'),
            ('drug_data_total.pt', 'Drug fingerprints'),
            ('dgdrp_drug_target_profile.tsv', 'Drug target profile'),
            ('dgdrp_target_adjacency_matrix.tsv', 'Template adjacency'),
            ('dti_info_final_common_drugs_only.tsv', 'DTI info'),
            ('9606.protein.links.symbols.v11.5.txt', 'STRING PPI network'),
        ]

        for filename, description in required_inputs:
            filepath = self.input_dir / filename
            if not filepath.exists():
                errors.append(
                    f"Missing input file: {description} ({filename})"
                )
                logger.warning(f"  ✗ {description}: NOT FOUND")
            else:
                filesize = filepath.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"  ✓ {description}: {filesize:.1f}MB")

        # Check 1-stage output
        drug_networks_dir = self.input_dir / f'drug_networks_{self.n_indirect_targets}_indirect_targets'
        if not drug_networks_dir.exists():
            errors.append(
                f"1-stage preprocessing not completed\n"
                f"  Expected directory: {drug_networks_dir}\n"
                f"  Please run: bash scripts/preprocess_dgdrp.sh"
            )
            logger.warning(f"  ✗ Drug networks (1-stage): NOT FOUND")
        else:
            num_networks = len(list(drug_networks_dir.glob('*.tsv')))
            if num_networks > 0:
                logger.info(f"  ✓ Drug networks (1-stage): {num_networks} files")
            else:
                errors.append(
                    f"Drug networks directory empty: {drug_networks_dir}"
                )

        # Check 2-stage output
        graph_pyg_dir = self.input_dir / f'graph_pyg/{self.n_indirect_targets}_indirect_targets'
        if not graph_pyg_dir.exists():
            errors.append(
                f"2-stage preprocessing not completed\n"
                f"  Expected directory: {graph_pyg_dir}\n"
                f"  Please run: bash scripts/preprocess_dgdrp.sh"
            )
            logger.warning(f"  ✗ Graph PyG (2-stage): NOT FOUND")
        else:
            num_graphs = len(list(graph_pyg_dir.glob('*.pt')))
            if num_graphs > 0:
                logger.info(f"  ✓ Graph PyG (2-stage): {num_graphs} files")
            else:
                errors.append(
                    f"Graph PyG directory empty: {graph_pyg_dir}"
                )

        # Check completion flag
        completion_flag = self.input_dir / '.dgdrp_preprocessing_complete'
        if completion_flag.exists():
            logger.info(f"  ✓ Preprocessing completion flag found")
        else:
            logger.warning(f"  ⚠ Preprocessing completion flag not found")

        return len(errors) == 0, errors

    def get_status(self) -> str:
        """
        Get human-readable validation status.

        Returns:
            Status message
        """
        is_valid, errors = self.validate()

        if is_valid:
            return "✓ DGDRP preprocessing complete and valid"
        else:
            status = "✗ DGDRP preprocessing validation failed:\n"
            for error in errors:
                status += f"\n  - {error}"
            status += "\n\nTo complete preprocessing, run:"
            status += "\n  bash scripts/preprocess_dgdrp.sh"
            return status

    def print_status(self):
        """Print validation status to logger"""
        logger.info("\n" + "=" * 60)
        logger.info("DGDRP Preprocessing Validation")
        logger.info("=" * 60)

        is_valid, errors = self.validate()

        if is_valid:
            logger.info("✓ All preprocessing steps completed successfully!")
            logger.info("\nReady to train DGDRP model")
        else:
            logger.error("✗ Preprocessing validation failed:")
            for error in errors:
                logger.error(f"\n  {error}")
            logger.error("\nPlease complete preprocessing before training")

        logger.info("=" * 60)

        return is_valid


def validate_dgdrp_preprocessing(input_dir: str = None) -> bool:
    """
    Convenience function to validate DGDRP preprocessing.

    Args:
        input_dir: Input data directory

    Returns:
        True if validation passes, False otherwise
    """
    validator = DGDRPPreprocessingValidator(input_dir=input_dir)
    is_valid = validator.print_status()
    return is_valid


if __name__ == '__main__':
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run validation
    is_valid = validate_dgdrp_preprocessing()
    sys.exit(0 if is_valid else 1)
