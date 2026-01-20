"""
RDKit Handler with Error Handling and Fallback
Safely handles molecular structure operations
"""

import logging
from typing import Optional, Tuple
from io import BytesIO

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

import matplotlib.pyplot as plt
from PIL import Image

logger = logging.getLogger(__name__)


class RDKitHandler:
    """Wrapper for RDKit operations with error handling"""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit is not available. Molecular visualizations will use fallback.")
        self.rdkit_available = RDKIT_AVAILABLE

    def smiles_to_mol(self, smiles: str) -> Optional:
        """
        Convert SMILES string to RDKit Mol object

        Args:
            smiles: SMILES notation string

        Returns:
            RDKit Mol object or None if invalid
        """
        if not self.rdkit_available:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES string: {smiles}")
            return mol
        except Exception as e:
            logger.error(f"Error converting SMILES: {e}")
            return None

    def draw_molecule(self, smiles: str, size: Tuple[int, int] = (300, 300)) -> Optional[Image.Image]:
        """
        Draw molecule structure from SMILES

        Args:
            smiles: SMILES notation string
            size: Image size (width, height)

        Returns:
            PIL Image or None
        """
        if not self.rdkit_available:
            return self._create_fallback_image(smiles, size)

        try:
            mol = self.smiles_to_mol(smiles)
            if mol is None:
                return self._create_fallback_image(smiles, size)

            # Draw molecule with RDKit
            img = Draw.MolToImage(mol, size=size)
            return img

        except Exception as e:
            logger.error(f"Error drawing molecule: {e}")
            return self._create_fallback_image(smiles, size)

    def calculate_descriptors(self, smiles: str) -> dict:
        """
        Calculate molecular descriptors from SMILES

        Args:
            smiles: SMILES notation string

        Returns:
            Dictionary of descriptors (MW, LogP, TPSA, etc.)
        """
        if not self.rdkit_available:
            return {"error": "RDKit not available"}

        try:
            mol = self.smiles_to_mol(smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}

            return {
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "logP": round(Descriptors.MolLogP(mol), 2),
                "tpsa": round(Descriptors.TPSA(mol), 2),
                "num_h_donors": Descriptors.NumHDonors(mol),
                "num_h_acceptors": Descriptors.NumHAcceptors(mol),
                "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            }

        except Exception as e:
            logger.error(f"Error calculating descriptors: {e}")
            return {"error": str(e)}

    def _create_fallback_image(self, smiles: str, size: Tuple[int, int]) -> Image.Image:
        """
        Create fallback image when RDKit fails

        Args:
            smiles: SMILES string to display as text
            size: Image size

        Returns:
            PIL Image with SMILES text
        """
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
        ax.axis('off')

        # Display SMILES as text
        ax.text(0.5, 0.5, f"SMILES:\n{smiles}",
                ha='center', va='center',
                fontsize=10, wrap=True,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Convert to PIL Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf)

    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate if SMILES string is valid

        Args:
            smiles: SMILES notation string

        Returns:
            True if valid, False otherwise
        """
        if not self.rdkit_available:
            return False

        mol = self.smiles_to_mol(smiles)
        return mol is not None
