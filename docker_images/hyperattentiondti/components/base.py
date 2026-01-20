"""Base Classes for HyperAttentionDTI components

All encoders/decoders should inherit from these base classes.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDrugEncoder(nn.Module, ABC):
    """Drug Encoder base interface (processes SMILES strings)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.output_dim: int = 0

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return output dimension"""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor (batch_size, seq_len)

        Returns:
            Encoded drug representation
        """
        pass


class BaseProteinEncoder(nn.Module, ABC):
    """Protein/Target Encoder base interface (processes protein sequences)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.output_dim: int = 0

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return output dimension"""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor (batch_size, seq_len)

        Returns:
            Encoded protein representation
        """
        pass


class BaseDecoder(nn.Module, ABC):
    """Decoder base interface"""

    def __init__(self, config: Dict[str, Any], drug_dim: int, protein_dim: int):
        super().__init__()
        self.config = config
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.output_dim: int = 2

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return output dimension"""
        pass

    @abstractmethod
    def forward(self, drug_encoded: torch.Tensor, protein_encoded: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            drug_encoded: Encoded drug
            protein_encoded: Encoded protein

        Returns:
            Prediction (batch_size, num_classes)
        """
        pass
