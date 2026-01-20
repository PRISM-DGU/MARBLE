"""Base classes for DeepDR components"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDrugEncoder(nn.Module, ABC):
    """Base class for drug encoders"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, x, **kwargs) -> torch.Tensor:
        pass


class BaseCellEncoder(nn.Module, ABC):
    """Base class for cell encoders"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, x, **kwargs) -> torch.Tensor:
        pass


class BaseDecoder(nn.Module, ABC):
    """Base class for decoders (fusion modules)"""

    def __init__(self, config: Dict[str, Any], drug_dim: int, cell_dim: int):
        super().__init__()
        self.config = config
        self.drug_dim = drug_dim
        self.cell_dim = cell_dim

    @abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, drug_encoded: torch.Tensor, cell_encoded: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
