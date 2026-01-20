import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseEncoder(nn.Module, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class BaseDecoder(nn.Module, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
