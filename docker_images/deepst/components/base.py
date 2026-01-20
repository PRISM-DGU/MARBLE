from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict


class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        pass


class BaseGraphEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, edge_index, **kwargs):
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        pass


class BaseDecoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, z, **kwargs):
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        pass
