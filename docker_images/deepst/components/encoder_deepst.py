import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm
from typing import Dict, List

from .registry import register_encoder
from .base import BaseEncoder


@register_encoder('encoder_deepst')
class Encoder(BaseEncoder):
    def __init__(self, config: Dict):
        super(Encoder, self).__init__()

        self.in_dim = config['in_dim']
        self.hidden_dims: List[int] = config.get('hidden_dims', [32, 20])
        self.dropout = config.get('dropout', 0.01)
        self.activate = config.get('activate', 'elu')

        self.layers = nn.Sequential()
        current_dim = self.in_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            self.layers.add_module(
                f'encoder_L{i}',
                self._build_block(current_dim, hidden_dim)
            )
            current_dim = hidden_dim

        self.out_dim = self.hidden_dims[-1]

    def _build_block(self, in_features: int, out_features: int) -> nn.Sequential:
        layers = [
            nn.Linear(in_features, out_features),
            BatchNorm(out_features, momentum=0.01, eps=0.001)
        ]

        if self.activate == 'elu':
            layers.append(nn.ELU())
        elif self.activate == 'relu':
            layers.append(nn.ReLU())
        elif self.activate == 'sigmoid':
            layers.append(nn.Sigmoid())

        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layers(x)

    def get_output_dim(self) -> int:
        return self.out_dim
