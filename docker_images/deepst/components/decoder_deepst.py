import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from typing import Dict, List

from .registry import register_decoder
from .base import BaseDecoder


@register_decoder('decoder_deepst')
class Decoder(BaseDecoder):
    def __init__(self, config: Dict):
        super(Decoder, self).__init__()

        self.in_dim = config['in_dim']
        self.hidden_dims: List[int] = config.get('hidden_dims', [32])
        self.out_dim = config['out_dim']
        self.dropout = config.get('dropout', 0.01)
        self.activate = config.get('activate', 'elu')

        self.layers = nn.Sequential()
        current_dim = self.in_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            self.layers.add_module(
                f'decoder_L{i}',
                self._build_block(current_dim, hidden_dim)
            )
            current_dim = hidden_dim

        self.layers.add_module(
            'decoder_out',
            nn.Linear(current_dim, self.out_dim)
        )

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

    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layers(z)

    def get_output_dim(self) -> int:
        return self.out_dim


class InnerProductDecoder(nn.Module):
    def __init__(self, dropout: float = 0.0, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.dropout(z, self.dropout, training=self.training)
        return self.act(torch.mm(z, z.t()))
