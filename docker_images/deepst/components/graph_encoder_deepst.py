import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, BatchNorm
from typing import Dict

from .registry import register_graph_encoder
from .base import BaseGraphEncoder


def _get_conv_class(conv_type: str):
    from torch_geometric.nn import (
        GCNConv, GATConv, SAGEConv, GraphConv, GatedGraphConv,
        ResGatedGraphConv, TransformerConv, TAGConv, ARMAConv,
        SGConv, MFConv, LEConv, ClusterGCNConv
    )

    conv_map = {
        'GCNConv': GCNConv,
        'GATConv': GATConv,
        'SAGEConv': SAGEConv,
        'GraphConv': GraphConv,
        'GatedGraphConv': GatedGraphConv,
        'ResGatedGraphConv': ResGatedGraphConv,
        'TransformerConv': TransformerConv,
        'TAGConv': TAGConv,
        'ARMAConv': ARMAConv,
        'SGConv': SGConv,
        'MFConv': MFConv,
        'LEConv': LEConv,
        'ClusterGCNConv': ClusterGCNConv,
    }

    if conv_type not in conv_map:
        raise ValueError(f"Unknown conv type: {conv_type}. Available: {list(conv_map.keys())}")

    return conv_map[conv_type]


@register_graph_encoder('graph_encoder_deepst')
class GraphEncoder(BaseGraphEncoder):
    def __init__(self, config: Dict):
        super(GraphEncoder, self).__init__()

        self.in_dim = config['in_dim']
        self.hidden_dim = config.get('hidden_dim', 64)
        self.out_dim = config.get('out_dim', 8)
        self.conv_type = config.get('conv_type', 'GATConv')
        self.dropout = config.get('dropout', 0.01)

        ConvClass = _get_conv_class(self.conv_type)

        self.conv = Sequential('x, edge_index', [
            (ConvClass(self.in_dim, self.hidden_dim), 'x, edge_index -> x'),
            BatchNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
        ])

        self.conv_mean = Sequential('x, edge_index', [
            (ConvClass(self.hidden_dim, self.out_dim), 'x, edge_index -> x')
        ])

        self.conv_logvar = Sequential('x, edge_index', [
            (ConvClass(self.hidden_dim, self.out_dim), 'x, edge_index -> x')
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> tuple:
        conv_x = self.conv(x, edge_index)
        mu = self.conv_mean(conv_x, edge_index)
        logvar = self.conv_logvar(conv_x, edge_index)
        return mu, logvar

    def get_output_dim(self) -> int:
        return self.out_dim
