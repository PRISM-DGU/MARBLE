import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from typing import Dict, Optional
from torch_geometric.typing import Adj, Size, OptTensor
from torch import Tensor

class Decoder(nn.Module):
    def __init__(self, config: Dict):
        super(Decoder, self).__init__()

        self.in_dim = config['in_dim']
        self.hidden_dim = config['hidden_dim']
        self.out_dim = config['out_dim']
        self.heads = config.get('heads', 1)
        self.dropout = config.get('dropout', 0.0)

        self.conv1 = GATConv(self.in_dim, self.hidden_dim, heads=self.heads, concat=False, dropout=self.dropout, add_self_loops=False)
        self.conv2 = GATConv(self.hidden_dim, self.out_dim, heads=self.heads, concat=False, dropout=self.dropout, add_self_loops=False)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor = None, encoder: nn.Module = None, **kwargs) -> torch.Tensor:
        if encoder is not None and hasattr(encoder, 'conv2') and hasattr(encoder, 'conv1'):
            self.conv1.lin_src.data = encoder.conv2.lin_src.transpose(0, 1)
            self.conv1.lin_dst.data = encoder.conv2.lin_dst.transpose(0, 1)
            self.conv2.lin_src.data = encoder.conv1.lin_src.transpose(0, 1)
            self.conv2.lin_dst.data = encoder.conv1.lin_dst.transpose(0, 1)

            attention_weights = encoder.get_attention_weights() if hasattr(encoder, 'get_attention_weights') else None
            h1 = F.elu(self.conv1(z, edge_index, attention=True, tied_attention=attention_weights))
            h2 = self.conv2(h1, edge_index, attention=False)
        else:
            h1 = F.elu(self.conv1(z, edge_index))
            h2 = self.conv2(h1, edge_index, attention=False)

        return h2

    def get_output_dim(self) -> int:
        return self.out_dim
    
class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        self._alpha = None

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None, attention: bool = True, tied_attention=None):
        H, C = self.heads, self.out_channels
        x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)

        if tied_attention is None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
        else:
            alpha = tied_attention

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

