"""DeepDR Decoder - DNN-based Fusion Module with Graph Pooling Support

DeepDR uses concatenation-based fusion followed by MLP layers
for drug response prediction. Supports both tensor and graph inputs.

Reference: DeepDR paper - FusionModule.DNN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union

try:
    from torch_geometric.nn import global_mean_pool, global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention for graph pooling (following DeepDR MHA)"""

    def __init__(self, cell_dim: int, drug_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = drug_dim // num_heads
        assert drug_dim % num_heads == 0, "drug_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(cell_dim, drug_dim)
        self.k_proj = nn.Linear(drug_dim, drug_dim)
        self.v_proj = nn.Linear(drug_dim, drug_dim)
        self.out_proj = nn.Linear(drug_dim, drug_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_ft: torch.Tensor, drug_input: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """
        Args:
            cell_ft: Cell features (batch_size, cell_dim)
            drug_input: Either tensor (batch_size, seq_len, drug_dim) or
                       tuple (node_features, graph_data) for graph input
        """
        if isinstance(drug_input, tuple):
            # Graph input: (node_features, graph_data)
            x, g = drug_input
            batch = g.batch

            # Aggregate per graph using attention
            batch_size = cell_ft.size(0)
            drug_dim = x.size(-1)

            # Get Q from cell features
            q = self.q_proj(cell_ft)  # (batch_size, drug_dim)
            q = q.unsqueeze(1)  # (batch_size, 1, drug_dim)

            # Process each graph in batch
            outputs = []
            for i in range(batch_size):
                mask = batch == i
                node_ft = x[mask]  # (num_nodes, drug_dim)
                if node_ft.size(0) == 0:
                    outputs.append(torch.zeros(drug_dim, device=x.device))
                    continue

                k = self.k_proj(node_ft)  # (num_nodes, drug_dim)
                v = self.v_proj(node_ft)  # (num_nodes, drug_dim)

                # Attention
                attn_weights = torch.matmul(q[i:i+1], k.t()) / (self.head_dim ** 0.5)
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_weights = self.dropout(attn_weights)

                out = torch.matmul(attn_weights, v)  # (1, drug_dim)
                outputs.append(out.squeeze(0))

            drug_ft = torch.stack(outputs, dim=0)  # (batch_size, drug_dim)
        else:
            # Tensor input: (batch_size, seq_len, drug_dim) or (batch_size, drug_dim)
            if len(drug_input.shape) == 2:
                return drug_input  # Already pooled

            batch_size, seq_len, drug_dim = drug_input.shape

            q = self.q_proj(cell_ft).unsqueeze(1)  # (batch_size, 1, drug_dim)
            k = self.k_proj(drug_input)  # (batch_size, seq_len, drug_dim)
            v = self.v_proj(drug_input)  # (batch_size, seq_len, drug_dim)

            # Scaled dot-product attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            drug_ft = torch.matmul(attn_weights, v).squeeze(1)  # (batch_size, drug_dim)

        return self.out_proj(drug_ft)


class Decoder(nn.Module):
    """DNN-based Decoder/Fusion Module with Graph Pooling (DeepDR optimal)

    Architecture: Pool(drug_emb) + Concat(drug, cell) -> MLP layers -> IC50 prediction

    Supports three pooling modes for graph drug features:
    - 'mean': Global mean pooling
    - 'max': Global max pooling
    - 'attention': Multi-head attention pooling (MHA)
    - 'mix': Combination of mean, max, and attention

    Config parameters (from model.decoder):
        architecture.fusion_dim: Dimension after fusion
        architecture.dropout: Dropout rate
        architecture.use_batch_norm: Whether to use batch normalization
        architecture.pool_type: Pooling type for graph features (mean/max/attention/mix)
        architecture.num_heads: Number of attention heads (for MHA pooling)

    Config parameters (from model.predictor):
        architecture.hidden_layers: Predictor hidden layers
        output.dim: Output dimension (1 for IC50)
    """

    def __init__(self, model_config: Dict[str, Any], drug_dim: int, cell_dim: int):
        super().__init__()

        self.drug_dim = drug_dim
        self.cell_dim = cell_dim

        # Get decoder/fusion config
        decoder_config = model_config.get('decoder', model_config.get('fusion', {}))
        arch = decoder_config.get('architecture', {})
        dropout_rate = arch.get('dropout', 0.1)
        use_batch_norm = arch.get('use_batch_norm', False)
        self.pool_type = arch.get('pool_type', 'mean')
        num_heads = arch.get('num_heads', 8)

        # Attention module for 'attention' and 'mix' pooling
        if self.pool_type in ['attention', 'mix']:
            self.attention = MultiHeadAttention(
                cell_dim=cell_dim,
                drug_dim=drug_dim,
                num_heads=num_heads,
                dropout=dropout_rate
            )

        concat_dim = drug_dim + cell_dim
        fusion_dim = arch.get('fusion_dim', concat_dim)

        # Input linear layers for drug and cell
        self.input_drug = nn.Linear(drug_dim, drug_dim)
        self.input_cell = nn.Linear(cell_dim, cell_dim)

        # Optional projection layer if fusion_dim differs
        self.concat_mode = arch.get('concat', True)
        if self.concat_mode:
            proj_input_dim = drug_dim + cell_dim
        else:
            # Add mode: project both to same dimension then add
            proj_input_dim = drug_dim

        if fusion_dim != proj_input_dim:
            proj_layers = [nn.Linear(proj_input_dim, fusion_dim)]
            if use_batch_norm:
                proj_layers.append(nn.BatchNorm1d(fusion_dim))
            proj_layers.extend([nn.ReLU(), nn.Dropout(dropout_rate)])
            self.projection = nn.Sequential(*proj_layers)
        else:
            self.projection = None
            fusion_dim = proj_input_dim

        # Get predictor config
        predictor_config = model_config.get('predictor', {})
        pred_arch = predictor_config.get('architecture', {})
        hidden_layers = pred_arch.get('hidden_layers', [512, 256, 128])
        if isinstance(hidden_layers, list) and len(hidden_layers) > 0:
            if isinstance(hidden_layers[0], dict):
                hidden_layers = [layer['dim'] for layer in hidden_layers]

        pred_dropout = pred_arch.get('dropout', 0.1)
        self.output_dim = predictor_config.get('output', {}).get('dim', 1)

        # Build predictor layers (following DeepDR FusionModule.DNN)
        layer_dims = [fusion_dim] + hidden_layers
        self.encode_dnn = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(p=pred_dropout)
            for _ in range(len(layer_dims) - 1)
        ])
        self.output = nn.Linear(layer_dims[-1], self.output_dim)

    def get_output_dim(self) -> int:
        return self.output_dim

    def _pool_graph_features(self, drug_input: Union[torch.Tensor, Tuple],
                             cell_ft: torch.Tensor) -> torch.Tensor:
        """Pool graph features to fixed-size vector

        Args:
            drug_input: Either tensor or tuple (node_features, graph_data)
            cell_ft: Cell features for attention pooling

        Returns:
            Pooled drug features (batch_size, drug_dim)
        """
        if isinstance(drug_input, torch.Tensor):
            # Already a tensor
            if len(drug_input.shape) == 3:
                # Sequence of vectors: (batch, seq, dim)
                if self.pool_type == 'mean':
                    return torch.mean(drug_input, dim=1)
                elif self.pool_type == 'max':
                    return torch.max(drug_input, dim=1)[0]
                elif self.pool_type == 'attention':
                    return self.attention(cell_ft, drug_input)
                else:  # mix
                    mean_pool = torch.mean(drug_input, dim=1)
                    max_pool = torch.max(drug_input, dim=1)[0]
                    attn_pool = self.attention(cell_ft, drug_input)
                    return mean_pool + max_pool + attn_pool
            else:
                # Already pooled: (batch, dim)
                return drug_input
        else:
            # Graph input: (node_features, graph_data)
            x, g = drug_input
            batch = g.batch

            if not HAS_PYG:
                raise ImportError("torch_geometric required for graph pooling")

            if self.pool_type == 'mean':
                return global_mean_pool(x, batch)
            elif self.pool_type == 'max':
                return global_max_pool(x, batch)
            elif self.pool_type == 'attention':
                return self.attention(cell_ft, drug_input)
            else:  # mix
                mean_pool = global_mean_pool(x, batch)
                max_pool = global_max_pool(x, batch)
                attn_pool = self.attention(cell_ft, drug_input)
                return mean_pool + max_pool + attn_pool

    def forward(self, drug_encoded: Union[torch.Tensor, Tuple],
                cell_encoded: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass

        Args:
            drug_encoded: Drug embedding - either tensor (batch_size, drug_dim)
                         or tuple (node_features, graph_data) from graph encoder
            cell_encoded: Cell embedding from cell encoder (batch_size, cell_dim)

        Returns:
            IC50 prediction (batch_size, 1)
        """
        # Pool graph features if needed
        drug_ft = self._pool_graph_features(drug_encoded, cell_encoded)
        cell_ft = cell_encoded

        # Input projections
        drug_ft = F.relu(self.input_drug(drug_ft))
        cell_ft = F.relu(self.input_cell(cell_ft))

        # Fusion
        if self.concat_mode:
            fused = torch.cat([cell_ft, drug_ft], dim=1)
        else:
            fused = cell_ft + drug_ft

        # Optional projection
        if self.projection is not None:
            fused = self.projection(fused)

        # Pass through predictor layers
        x = fused
        for linear, dropout in zip(self.encode_dnn, self.dropout_layers):
            x = F.relu(linear(x))
            x = dropout(x)

        # Output layer
        x = self.output(x)

        return x
