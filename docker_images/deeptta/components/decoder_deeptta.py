"""DeepTTA MLP Decoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class Decoder(nn.Module):
    """Simple MLP based Decoder (DeepTTA style)"""

    def __init__(self, model_config: Dict, drug_dim: int, cell_dim: int):
        super().__init__()

        decoder_config = model_config.get('decoder', model_config.get('fusion', {}))
        arch = decoder_config.get('architecture', {})
        dropout_rate = arch.get('dropout', 0.1)

        concat_dim = drug_dim + cell_dim
        fusion_dim = arch.get('fusion_dim', concat_dim)

        if fusion_dim != concat_dim:
            self.projection = nn.Sequential(
                nn.Linear(concat_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        else:
            self.projection = None
            fusion_dim = concat_dim

        predictor_config = model_config.get('predictor', {})
        arch_config = predictor_config.get('architecture', {})
        hidden_layers = arch_config.get('hidden_layers', [128, 64])
        pred_dropout = arch_config.get('dropout', 0.1)
        self.output_dim = predictor_config.get('output', {}).get('dim', 1)

        self.dropout = nn.Dropout(pred_dropout)
        layer_dims = [fusion_dim] + hidden_layers + [self.output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, drug_encoded: torch.Tensor, cell_encoded: torch.Tensor, **kwargs) -> torch.Tensor:
        fused = torch.cat([drug_encoded, cell_encoded], dim=1)

        if self.projection is not None:
            fused = self.projection(fused)

        x = fused
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x)
            else:
                x = F.relu(self.dropout(layer(x)))
        return x
