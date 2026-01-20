"""DeepST - Encoder Template

This file is a template for LLM (code expert) to implement code based on research papers.

[LLM CODING RULES]
1. Class name MUST remain 'Encoder' - DO NOT CHANGE
2. MUST inherit from BaseEncoder
3. MUST implement: __init__, get_output_dim, forward methods
4. forward input: torch.Tensor - gene expression features
5. forward output: torch.Tensor - encoded representations
6. Read architecture parameters from config

[Config Example - model.encoder section in config.yaml]
encoder:
  type: encoder_other
  in_dim: 3000
  hidden_dims: [32, 20]
  dropout: 0.01
  activate: elu
"""

import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm
from typing import Dict, List

from .registry import register_encoder
from .base import BaseEncoder


# ==============================================================================
# AUXILIARY MODULES - LLM can add helper classes here
# ==============================================================================




# ==============================================================================
# MAIN ENCODER CLASS - LLM MUST IMPLEMENT
# ==============================================================================
@register_encoder('encoder_other')
class Encoder(BaseEncoder):
    """Encoder for Spatial Transcriptomics

    [LLM IMPLEMENTATION AREA]
    - Implement the encoder architecture proposed in the paper
    - MUST define self.out_dim

    Args:
        config: encoder configuration dictionary
    """

    def __init__(self, config: Dict):
        super(Encoder, self).__init__()

        self.in_dim = config['in_dim']
        self.hidden_dims: List[int] = config.get('hidden_dims', [32, 20])
        self.dropout = config.get('dropout', 0.01)
        self.activate = config.get('activate', 'elu')

        # ============================================================
        # [LLM IMPLEMENTATION AREA]
        # 1. Define self.out_dim
        # 2. Define encoder layers (Linear, BatchNorm, etc.)
        # ============================================================

        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input features (batch_size, in_dim)
            **kwargs: Additional arguments

        Returns:
            Encoded representations (batch_size, out_dim)
        """
        # ============================================================
        # [LLM IMPLEMENTATION AREA]
        # Process input through encoder layers
        # ============================================================

        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")

    def get_output_dim(self) -> int:
        return self.out_dim
