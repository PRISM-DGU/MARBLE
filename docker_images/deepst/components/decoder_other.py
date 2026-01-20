"""DeepST - Decoder Template

This file is a template for LLM (code expert) to implement code based on research papers.

[LLM CODING RULES]
1. Class name MUST remain 'Decoder' - DO NOT CHANGE
2. MUST inherit from BaseDecoder
3. MUST implement: __init__, get_output_dim, forward methods
4. forward input: torch.Tensor - encoded representations
5. forward output: torch.Tensor - decoded/reconstructed output
6. Read architecture parameters from config

[Config Example - model.decoder section in config.yaml]
decoder:
  type: decoder_other
  in_dim: 20
  hidden_dims: [32]
  out_dim: 3000
  dropout: 0.01
  activate: elu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from typing import Dict, List

from .registry import register_decoder
from .base import BaseDecoder


# ==============================================================================
# AUXILIARY MODULES - LLM can add helper classes here
# ==============================================================================




# ==============================================================================
# MAIN DECODER CLASS - LLM MUST IMPLEMENT
# ==============================================================================
@register_decoder('decoder_other')
class Decoder(BaseDecoder):
    """Decoder for Spatial Transcriptomics

    [LLM IMPLEMENTATION AREA]
    - Implement the decoder architecture proposed in the paper
    - Reconstruct original gene expression from latent representations

    Args:
        config: decoder configuration dictionary
    """

    def __init__(self, config: Dict):
        super(Decoder, self).__init__()

        self.in_dim = config['in_dim']
        self.hidden_dims: List[int] = config.get('hidden_dims', [32])
        self.out_dim = config['out_dim']
        self.dropout = config.get('dropout', 0.01)
        self.activate = config.get('activate', 'elu')

        # ============================================================
        # [LLM IMPLEMENTATION AREA]
        # Define decoder layers (Linear, BatchNorm, etc.)
        # ============================================================

        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")

    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass

        Args:
            z: Latent representations (batch_size, in_dim)
            **kwargs: Additional arguments

        Returns:
            Reconstructed output (batch_size, out_dim)
        """
        # ============================================================
        # [LLM IMPLEMENTATION AREA]
        # Process latent through decoder layers
        # ============================================================

        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")

    def get_output_dim(self) -> int:
        return self.out_dim
