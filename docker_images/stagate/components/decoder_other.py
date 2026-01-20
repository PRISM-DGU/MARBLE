"""Spatial Transcriptomics Decoder - LLM Coding Template

This file is a template for LLM (code expert) to implement code based on research papers.

[LLM CODING RULES]
1. Class name MUST remain 'Decoder' - DO NOT CHANGE
2. MUST inherit from BaseDecoder
3. MUST implement: __init__, get_output_dim, forward methods
4. forward uses **kwargs pattern - take only what you need
5. forward input options:
   - z (torch.Tensor): latent embedding from encoder, shape (n_spots, latent_dim)
   - edge_index (torch.Tensor): graph connectivity, shape (2, n_edges) - optional
   - encoder (nn.Module): encoder instance for tied weights - optional
   - adata (AnnData): full anndata object - optional
6. forward output: torch.Tensor - shape (n_spots, output_dim) for reconstruction
7. Read architecture parameters from config

[Config Example - model.decoder section in config.yaml]
decoder:
  type: decoder_other
  architecture:
    in_dim: 30
    hidden_dim: 512
    out_dim: 3000
    # ... additional parameters based on paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseDecoder


# ==============================================================================
# AUXILIARY MODULES
# ==============================================================================
# Define any helper classes here BEFORE the main decoder class.
# Examples: custom GNN layers, attention blocks, MLP blocks, etc.




# ==============================================================================
# MAIN DECODER CLASS
# ==============================================================================
class Decoder(BaseDecoder):
    """Spatial Transcriptomics Decoder

    [LLM IMPLEMENTATION AREA]
    - Implement the decoder architecture proposed in the paper
    - Define layers in __init__
    - Implement forward pass in forward method
    - Optionally use tied weights from encoder

    Args:
        config: decoder architecture configuration dictionary
            - in_dim: input dimension (latent dim from encoder)
            - hidden_dim: hidden layer dimension
            - out_dim: output dimension (gene expression dim for reconstruction)
            - ... additional architecture parameters
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # ============================================================
        # [FIXED] Read basic dimensions from config
        # ============================================================
        self.in_dim = config.get('in_dim', 30)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.out_dim = config.get('out_dim', 3000)

        # ============================================================
        # [LLM IMPLEMENTATION AREA] Read additional parameters
        # Example:
        # self.n_layers = config.get('n_layers', 2)
        # self.dropout = config.get('dropout', 0.0)
        # self.use_tied_weights = config.get('tied_weights', False)
        # ============================================================

        # TODO: LLM implements parameter reading based on paper

        # ============================================================
        # [LLM IMPLEMENTATION AREA] Define layers
        # Example (MLP):
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.in_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.out_dim)
        # )
        #
        # Example (GNN):
        # self.conv1 = GCNConv(self.in_dim, self.hidden_dim)
        # self.conv2 = GCNConv(self.hidden_dim, self.out_dim)
        #
        # Example (Tied weights - layers defined but weights set in forward):
        # self.linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        # self.linear2 = nn.Linear(self.hidden_dim, self.out_dim)
        # ============================================================

        # TODO: LLM implements layer definitions based on paper
        raise NotImplementedError("LLM must implement this section")

    def get_output_dim(self) -> int:
        return self.out_dim

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor = None, encoder: nn.Module = None, adata=None, **kwargs) -> torch.Tensor:
        """Forward pass

        [FIXED] Input/Output format - use **kwargs, take only what you need
        Args:
            z: Latent embedding from encoder (n_spots, latent_dim)
            edge_index: Graph connectivity (2, n_edges) - optional, for GNN
            encoder: Encoder instance - optional, for tied weights
            adata: AnnData object - optional, for additional info
            **kwargs: Additional arguments (ignored if not needed)

        Returns:
            Reconstruction (n_spots, out_dim)

        [LLM IMPLEMENTATION AREA]
        - Implement forward pass using defined layers
        - Optionally use encoder weights (tied weights)
        - Use only the inputs you need
        """
        # ============================================================
        # [LLM IMPLEMENTATION AREA] Forward pass logic
        # Example (MLP - only uses z):
        # return self.decoder(z)
        #
        # Example (GNN - uses z and edge_index):
        # h = F.relu(self.conv1(z, edge_index))
        # h = self.conv2(h, edge_index)
        # return h
        #
        # Example (Tied weights - uses encoder):
        # if encoder is not None:
        #     self.linear1.weight.data = encoder.linear2.weight.data.T
        #     self.linear2.weight.data = encoder.linear1.weight.data.T
        # h = F.relu(self.linear1(z))
        # return self.linear2(h)
        # ============================================================

        # TODO: LLM implements forward pass based on paper
        raise NotImplementedError("LLM must implement this section")
