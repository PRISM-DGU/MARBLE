"""Spatial Transcriptomics Encoder - LLM Coding Template

This file is a template for LLM (code expert) to implement code based on research papers.

[LLM CODING RULES]
1. Class name MUST remain 'Encoder' - DO NOT CHANGE
2. MUST inherit from BaseEncoder
3. MUST implement: __init__, get_output_dim, forward methods
4. forward uses **kwargs pattern - take only what you need
5. forward input options:
   - x (torch.Tensor): node features, shape (n_spots, gene_dim)
   - edge_index (torch.Tensor): graph connectivity, shape (2, n_edges) - optional
   - adata (AnnData): full anndata object - optional
6. forward output: torch.Tensor - shape (n_spots, output_dim)
7. Read architecture parameters from config

[Config Example - model.encoder section in config.yaml]
encoder:
  type: encoder_other
  architecture:
    in_dim: 3000
    hidden_dim: 512
    out_dim: 30
    # ... additional parameters based on paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseEncoder


# ==============================================================================
# AUXILIARY MODULES
# ==============================================================================
# Define any helper classes here BEFORE the main encoder class.
# Examples: custom GNN layers, attention blocks, MLP blocks, etc.




# ==============================================================================
# MAIN ENCODER CLASS
# ==============================================================================
class Encoder(BaseEncoder):
    """Spatial Transcriptomics Encoder

    [LLM IMPLEMENTATION AREA]
    - Implement the encoder architecture proposed in the paper
    - Define layers in __init__
    - Implement forward pass in forward method

    Args:
        config: encoder architecture configuration dictionary
            - in_dim: input dimension (gene expression dim)
            - hidden_dim: hidden layer dimension
            - out_dim: output/latent dimension
            - ... additional architecture parameters
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # ============================================================
        # [FIXED] Read basic dimensions from config
        # ============================================================
        self.in_dim = config.get('in_dim', 3000)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.out_dim = config.get('out_dim', 30)

        # ============================================================
        # [LLM IMPLEMENTATION AREA] Read additional parameters
        # Example:
        # self.n_layers = config.get('n_layers', 2)
        # self.dropout = config.get('dropout', 0.0)
        # self.heads = config.get('heads', 1)
        # ============================================================

        # TODO: LLM implements parameter reading based on paper

        # ============================================================
        # [LLM IMPLEMENTATION AREA] Define layers
        # Example (MLP):
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.in_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.out_dim)
        # )
        #
        # Example (GNN):
        # self.conv1 = GCNConv(self.in_dim, self.hidden_dim)
        # self.conv2 = GCNConv(self.hidden_dim, self.out_dim)
        #
        # Example (Transformer):
        # self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        # self.transformer = nn.TransformerEncoder(...)
        # self.output_proj = nn.Linear(self.hidden_dim, self.out_dim)
        # ============================================================

        # TODO: LLM implements layer definitions based on paper
        raise NotImplementedError("LLM must implement this section")

    def get_output_dim(self) -> int:
        return self.out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, adata=None, **kwargs) -> torch.Tensor:
        """Forward pass

        [FIXED] Input/Output format - use **kwargs, take only what you need
        Args:
            x: Node features (n_spots, gene_dim)
            edge_index: Graph connectivity (2, n_edges) - optional, for GNN
            adata: AnnData object - optional, for additional info
            **kwargs: Additional arguments (ignored if not needed)

        Returns:
            Latent embedding (n_spots, out_dim)

        [LLM IMPLEMENTATION AREA]
        - Implement forward pass using defined layers
        - Use only the inputs you need
        """
        # ============================================================
        # [LLM IMPLEMENTATION AREA] Forward pass logic
        # Example (MLP - only uses x):
        # return self.encoder(x)
        #
        # Example (GNN - uses x and edge_index):
        # h = F.relu(self.conv1(x, edge_index))
        # h = self.conv2(h, edge_index)
        # return h
        #
        # Example (with adata):
        # spatial_coords = torch.tensor(adata.obsm['spatial'])
        # ...
        # ============================================================

        # TODO: LLM implements forward pass based on paper
        raise NotImplementedError("LLM must implement this section")
