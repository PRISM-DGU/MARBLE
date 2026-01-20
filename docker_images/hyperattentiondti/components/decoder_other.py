"""Drug-Target Interaction - Decoder Template

This file is a template for LLM (code expert) to implement code based on research papers.

[LLM CODING RULES]
1. Class name MUST remain 'Decoder' - DO NOT CHANGE
2. MUST implement: __init__, get_output_dim, forward methods
3. forward takes: drug_conv, protein_conv, drug_max_pool, protein_max_pool
4. forward output: torch.Tensor - shape (batch_size, num_classes) for classification
5. Read architecture parameters from config

[Config Example - model.decoder section in config.yaml]
decoder:
  type: decoder_other
  architecture:
    fc_dims: [1024, 1024, 512]
    dropout: 0.1
    num_classes: 2
"""

import torch
import torch.nn as nn
from typing import Dict, Any


# ==============================================================================
# AUXILIARY MODULES - LLM can add helper classes here
# ==============================================================================




# ==============================================================================
# MAIN DECODER CLASS - LLM MUST IMPLEMENT
# ==============================================================================
class Decoder(nn.Module):
    """Decoder for Drug-Target Interaction

    [LLM IMPLEMENTATION AREA]
    - Implement attention mechanism between drug and protein
    - Implement MLP classifier
    - Handle cross-attention and fusion

    Args:
        config: decoder configuration dictionary
        drug_dim: output dimension from drug encoder
        protein_dim: output dimension from protein encoder
    """

    def __init__(self, config: Dict[str, Any], drug_dim: int, protein_dim: int):
        super().__init__()

        self.drug_dim = drug_dim
        self.protein_dim = protein_dim

        arch_config = config.get('architecture', {})
        self.fc_dims = arch_config.get('fc_dims', [1024, 1024, 512])
        self.dropout_rate = arch_config.get('dropout', 0.1)
        self.num_classes = arch_config.get('num_classes', 2)
        self.output_dim = self.num_classes

        # ============================================================
        # [LLM IMPLEMENTATION AREA]
        # 1. Define attention layers for drug-protein interaction
        # 2. Define MLP classifier layers
        # 3. Define dropout, activation functions
        # ============================================================

        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, drug_conv: torch.Tensor, protein_conv: torch.Tensor,
                drug_max_pool: nn.MaxPool1d, protein_max_pool: nn.MaxPool1d) -> torch.Tensor:
        """Forward pass with cross-attention

        Args:
            drug_conv: Drug CNN features (batch_size, drug_dim, drug_seq_len)
            protein_conv: Protein CNN features (batch_size, protein_dim, protein_seq_len)
            drug_max_pool: MaxPool1d layer for drug
            protein_max_pool: MaxPool1d layer for protein

        Returns:
            Prediction logits (batch_size, num_classes)
        """
        # ============================================================
        # [LLM IMPLEMENTATION AREA]
        # 1. Compute cross-attention between drug and protein
        # 2. Apply attention weights to features
        # 3. Pool and concatenate features
        # 4. Pass through MLP classifier
        # ============================================================

        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")
