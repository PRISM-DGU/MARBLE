"""Drug-Target Interaction - Protein Encoder Template

This file is a template for LLM (code expert) to implement code based on research papers.

[LLM CODING RULES]
1. Class name MUST remain 'ProteinEncoder' - DO NOT CHANGE
2. MUST implement: __init__, get_output_dim, forward methods
3. forward input: Integer-encoded protein sequence (batch_size, max_length)
4. forward output: torch.Tensor - shape (batch_size, output_dim, seq_len) for attention
5. Read architecture parameters from config
6. MUST define self.max_pool for decoder attention mechanism

[Config Example - model.protein_encoder section in config.yaml]
protein_encoder:
  type: protein_encoder_other
  input:
    vocab_size: 26
    max_length: 1000
  architecture:
    embedding_dim: 64
    conv_filters: 40
    kernel_sizes: [4, 8, 12]
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List

# Protein character encoding (25 amino acids + padding) - DO NOT MODIFY
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}


# ==============================================================================
# AUXILIARY MODULES - LLM can add helper classes here
# ==============================================================================




# ==============================================================================
# MAIN PROTEIN ENCODER CLASS - LLM MUST IMPLEMENT
# ==============================================================================
class ProteinEncoder(nn.Module):
    """Protein Encoder for Drug-Target Interaction

    [LLM IMPLEMENTATION AREA]
    - Implement the protein encoder architecture proposed in the paper
    - MUST define self.output_dim
    - MUST define self.max_pool for decoder attention

    Args:
        config: protein_encoder configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        input_config = config.get('input', {})
        arch_config = config.get('architecture', {})

        self.vocab_size = input_config.get('vocab_size', 26)
        self.max_length = input_config.get('max_length', 1000)
        self.embedding_dim = arch_config.get('embedding_dim', 64)
        self.conv_filters = arch_config.get('conv_filters', 40)
        self.kernel_sizes = arch_config.get('kernel_sizes', [4, 8, 12])

        # ============================================================
        # [LLM IMPLEMENTATION AREA]
        # 1. Define self.output_dim
        # 2. Define encoder layers (embedding, CNN, etc.)
        # 3. Define self.max_pool for attention mechanism
        # ============================================================

        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Integer-encoded protein sequence (batch_size, max_length)

        Returns:
            CNN features (batch_size, output_dim, seq_len') for attention
        """
        # ============================================================
        # [LLM IMPLEMENTATION AREA]
        # Return CNN features BEFORE max pooling (for attention mechanism)
        # ============================================================

        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")


# ==============================================================================
# PROTEIN DATA LOADER - DO NOT MODIFY
# ==============================================================================
class ProteinDataLoader:
    """Protein Data Loader for HyperAttentionDTI - DO NOT MODIFY"""

    def __init__(self, config: Dict, base_path: str = None):
        self.config = config
        self.base_path = base_path
        input_config = config.get('model', {}).get('protein_encoder', {}).get('input', {})
        self.max_length = input_config.get('max_length', 1000)

    @staticmethod
    def label_sequence(sequence: str, max_length: int = 1000) -> np.ndarray:
        encoding = np.zeros(max_length, dtype=np.int64)
        for idx, ch in enumerate(sequence[:max_length]):
            encoding[idx] = CHARPROTSET.get(ch, 0)
        return encoding

    def encode(self, sequence: str) -> np.ndarray:
        return self.label_sequence(sequence, self.max_length)

    def encode_batch(self, sequence_list: List[str]) -> torch.Tensor:
        batch = np.zeros((len(sequence_list), self.max_length), dtype=np.int64)
        for i, seq in enumerate(sequence_list):
            batch[i] = self.label_sequence(seq, self.max_length)
        return torch.from_numpy(batch)
