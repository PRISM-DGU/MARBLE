"""Drug-Target Interaction - Drug Encoder Template

This file is a template for LLM (code expert) to implement code based on research papers.

[LLM CODING RULES]
1. Class name MUST remain 'DrugEncoder' - DO NOT CHANGE
2. MUST implement: __init__, get_output_dim, forward methods
3. forward input: Integer-encoded SMILES (batch_size, max_length)
4. forward output: torch.Tensor - shape (batch_size, output_dim, seq_len) for attention
5. Read architecture parameters from config
6. MUST define self.max_pool for decoder attention mechanism

[Config Example - model.drug_encoder section in config.yaml]
drug_encoder:
  type: drug_encoder_other
  input:
    vocab_size: 65
    max_length: 100
  architecture:
    embedding_dim: 64
    conv_filters: 40
    kernel_sizes: [4, 6, 8]
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List

# SMILES character encoding (64 characters + padding) - DO NOT MODIFY
CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64
}


# ==============================================================================
# AUXILIARY MODULES - LLM can add helper classes here
# ==============================================================================




# ==============================================================================
# MAIN DRUG ENCODER CLASS - LLM MUST IMPLEMENT
# ==============================================================================
class DrugEncoder(nn.Module):
    """Drug Encoder for Drug-Target Interaction

    [LLM IMPLEMENTATION AREA]
    - Implement the drug encoder architecture proposed in the paper
    - MUST define self.output_dim
    - MUST define self.max_pool for decoder attention

    Args:
        config: drug_encoder configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        input_config = config.get('input', {})
        arch_config = config.get('architecture', {})

        self.vocab_size = input_config.get('vocab_size', 65)
        self.max_length = input_config.get('max_length', 100)
        self.embedding_dim = arch_config.get('embedding_dim', 64)
        self.conv_filters = arch_config.get('conv_filters', 40)
        self.kernel_sizes = arch_config.get('kernel_sizes', [4, 6, 8])

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
            x: Integer-encoded SMILES (batch_size, max_length)

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
# DRUG DATA LOADER - DO NOT MODIFY
# ==============================================================================
class DrugDataLoader:
    """Drug Data Loader for HyperAttentionDTI - DO NOT MODIFY"""

    def __init__(self, config: Dict, base_path: str = None):
        self.config = config
        self.base_path = base_path
        input_config = config.get('model', {}).get('drug_encoder', {}).get('input', {})
        self.max_length = input_config.get('max_length', 100)

    @staticmethod
    def label_smiles(smiles: str, max_length: int = 100) -> np.ndarray:
        encoding = np.zeros(max_length, dtype=np.int64)
        for idx, ch in enumerate(smiles[:max_length]):
            encoding[idx] = CHARISOSMISET.get(ch, 0)
        return encoding

    def encode(self, smiles: str) -> np.ndarray:
        return self.label_smiles(smiles, self.max_length)

    def encode_batch(self, smiles_list: List[str]) -> torch.Tensor:
        batch = np.zeros((len(smiles_list), self.max_length), dtype=np.int64)
        for i, smiles in enumerate(smiles_list):
            batch[i] = self.label_smiles(smiles, self.max_length)
        return torch.from_numpy(batch)
