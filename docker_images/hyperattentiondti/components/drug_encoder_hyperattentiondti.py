"""HyperAttentionDTI Drug Encoder - CNN-based SMILES Encoder

This module contains:
- DrugEncoder: CNN-based encoder for SMILES character sequences
- DrugDataLoader: SMILES string to integer encoding conversion
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List

# SMILES character encoding (64 characters + padding)
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


class DrugEncoder(nn.Module):
    """CNN-based Drug Encoder for HyperAttentionDTI

    Encodes SMILES character sequences using Embedding + 3-layer CNN.
    Input: Integer-encoded SMILES (batch_size, max_length)
    Output: CNN features (batch_size, conv*4, seq_len') for attention
    """

    def __init__(self, config: Dict):
        super().__init__()

        input_config = config.get('input', {})
        arch_config = config.get('architecture', {})

        vocab_size = input_config.get('vocab_size', 65)
        max_length = input_config.get('max_length', 100)

        embedding_dim = arch_config.get('embedding_dim', 64)
        conv_filters = arch_config.get('conv_filters', 40)
        kernel_sizes = arch_config.get('kernel_sizes', [4, 6, 8])

        self.max_length = max_length
        self.output_dim = conv_filters * 4  # After 3 CNN layers: conv -> conv*2 -> conv*4

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # CNN layers (dim -> conv -> conv*2 -> conv*4)
        self.cnns = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=conv_filters, kernel_size=kernel_sizes[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_filters, out_channels=conv_filters * 2, kernel_size=kernel_sizes[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_filters * 2, out_channels=conv_filters * 4, kernel_size=kernel_sizes[2]),
            nn.ReLU(),
        )

        # Calculate output sequence length after CNN
        cnn_out_len = max_length - kernel_sizes[0] - kernel_sizes[1] - kernel_sizes[2] + 3
        self.max_pool = nn.MaxPool1d(cnn_out_len)

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Integer-encoded SMILES (batch_size, max_length)

        Returns:
            CNN features (batch_size, output_dim, seq_len') - before max pooling
            Used for attention mechanism in decoder
        """
        # Embedding: (batch, seq_len, embed_dim)
        x = self.embedding(x)

        # Transpose for Conv1d: (batch, embed_dim, seq_len)
        x = x.permute(0, 2, 1)

        # CNN: (batch, conv*4, seq_len')
        x = self.cnns(x)

        return x

    def forward_with_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with max pooling (for standalone use)

        Args:
            x: Integer-encoded SMILES (batch_size, max_length)

        Returns:
            Pooled features (batch_size, output_dim)
        """
        x = self.forward(x)
        x = self.max_pool(x).squeeze(2)
        return x


class DrugDataLoader:
    """Drug Data Loader for HyperAttentionDTI

    Converts SMILES strings to integer encodings.
    """

    def __init__(self, config: Dict, base_path: str = None):
        self.config = config
        self.base_path = base_path

        input_config = config.get('model', {}).get('drug_encoder', {}).get('input', {})
        self.max_length = input_config.get('max_length', 100)

    @staticmethod
    def label_smiles(smiles: str, max_length: int = 100) -> np.ndarray:
        """Integer encoding for SMILES string.

        Args:
            smiles: SMILES string
            max_length: Maximum encoding length

        Returns:
            Integer-encoded sequence as numpy array
        """
        encoding = np.zeros(max_length, dtype=np.int64)
        for idx, ch in enumerate(smiles[:max_length]):
            encoding[idx] = CHARISOSMISET.get(ch, 0)
        return encoding

    def encode(self, smiles: str) -> np.ndarray:
        """Encode a single SMILES string"""
        return self.label_smiles(smiles, self.max_length)

    def encode_batch(self, smiles_list: List[str]) -> torch.Tensor:
        """Encode a batch of SMILES strings

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Tensor of encoded SMILES (batch_size, max_length)
        """
        batch = np.zeros((len(smiles_list), self.max_length), dtype=np.int64)
        for i, smiles in enumerate(smiles_list):
            batch[i] = self.label_smiles(smiles, self.max_length)
        return torch.from_numpy(batch)
