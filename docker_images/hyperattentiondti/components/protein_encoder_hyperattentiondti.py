"""HyperAttentionDTI Protein Encoder - CNN-based Protein Sequence Encoder

This module contains:
- ProteinEncoder: CNN-based encoder for protein character sequences
- ProteinDataLoader: Protein sequence string to integer encoding conversion
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List

# Protein character encoding (25 amino acids + padding)
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}


class ProteinEncoder(nn.Module):
    """CNN-based Protein Encoder for HyperAttentionDTI

    Encodes protein character sequences using Embedding + 3-layer CNN.
    Input: Integer-encoded protein sequence (batch_size, max_length)
    Output: CNN features (batch_size, conv*4, seq_len') for attention
    """

    def __init__(self, config: Dict):
        super().__init__()

        input_config = config.get('input', {})
        arch_config = config.get('architecture', {})

        vocab_size = input_config.get('vocab_size', 26)
        max_length = input_config.get('max_length', 1000)

        embedding_dim = arch_config.get('embedding_dim', 64)
        conv_filters = arch_config.get('conv_filters', 40)
        kernel_sizes = arch_config.get('kernel_sizes', [4, 8, 12])

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
            x: Integer-encoded protein sequence (batch_size, max_length)

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
            x: Integer-encoded protein sequence (batch_size, max_length)

        Returns:
            Pooled features (batch_size, output_dim)
        """
        x = self.forward(x)
        x = self.max_pool(x).squeeze(2)
        return x


class ProteinDataLoader:
    """Protein Data Loader for HyperAttentionDTI

    Converts protein sequences to integer encodings.
    """

    def __init__(self, config: Dict, base_path: str = None):
        self.config = config
        self.base_path = base_path

        input_config = config.get('model', {}).get('protein_encoder', {}).get('input', {})
        self.max_length = input_config.get('max_length', 1000)

    @staticmethod
    def label_sequence(sequence: str, max_length: int = 1000) -> np.ndarray:
        """Integer encoding for protein sequence string.

        Args:
            sequence: Protein sequence string
            max_length: Maximum encoding length

        Returns:
            Integer-encoded sequence as numpy array
        """
        encoding = np.zeros(max_length, dtype=np.int64)
        for idx, ch in enumerate(sequence[:max_length]):
            encoding[idx] = CHARPROTSET.get(ch, 0)
        return encoding

    def encode(self, sequence: str) -> np.ndarray:
        """Encode a single protein sequence"""
        return self.label_sequence(sequence, self.max_length)

    def encode_batch(self, sequence_list: List[str]) -> torch.Tensor:
        """Encode a batch of protein sequences

        Args:
            sequence_list: List of protein sequences

        Returns:
            Tensor of encoded sequences (batch_size, max_length)
        """
        batch = np.zeros((len(sequence_list), self.max_length), dtype=np.int64)
        for i, seq in enumerate(sequence_list):
            batch[i] = self.label_sequence(seq, self.max_length)
        return torch.from_numpy(batch)
