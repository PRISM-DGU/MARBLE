"""HyperAttentionDTI Decoder - Cross-Attention + MLP Classifier

This module contains:
- Decoder: Cross-Attention mechanism between drug and protein + MLP classifier

The attention mechanism computes interaction between drug and protein representations,
then applies attention-weighted features for final classification.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class Decoder(nn.Module):
    """Cross-Attention + MLP Decoder for HyperAttentionDTI

    Computes cross-attention between drug and protein CNN features,
    applies attention-weighted pooling, and classifies using MLP.
    """

    def __init__(self, config: Dict, drug_dim: int, protein_dim: int):
        super().__init__()

        self.drug_dim = drug_dim  # conv * 4 (e.g., 160)
        self.protein_dim = protein_dim  # conv * 4 (e.g., 160)

        arch_config = config.get('architecture', {})
        fc_dims = arch_config.get('fc_dims', [1024, 1024, 512])
        dropout = arch_config.get('dropout', 0.1)
        num_classes = arch_config.get('num_classes', 2)

        self.output_dim = num_classes

        # Attention layers
        self.attention_layer = nn.Linear(drug_dim, drug_dim)
        self.drug_attention_layer = nn.Linear(drug_dim, drug_dim)
        self.protein_attention_layer = nn.Linear(protein_dim, protein_dim)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # MLP layers: concat_dim (drug_dim + protein_dim) -> fc_dims -> num_classes
        concat_dim = drug_dim + protein_dim  # e.g., 320
        self.fc1 = nn.Linear(concat_dim, fc_dims[0])
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.fc3 = nn.Linear(fc_dims[1], fc_dims[2])
        self.out = nn.Linear(fc_dims[2], num_classes)

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
        # Compute attention projections
        # drug_conv: (batch, channels, seq_len) -> (batch, seq_len, channels)
        drug_att = self.drug_attention_layer(drug_conv.permute(0, 2, 1))
        protein_att = self.protein_attention_layer(protein_conv.permute(0, 2, 1))

        # Create attention matrix
        # d_att_layers: (batch, drug_seq, 1, channels) -> repeat along protein
        # p_att_layers: (batch, 1, protein_seq, channels) -> repeat along drug
        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, protein_conv.shape[-1], 1)
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, drug_conv.shape[-1], 1, 1)

        # Combined attention matrix
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))

        # Average across dimensions to get attention weights
        Compound_atte = torch.mean(Atten_matrix, 2)  # Average over protein dimension
        Protein_atte = torch.mean(Atten_matrix, 1)   # Average over drug dimension

        # Apply sigmoid and transpose back
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))

        # Apply attention: residual connection with attention weights
        drug_conv = drug_conv * 0.5 + drug_conv * Compound_atte
        protein_conv = protein_conv * 0.5 + protein_conv * Protein_atte

        # Max pooling to get fixed-size representations
        drug_pooled = drug_max_pool(drug_conv).squeeze(2)  # (batch, drug_dim)
        protein_pooled = protein_max_pool(protein_conv).squeeze(2)  # (batch, protein_dim)

        # Concatenate drug and protein features
        pair = torch.cat([drug_pooled, protein_pooled], dim=1)

        # MLP classifier
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict

    def forward_simple(self, drug_pooled: torch.Tensor, protein_pooled: torch.Tensor) -> torch.Tensor:
        """Simple forward pass without attention (for pre-pooled inputs)

        Args:
            drug_pooled: Pooled drug features (batch_size, drug_dim)
            protein_pooled: Pooled protein features (batch_size, protein_dim)

        Returns:
            Prediction logits (batch_size, num_classes)
        """
        # Concatenate drug and protein features
        pair = torch.cat([drug_pooled, protein_pooled], dim=1)

        # MLP classifier
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict
