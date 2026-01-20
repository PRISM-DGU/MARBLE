"""HyperAttentionDTI Model

Unified model that combines drug encoder, protein encoder, and decoder.
Uses registry to dynamically load components based on config.
"""

import torch
import torch.nn as nn
from typing import Dict

from components import get_drug_encoder, get_protein_encoder, get_decoder


class HyperAttentionDTI(nn.Module):
    """HyperAttentionDTI Model

    Drug-Target Interaction prediction model using:
    - Drug Encoder: CNN-based SMILES encoder
    - Protein Encoder: CNN-based protein sequence encoder
    - Decoder: Cross-attention + MLP classifier
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        model_config = config.get('model', {})

        # Get encoder/decoder types from config
        drug_enc_type = model_config.get('drug_encoder', {}).get('type', 'drug_encoder_hyperattentiondti')
        protein_enc_type = model_config.get('protein_encoder', {}).get('type', 'protein_encoder_hyperattentiondti')
        decoder_type = model_config.get('decoder', {}).get('type', 'decoder_hyperattentiondti')

        # Get component classes from registry
        DrugEncoderClass = get_drug_encoder(drug_enc_type)
        ProteinEncoderClass = get_protein_encoder(protein_enc_type)
        DecoderClass = get_decoder(decoder_type)

        # Initialize encoders
        drug_enc_config = model_config.get('drug_encoder', {})
        protein_enc_config = model_config.get('protein_encoder', {})
        decoder_config = model_config.get('decoder', {})

        self.drug_encoder = DrugEncoderClass(drug_enc_config)
        self.protein_encoder = ProteinEncoderClass(protein_enc_config)

        # Get output dimensions
        drug_dim = self.drug_encoder.get_output_dim()
        protein_dim = self.protein_encoder.get_output_dim()

        # Initialize decoder
        self.decoder = DecoderClass(decoder_config, drug_dim, protein_dim)

        # Store max pool layers for decoder attention
        self.drug_max_pool = self.drug_encoder.max_pool
        self.protein_max_pool = self.protein_encoder.max_pool

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            drug: Integer-encoded SMILES (batch_size, drug_max_length)
            protein: Integer-encoded protein sequence (batch_size, protein_max_length)

        Returns:
            Prediction logits (batch_size, num_classes)
        """
        # Encode drug and protein (without pooling - for attention)
        drug_conv = self.drug_encoder(drug)
        protein_conv = self.protein_encoder(protein)

        # Decode with cross-attention
        predict = self.decoder(drug_conv, protein_conv, self.drug_max_pool, self.protein_max_pool)

        return predict

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_drug_encoder(self) -> nn.Module:
        """Get drug encoder module"""
        return self.drug_encoder

    def get_protein_encoder(self) -> nn.Module:
        """Get protein encoder module"""
        return self.protein_encoder

    def get_decoder(self) -> nn.Module:
        """Get decoder module"""
        return self.decoder
