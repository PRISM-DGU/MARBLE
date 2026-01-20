"""DeepDR Model - Registry-based component loading"""

import torch
import torch.nn as nn
from typing import Dict

from components import get_cell_encoder, get_drug_encoder, get_decoder


class DeepDRModel(nn.Module):
    """DeepDR Drug Response Prediction Model

    A modular model that uses registry pattern to dynamically load
    drug encoder, cell encoder, and decoder components.

    Components can be swapped via the 'type' field in config:
    - drug_encoder.type: drug_encoder_deepdr, drug_encoder_other
    - cell_encoder.type: cell_encoder_deepdr, cell_encoder_other
    - decoder.type: decoder_deepdr, decoder_other
    """

    def __init__(self, config: Dict, **kwargs):
        super().__init__()
        self.config = config

        model_config = config['model']
        drug_enc_config = model_config['drug_encoder']
        cell_enc_config = model_config['cell_encoder']

        # Read type from config for dynamic class loading
        cell_enc_type = cell_enc_config.get('type', 'cell_encoder_deepdr')
        drug_enc_type = drug_enc_config.get('type', 'drug_encoder_deepdr')
        decoder_type = model_config.get('decoder', {}).get('type', 'decoder_deepdr')

        # Get classes from registry and instantiate
        CellEncoderClass = get_cell_encoder(cell_enc_type)
        DrugEncoderClass = get_drug_encoder(drug_enc_type)
        DecoderClass = get_decoder(decoder_type)

        # Instantiate encoders
        self.drug_encoder = DrugEncoderClass(drug_enc_config, **kwargs)
        self.cell_encoder = CellEncoderClass(cell_enc_config)

        # Get output dimensions from encoders
        drug_dim = self.drug_encoder.get_output_dim()
        cell_dim = self.cell_encoder.get_output_dim()

        # Instantiate decoder with encoder output dimensions
        self.decoder = DecoderClass(model_config, drug_dim, cell_dim)

    def forward(self, drug_features, cell_features):
        """Forward pass

        Args:
            drug_features: Drug input (ECFP tensor, graph batch, etc.)
            cell_features: Cell input (gene expression tensor, etc.)

        Returns:
            IC50 prediction (batch_size, 1)
        """
        drug_encoded = self.drug_encoder(drug_features)
        cell_encoded = self.cell_encoder(cell_features)
        return self.decoder(drug_encoded, cell_encoded)

    def get_output_dim(self) -> int:
        """Get output dimension of the model"""
        return self.decoder.get_output_dim()

    def get_encoder_dims(self) -> Dict[str, int]:
        """Get output dimensions of encoders"""
        return {
            'drug_encoder': self.drug_encoder.get_output_dim(),
            'cell_encoder': self.cell_encoder.get_output_dim(),
        }


def create_model(config: Dict, **kwargs) -> DeepDRModel:
    """Factory function to create DeepDR model

    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments passed to model

    Returns:
        DeepDRModel instance
    """
    return DeepDRModel(config, **kwargs)
