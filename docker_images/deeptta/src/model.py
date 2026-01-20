"""DeepTTA Model - Registry-based component loading"""

import torch
import torch.nn as nn
from typing import Dict

from components import get_cell_encoder, get_drug_encoder, get_decoder


class DeepTTAModel(nn.Module):
    """DeepTTA Drug Response Prediction Model

    Components can be dynamically swapped via the 'type' field in config.
    """

    def __init__(self, config: Dict, vocab_path: str = None):
        super().__init__()
        self.config = config

        model_config = config['model']
        drug_enc_config = model_config['drug_encoder']
        cell_enc_config = model_config['cell_encoder']

        # Read type from config for dynamic class loading
        cell_enc_type = cell_enc_config.get('type', 'cell_encoder_deeptta')
        drug_enc_type = drug_enc_config.get('type', 'drug_encoder_deeptta')
        decoder_type = model_config.get('decoder', {}).get('type', 'decoder_deeptta')

        # Get classes from registry and instantiate
        CellEncoderClass = get_cell_encoder(cell_enc_type)
        DrugEncoderClass = get_drug_encoder(drug_enc_type)
        DecoderClass = get_decoder(decoder_type)

        # DeepTTA drug encoder needs vocab_path
        if drug_enc_type == 'drug_encoder_deeptta' and vocab_path:
            self.drug_encoder = DrugEncoderClass(drug_enc_config, vocab_path=vocab_path)
        else:
            self.drug_encoder = DrugEncoderClass(drug_enc_config)

        self.cell_encoder = CellEncoderClass(cell_enc_config)

        drug_dim = self.drug_encoder.get_output_dim()
        cell_dim = self.cell_encoder.get_output_dim()

        self.decoder = DecoderClass(model_config, drug_dim, cell_dim)

    def forward(self, drug_features, cell_features):
        drug_encoded = self.drug_encoder(drug_features)
        cell_encoded = self.cell_encoder(cell_features)
        return self.decoder(drug_encoded, cell_encoded)

    def get_output_dim(self) -> int:
        return self.decoder.get_output_dim()


def create_model(config: Dict, vocab_path: str = None) -> DeepTTAModel:
    return DeepTTAModel(config, vocab_path=vocab_path)
