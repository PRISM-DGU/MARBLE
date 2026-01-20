import torch
import torch.nn as nn
from typing import Dict

from components import get_encoder, get_decoder
from runtime_validator import RuntimeValidator


class STAGATE(nn.Module):
    def __init__(self, config: Dict):
        super(STAGATE, self).__init__()

        model_config = config['model']

        encoder_type = model_config['encoder'].get('type', 'encoder_stagate_gat')
        decoder_type = model_config['decoder'].get('type', 'decoder_stagate_gat')

        EncoderClass = get_encoder(encoder_type)
        self.encoder = EncoderClass(model_config['encoder']['architecture'])

        DecoderClass = get_decoder(decoder_type)
        self.decoder = DecoderClass(model_config['decoder']['architecture'])

        # Dynamic fix: Check and fix GATConv dimensions at initialization
        self._fix_model_dimensions()

    def _fix_model_dimensions(self):
        """Fix common dimension issues in encoder/decoder"""
        fixed_encoder = RuntimeValidator.fix_gatconv_dimensions(self.encoder)
        fixed_decoder = RuntimeValidator.fix_gatconv_dimensions(self.decoder)

        if fixed_encoder or fixed_decoder:
            print("[RuntimeValidator] Fixed GATConv dimension issues")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, adata=None, **kwargs):
        # Dynamic fix: Validate edge_index
        if edge_index is not None:
            edge_index = RuntimeValidator.validate_edge_index(edge_index, x.shape[0])

        # Run encoder
        z_raw = self.encoder(x=x, edge_index=edge_index, adata=adata, **kwargs)

        # Dynamic fix: Normalize encoder output (handles tuple/dict returns)
        z = RuntimeValidator.validate_encoder_output(z_raw)

        # Run decoder
        recon = self.decoder(z=z, edge_index=edge_index, encoder=self.encoder, adata=adata, **kwargs)

        return z, recon

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor = None, adata=None, **kwargs):
        z_raw = self.encoder(x=x, edge_index=edge_index, adata=adata, **kwargs)
        return RuntimeValidator.validate_encoder_output(z_raw)
