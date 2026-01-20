import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Dict

from components import get_encoder, get_graph_encoder, get_decoder, InnerProductDecoder


class DeepST(nn.Module):
    def __init__(self, config: Dict):
        super(DeepST, self).__init__()

        model_config = config['model']

        encoder_type = model_config['encoder'].get('type', 'encoder_deepst')
        graph_encoder_type = model_config['graph_encoder'].get('type', 'graph_encoder_deepst')
        decoder_type = model_config['decoder'].get('type', 'decoder_deepst')

        EncoderClass = get_encoder(encoder_type)
        self.encoder = EncoderClass(model_config['encoder']['architecture'])

        GraphEncoderClass = get_graph_encoder(graph_encoder_type)
        self.graph_encoder = GraphEncoderClass(model_config['graph_encoder']['architecture'])

        DecoderClass = get_decoder(decoder_type)
        self.decoder = DecoderClass(model_config['decoder']['architecture'])

        self.dc = InnerProductDecoder(dropout=model_config['encoder']['architecture'].get('dropout', 0.01))

        dec_config = model_config.get('dec_cluster', {})
        self.dec_cluster_n = dec_config.get('n_clusters', 20)
        self.alpha = dec_config.get('alpha', 0.9)

        latent_dim = self.encoder.get_output_dim() + self.graph_encoder.get_output_dim()
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, latent_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> tuple:
        feat_x = self.encoder(x)

        mu, logvar = self.graph_encoder(feat_x, edge_index)
        gnn_z = self.reparameterize(mu, logvar)

        z = torch.cat((feat_x, gnn_z), dim=1)

        de_feat = self.decoder(z)

        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        feat_x = self.encoder(x)
        mu, logvar = self.graph_encoder(feat_x, edge_index)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), dim=1)
        return z

    def target_distribution(self, target: torch.Tensor) -> torch.Tensor:
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
