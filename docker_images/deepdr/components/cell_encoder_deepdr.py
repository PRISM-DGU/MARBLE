"""DeepDR Cell Encoder - DNN-based Gene Expression Encoder

DeepDR uses gene expression profiles as cell features and processes
them through a multi-layer DNN. Optionally supports DAE (Denoising
Autoencoder) pretrained weights.

Reference: DeepDR paper - DNN(EXP) and DAE(EXP) architectures
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class CellEncoder(nn.Module):
    """DNN-based Cell Encoder for gene expression (DeepDR style)

    Architecture: Gene Expression -> Linear layers with ReLU -> Cell embedding

    Supports loading pretrained DAE weights for better initialization.

    Config parameters:
        input.dim: Gene expression dimension (default: 6163 for DeepDR)
        architecture.hidden_layers: List of hidden layer dimensions
        architecture.dropout: Dropout rate
        architecture.use_dae: Whether to use DAE pretrained weights
        architecture.dae_weights_path: Path to pretrained DAE weights
        output.dim: Output embedding dimension
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Read dimensions from config
        self.input_dim = config.get('input', {}).get('dim', 6163)
        self.output_dim = config.get('output', {}).get('dim', 100)
        arch = config.get('architecture', {})

        # Architecture parameters
        hidden_layers = arch.get('hidden_layers', [2048, 512])
        if isinstance(hidden_layers, list) and len(hidden_layers) > 0:
            if isinstance(hidden_layers[0], dict):
                hidden_layers = [layer['dim'] for layer in hidden_layers]

        dropout = arch.get('dropout', 0.1)
        self.use_dae = arch.get('use_dae', False)
        self.dae_weights_path = arch.get('dae_weights_path', None)

        # Build layers
        dims = [self.input_dim] + hidden_layers + [self.output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.dropout = nn.Dropout(dropout)

        # Load DAE pretrained weights if specified
        if self.use_dae and self.dae_weights_path and os.path.exists(self.dae_weights_path):
            self._load_dae_weights()

    def _load_dae_weights(self):
        """Load pretrained DAE encoder weights"""
        try:
            state_dict = torch.load(self.dae_weights_path, map_location='cpu')
            # Only load weights for matching layers
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items()
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded DAE weights from {self.dae_weights_path}")
        except Exception as e:
            print(f"Warning: Failed to load DAE weights: {e}")

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, v, **kwargs) -> torch.Tensor:
        """Forward pass

        Args:
            v: Gene expression tensor (batch_size, input_dim)

        Returns:
            Cell embedding (batch_size, output_dim)
        """
        v = v.float()
        if len(self.layers) > 0:
            v = v.to(self.layers[0].weight.device)

        for i, layer in enumerate(self.layers):
            v = layer(v)
            if i < len(self.layers) - 1:  # No activation after last layer
                v = F.relu(v)
                v = self.dropout(v)

        return v


class CellDataLoader:
    """DeepDR Cell Data Loader - Gene Expression feature extraction

    Loads gene expression data from shared dataset files.
    """

    def __init__(self, config: Dict, base_path: str = None):
        self.config = config

        if base_path is None:
            base_path = config.get('data', {}).get('base_path', '/workspace/datasets/shared')
        self.base_path = base_path

        self.cache_dir = config.get('data', {}).get('preprocessing', {}).get('cache_dir', './cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Feature dimension from config
        self.feature_dim = config.get('data_dimensions', {}).get('raw', {}).get('gene_expression_dim', 6163)

    def load_gene_expression(self) -> pd.DataFrame:
        """Load gene expression data from file"""
        gene_file = self.config.get('data', {}).get('gene_expression_file', 'gene_expression.txt')
        gene_path = os.path.join(self.base_path, gene_file)

        if not os.path.exists(gene_path):
            print(f"Gene expression file not found: {gene_path}")
            return pd.DataFrame()

        gene_df = pd.read_csv(gene_path, sep='\t')
        print(f"Loaded gene expression: {gene_df.shape}")
        return gene_df

    def get_cell_features(self, cosmic_ids: List) -> Tuple[Dict, List]:
        """Extract gene expression features for cells

        Args:
            cosmic_ids: List of COSMIC IDs to process

        Returns:
            Tuple of (cell_features_dict, failed_cells_list)
        """
        cache_file = os.path.join(self.cache_dir, 'cell_features_deepdr.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cell_dict = pickle.load(f)
            print(f"Loaded {len(cell_dict)} cell features from cache")
            return cell_dict, []

        gene_df = self.load_gene_expression()
        if gene_df.empty:
            return {}, cosmic_ids

        cell_dict = {}
        failed = []

        for cosmic_id in cosmic_ids:
            if pd.isna(cosmic_id):
                continue

            cosmic_id = int(cosmic_id)
            col_name = f'DATA.{cosmic_id}'

            if col_name in gene_df.columns:
                features = gene_df[col_name].values.astype(np.float32)

                # Adjust dimension if needed
                if len(features) > self.feature_dim:
                    features = features[:self.feature_dim]
                elif len(features) < self.feature_dim:
                    features = np.pad(features, (0, self.feature_dim - len(features)), mode='constant')

                cell_dict[cosmic_id] = features
            else:
                failed.append(cosmic_id)

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(cell_dict, f)

        print(f"Generated {len(cell_dict)} cell features, {len(failed)} failed")
        return cell_dict, failed
