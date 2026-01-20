"""DeepTTA MLP Cell Encoder"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class CellEncoder(nn.Module):
    """Multi-Layer Perceptron based Cell Encoder (DeepTTA style)"""

    def __init__(self, config: Dict):
        super().__init__()

        input_dim = config.get('input', {}).get('dim', 17737)
        output_dim = config.get('output', {}).get('dim', 128)
        arch = config.get('architecture', {})

        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_layers = arch.get('hidden_layers', [4096, 2048, 512])
        if isinstance(hidden_layers[0], dict):
            hidden_layers = [layer['dim'] for layer in hidden_layers]

        dims = [input_dim] + hidden_layers + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, v, **kwargs):
        v = v.float().to(self.layers[0].weight.device)
        for layer in self.layers:
            v = F.relu(layer(v))
        return v


class CellDataLoader:
    """DeepTTA Cell Data Loader"""

    def __init__(self, config: Dict, base_path: str = None):
        self.config = config
        if base_path is None:
            base_path = config.get('data', {}).get('base_path', '/workspace/datasets/shared')
        self.base_path = base_path
        self.cache_dir = config.get('data', {}).get('preprocessing', {}).get('cache_dir', './cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.feature_dim = config.get('data_dimensions', {}).get('raw', {}).get('gene_expression_dim', 17737)

    def load_gene_expression(self) -> pd.DataFrame:
        gene_file = self.config.get('data', {}).get('gene_expression_file', 'gene_expression.txt')
        gene_path = os.path.join(self.base_path, gene_file)

        if not os.path.exists(gene_path):
            print(f"Gene expression file not found: {gene_path}")
            return pd.DataFrame()

        gene_df = pd.read_csv(gene_path, sep='\t')
        print(f"Loaded gene expression: {gene_df.shape}")
        return gene_df

    def get_cell_features(self, cosmic_ids: List) -> Tuple[Dict, List]:
        cache_file = os.path.join(self.cache_dir, 'cell_features_deeptta.pkl')
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
                if len(features) > self.feature_dim:
                    features = features[:self.feature_dim]
                elif len(features) < self.feature_dim:
                    features = np.pad(features, (0, self.feature_dim - len(features)), mode='constant')
                cell_dict[cosmic_id] = features
            else:
                failed.append(cosmic_id)

        with open(cache_file, 'wb') as f:
            pickle.dump(cell_dict, f)

        print(f"Generated {len(cell_dict)} cell features, {len(failed)} failed")
        return cell_dict, failed
