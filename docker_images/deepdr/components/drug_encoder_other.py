"""Drug Response Prediction - Drug Encoder Template

This file is a template for LLM (code expert) to implement code based on research papers.

[LLM CODING RULES]
1. Class name MUST remain 'DrugEncoder' - DO NOT CHANGE
2. MUST inherit from BaseDrugEncoder
3. MUST implement: __init__, get_output_dim, forward methods
4. forward uses **kwargs pattern - take only what you need
5. forward input options:
   - v (torch.Tensor): For ECFP/fingerprint tensor-based input
   - v (torch_geometric.data.Batch): For molecular graph-based input
6. forward output: torch.Tensor - shape (batch_size, output_dim)
7. Read architecture parameters from config

[Config Example - model.drug_encoder section in config.yaml]
drug_encoder:
  type: drug_encoder_other
  input:
    dim: 512
  architecture:
    hidden_layers: [256, 128]
    dropout: 0.1
    # ... additional parameters based on paper
  output:
    dim: 128
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from rdkit import Chem
from rdkit.Chem import AllChem

from .base import BaseDrugEncoder


# ==============================================================================
# AUXILIARY MODULES
# ==============================================================================
# Define any helper classes here BEFORE the main encoder class.
# Examples: custom GNN layers, attention blocks, transformer layers, etc.




# ==============================================================================
# MAIN DRUG ENCODER CLASS
# ==============================================================================
class DrugEncoder(BaseDrugEncoder):
    """Drug Encoder for Drug Response Prediction

    [LLM IMPLEMENTATION AREA]
    - Implement the drug encoder architecture proposed in the paper
    - Define layers in __init__
    - Implement forward pass in forward method

    Args:
        config: drug_encoder configuration dictionary
            - input.dim: input dimension (ECFP bits or graph node features)
            - architecture.*: architecture parameters
            - output.dim: output/embedding dimension
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)

        # ============================================================
        # [FIXED] Read basic dimensions from config
        # ============================================================
        self.input_dim = config.get('input', {}).get('dim', 512)
        self.output_dim = config.get('output', {}).get('dim', 128)
        arch = config.get('architecture', {})
        self.hidden_dim = arch.get('hidden_dim', 128)

        # ============================================================
        # [LLM IMPLEMENTATION AREA] Read additional parameters
        # Example:
        # self.hidden_layers = arch.get('hidden_layers', [256, 128])
        # self.dropout = arch.get('dropout', 0.1)
        # self.n_heads = arch.get('n_heads', 8)
        # ============================================================

        # TODO: LLM implements parameter reading based on paper

        # ============================================================
        # [LLM IMPLEMENTATION AREA] Define layers
        # Example (MLP):
        # dims = [self.input_dim] + self.hidden_layers + [self.output_dim]
        # self.layers = nn.ModuleList([
        #     nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        # ])
        #
        # Example (GNN for molecular graph):
        # from torch_geometric.nn import GCNConv, global_mean_pool
        # self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        # self.conv2 = GCNConv(self.hidden_dim, self.output_dim)
        #
        # Example (Message Passing Graph - MPG):
        # self.mpg_layers = nn.ModuleList([...])
        # ============================================================

        # TODO: LLM implements layer definitions based on paper
        raise NotImplementedError("LLM must implement this section")

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, v, **kwargs) -> torch.Tensor:
        """Forward pass

        [FIXED] Input/Output format - use **kwargs, take only what you need
        Args:
            v: Drug input data
               - For tensor-based (ECFP): torch.Tensor (batch_size, input_dim)
               - For graph-based: torch_geometric.data.Batch

        Returns:
            Drug embedding (batch_size, output_dim)

        [LLM IMPLEMENTATION AREA]
        - Implement forward pass using defined layers
        - Handle the appropriate input type
        """
        # ============================================================
        # [LLM IMPLEMENTATION AREA] Forward pass logic
        # Example (MLP for ECFP):
        # x = v.float()
        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        #     if i < len(self.layers) - 1:
        #         x = F.relu(x)
        # return x
        #
        # Example (GNN for molecular graph):
        # x, edge_index, batch = v.x, v.edge_index, v.batch
        # h = F.relu(self.conv1(x, edge_index))
        # h = self.conv2(h, edge_index)
        # return global_mean_pool(h, batch)
        # ============================================================

        # TODO: LLM implements forward pass based on paper
        raise NotImplementedError("LLM must implement this section")


# ==============================================================================
# [FIXED] DRUG DATA LOADER - DO NOT MODIFY
# ==============================================================================
class DrugDataLoader:
    """Drug Data Loader

    [FIXED] Uses ECFP fingerprints as default.
    DO NOT MODIFY this class.
    """

    def __init__(self, config: Dict, vocab_path: str = None, base_path: str = None):
        self.config = config
        self.vocab_path = vocab_path

        if base_path is None:
            base_path = config.get('data', {}).get('base_path', '/workspace/datasets/shared')
        self.base_path = base_path

        self.cache_dir = config.get('data', {}).get('preprocessing', {}).get('cache_dir', './cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        drug_enc_config = config.get('model', {}).get('drug_encoder', {})
        self.fp_radius = drug_enc_config.get('architecture', {}).get('fp_radius', 2)
        self.fp_bits = drug_enc_config.get('input', {}).get('dim', 512)

    def load_smiles(self) -> Dict[str, str]:
        """Load SMILES data"""
        smiles_file = self.config.get('data', {}).get('drug_smiles_file', 'drug_smiles.csv')
        smiles_path = os.path.join(self.base_path, smiles_file)
        if not os.path.exists(smiles_path):
            print(f"SMILES file not found: {smiles_path}")
            return {}

        smiles_df = pd.read_csv(smiles_path)
        drug_smiles = {}
        for _, row in smiles_df.iterrows():
            drug_name = row.get('drug_name', row.get('drug_id'))
            smiles = row.get('SMILES', row.get('smiles'))
            if drug_name and smiles:
                drug_smiles[str(drug_name).strip()] = str(smiles).strip()
        print(f"Loaded {len(drug_smiles)} drug SMILES")
        return drug_smiles

    def smiles_to_ecfp(self, smiles: str) -> np.ndarray:
        """Convert SMILES to ECFP fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
        return np.array(fp)

    def get_drug_features(self, drug_names: List[str]) -> Tuple[Dict, List]:
        """Extract drug features"""
        cache_file = os.path.join(self.cache_dir, f'drug_features_deepdr_ecfp{self.fp_bits}.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                drug_dict = pickle.load(f)
            print(f"Loaded {len(drug_dict)} drug features from cache")
            return drug_dict, []

        drug_smiles = self.load_smiles()
        drug_dict = {}
        failed = []

        for drug_name in drug_names:
            if drug_name not in drug_smiles:
                failed.append(drug_name)
                continue
            try:
                ecfp = self.smiles_to_ecfp(drug_smiles[drug_name])
                if ecfp is not None:
                    drug_dict[drug_name] = ecfp.astype(np.float32)
                else:
                    failed.append(drug_name)
            except Exception:
                failed.append(drug_name)

        with open(cache_file, 'wb') as f:
            pickle.dump(drug_dict, f)

        print(f"Generated {len(drug_dict)} drug features, {len(failed)} failed")
        return drug_dict, failed
