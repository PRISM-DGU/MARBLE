"""Drug Response Prediction - Drug Encoder Template

This file is a template for LLM (code expert) to implement code based on research papers.

[LLM CODING RULES]
1. Class name MUST remain 'DrugEncoder' - DO NOT CHANGE
2. MUST inherit from BaseDrugEncoder
3. MUST implement: __init__, get_output_dim, forward methods
4. forward uses **kwargs pattern - take only what you need
5. forward input options:
   - v (tuple): For token-based input (tokens, masks)
   - v (torch.Tensor): For tensor-based input
   - v (torch_geometric.data.Batch): For graph-based input
6. forward output: torch.Tensor - shape (batch_size, output_dim)
7. Read architecture parameters from config

[Config Example - model.drug_encoder section in config.yaml]
drug_encoder:
  type: drug_encoder_other
  input:
    dim: 3000
    vocab_size: 3000
  architecture:
    hidden_dim: 128
    embedding_dim: 128
    n_layers: 2
    # ... additional parameters based on paper
  output:
    dim: 128
"""

import os
import codecs
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from subword_nmt.apply_bpe import BPE

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
            - input.dim: input dimension
            - input.vocab_size: vocabulary size (for token-based)
            - architecture.*: architecture parameters
            - output.dim: output/embedding dimension
    """

    def __init__(self, config: Dict[str, Any], vocab_path: str = None):
        super().__init__(config)

        # ============================================================
        # [FIXED] Read basic dimensions from config
        # ============================================================
        self.input_dim = config.get('input', {}).get('dim', 3000)
        self.output_dim = config.get('output', {}).get('dim', 128)
        arch = config.get('architecture', {})
        self.hidden_dim = arch.get('hidden_dim', 128)

        # ============================================================
        # [LLM IMPLEMENTATION AREA] Read additional parameters
        # Example:
        # self.n_layers = arch.get('n_layers', 2)
        # self.dropout = arch.get('dropout', 0.1)
        # self.n_heads = arch.get('n_heads', 8)
        # ============================================================

        # TODO: LLM implements parameter reading based on paper

        # ============================================================
        # [LLM IMPLEMENTATION AREA] Define layers
        # Example (MLP):
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.output_dim)
        # )
        #
        # Example (Transformer):
        # self.embedding = nn.Embedding(vocab_size, self.hidden_dim)
        # self.transformer = nn.TransformerEncoder(...)
        #
        # Example (GNN):
        # self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        # self.conv2 = GCNConv(self.hidden_dim, self.output_dim)
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
               - For token-based: tuple of (tokens, masks)
               - For tensor-based: torch.Tensor
               - For graph-based: torch_geometric.data.Batch
            **kwargs: Additional arguments (ignored if not needed)

        Returns:
            Drug embedding (batch_size, output_dim)

        [LLM IMPLEMENTATION AREA]
        - Implement forward pass using defined layers
        - Handle the appropriate input type
        """
        # ============================================================
        # [LLM IMPLEMENTATION AREA] Forward pass logic
        # Example (Token-based):
        # tokens, masks = v[0], v[1]
        # emb = self.embedding(tokens)
        # out = self.transformer(emb)
        # return out[:, 0]  # CLS token
        #
        # Example (Tensor-based):
        # return self.encoder(v)
        #
        # Example (Graph-based):
        # x, edge_index, batch = v.x, v.edge_index, v.batch
        # h = F.relu(self.conv1(x, edge_index))
        # h = self.conv2(h, edge_index)
        # return global_mean_pool(h, batch)
        # ============================================================

        # TODO: LLM implements forward pass based on paper
        raise NotImplementedError("LLM must implement this section")


# ==============================================================================
# [FIXED] SMILES ENCODER - DO NOT MODIFY
# ==============================================================================
class SMILESEncoder:
    """SMILES to BPE token encoder

    [FIXED] Uses the same BPE encoding as DeepTTA.
    DO NOT MODIFY this class.
    """

    def __init__(self, vocab_path: str, config: Dict):
        self.vocab_path = vocab_path
        self.config = config
        self.max_length = config.get('data_dimensions', {}).get('raw', {}).get('drug_sequence_length', 100)

        output_files = config.get('output_files', {})
        vocab_subdir = output_files.get('vocab_subdir', '')
        bpe_vocab = output_files.get('bpe_vocab', 'deeptta_drug_encoder_bpe_vocab.txt')
        bpe_subword_map = output_files.get('bpe_subword_map', 'deeptta_drug_encoder_bpe_mapping.csv')

        vocab_file = os.path.join(vocab_path, vocab_subdir, bpe_vocab) if vocab_subdir else os.path.join(vocab_path, bpe_vocab)
        sub_csv_path = os.path.join(vocab_path, vocab_subdir, bpe_subword_map) if vocab_subdir else os.path.join(vocab_path, bpe_subword_map)

        sub_csv = pd.read_csv(sub_csv_path)
        with codecs.open(vocab_file) as bpe_codes:
            self.bpe = BPE(bpe_codes, merges=-1, separator='')

        idx2word = sub_csv['index'].values
        self.words2idx = dict(zip(idx2word, range(len(idx2word))))

    def encode(self, smiles: str) -> Tuple[np.ndarray, np.ndarray]:
        """Encode SMILES to BPE tokens"""
        tokens = self.bpe.process_line(smiles).split()
        try:
            indices = np.asarray([self.words2idx[t] for t in tokens])
        except KeyError:
            indices = np.array([0])

        length = len(indices)
        if length < self.max_length:
            padded = np.pad(indices, (0, self.max_length - length), 'constant', constant_values=0)
            mask = [1] * length + [0] * (self.max_length - length)
        else:
            padded = indices[:self.max_length]
            mask = [1] * self.max_length

        return padded, np.asarray(mask)


# ==============================================================================
# [FIXED] DRUG DATA LOADER - DO NOT MODIFY
# ==============================================================================
class DrugDataLoader:
    """Drug Data Loader

    [FIXED] Uses the same data loading interface as DeepTTA.
    DO NOT MODIFY this class.
    """

    def __init__(self, config: Dict, vocab_path: str, base_path: str = None):
        self.config = config
        self.vocab_path = vocab_path
        if base_path is None:
            base_path = config.get('data', {}).get('base_path', '/workspace/datasets/shared')
        self.base_path = base_path
        self.cache_dir = config.get('data', {}).get('preprocessing', {}).get('cache_dir', './cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.smiles_encoder = SMILESEncoder(vocab_path, config)

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

    def get_drug_features(self, drug_names: List[str]) -> Tuple[Dict, List]:
        """Extract drug features"""
        cache_file = os.path.join(self.cache_dir, 'drug_features_deeptta.pkl')
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
                encoding, mask = self.smiles_encoder.encode(drug_smiles[drug_name])
                drug_dict[drug_name] = (encoding, mask)
            except Exception:
                failed.append(drug_name)

        with open(cache_file, 'wb') as f:
            pickle.dump(drug_dict, f)

        print(f"Generated {len(drug_dict)} drug features, {len(failed)} failed")
        return drug_dict, failed
