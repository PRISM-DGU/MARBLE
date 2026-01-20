"""DeepDR Drug Encoder - Graph + MPG (Message Passing Graph) encoder

This is the optimal drug encoder according to DeepDR paper benchmarks.
Uses molecular graphs as input and MPG (pre-trained MolGNet) for encoding.

Reference: DeepDR paper - EXP + Graph + MPG achieves best R2 (0.7688)
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from torch_geometric.data import Data as PyGData, Batch
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: torch_geometric not installed. Graph features will be limited.")


# Atom and bond feature dimensions (following DeepDR)
ATOM_FEATURES = {
    'atom_type': ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                  'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
                  'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                  'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'],
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'num_hs': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'Other'],
}

BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    'stereo': ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'],
}


def one_hot_encoding(x, allowable_set):
    """One-hot encoding with unknown handling"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_atom_features(atom) -> List[float]:
    """Extract atom features for GNN"""
    features = []
    features += one_hot_encoding(atom.GetSymbol(), ATOM_FEATURES['atom_type'])
    features += one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree'])
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    features += one_hot_encoding(str(atom.GetHybridization()), ATOM_FEATURES['hybridization'])
    features.append(atom.GetIsAromatic())
    return features


def get_bond_features(bond) -> List[float]:
    """Extract bond features for GNN"""
    features = []
    features += one_hot_encoding(str(bond.GetBondType()), BOND_FEATURES['bond_type'])
    features += one_hot_encoding(str(bond.GetStereo()), BOND_FEATURES['stereo'])
    features.append(bond.GetIsConjugated())
    features.append(bond.IsInRing())
    return features


def smiles_to_graph(smiles: str) -> Optional[PyGData]:
    """Convert SMILES to PyG graph data"""
    if not HAS_PYG:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Get bond indices and features
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
        bond_feat = get_bond_features(bond)
        edge_attrs += [bond_feat, bond_feat]

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float)

    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MolGNet(nn.Module):
    """Simplified MolGNet for molecular graph encoding

    Based on the MPG (Message Passing Graph) architecture from DeepDR.
    Uses multi-layer GCN with attention-like message passing.
    """

    def __init__(self, num_layer: int = 5, emb_dim: int = 768,
                 input_dim: int = 78, heads: int = 12,
                 num_message_passing: int = 3, drop_ratio: float = 0.0):
        super().__init__()

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        # Input projection
        self.input_proj = nn.Linear(input_dim, emb_dim)

        # GCN layers for message passing
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.convs.append(GCNConv(emb_dim, emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        # Output projection
        self.output_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Input projection
        x = self.input_proj(x)

        # Message passing layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            if self.drop_ratio > 0:
                x_new = F.dropout(x_new, p=self.drop_ratio, training=self.training)
            x = x + x_new  # Residual connection

        # Output projection
        x = self.output_proj(x)
        return x


class DrugEncoder(nn.Module):
    """Graph + MPG Drug Encoder (DeepDR optimal architecture)

    Architecture: Molecular Graph -> MPG (MolGNet) -> GCNConv -> Drug embedding

    This encoder achieves the best performance according to DeepDR benchmarks:
    - Drug feature: Graph (molecular graph)
    - Drug encoder: MPG (Message Passing Graph / MolGNet)
    - Best R2: 0.7688 (with EXP cell encoder and DNN fusion)

    Config parameters:
        input.dim: Not used for graph (determined by atom features)
        architecture.mpg_dim: MPG feature dimension (default: 768)
        architecture.freeze: Use pre-computed MPG features (default: True)
        architecture.num_layer: Number of GCN layers (default: 5)
        architecture.dropout: Dropout rate
        output.dim: Output embedding dimension (default: 128)
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__()

        if not HAS_PYG:
            raise ImportError("torch_geometric is required for Graph+MPG encoder. "
                            "Install with: pip install torch-geometric torch-scatter torch-sparse")

        # Read dimensions from config
        self.output_dim = config.get('output', {}).get('dim', 128)
        arch = config.get('architecture', {})

        # MPG parameters
        self.mpg_dim = arch.get('mpg_dim', 768)
        self.freeze = arch.get('freeze', True)
        self.use_conv = arch.get('use_conv', True)
        self.pool_type = arch.get('pool_type', 'mean')  # mean, max, or mix

        # Atom feature dimension (calculated from ATOM_FEATURES)
        atom_feat_dim = (len(ATOM_FEATURES['atom_type']) +
                        len(ATOM_FEATURES['degree']) +
                        len(ATOM_FEATURES['formal_charge']) +
                        len(ATOM_FEATURES['num_hs']) +
                        len(ATOM_FEATURES['hybridization']) + 1)  # +1 for is_aromatic

        # Build MPG encoder (if not using frozen features)
        if not self.freeze:
            num_layer = arch.get('num_layer', 5)
            dropout = arch.get('dropout', 0.0)
            self.mpg = MolGNet(
                num_layer=num_layer,
                emb_dim=self.mpg_dim,
                input_dim=atom_feat_dim,
                drop_ratio=dropout
            )

            # Load pre-trained weights if provided
            pt_path = arch.get('pretrained_path', None)
            if pt_path and os.path.exists(pt_path):
                self.mpg.load_state_dict(torch.load(pt_path))
                print(f"Loaded pre-trained MPG weights from {pt_path}")

        # Output GCN conv layer (following DeepDR)
        if self.use_conv:
            self.output_conv = GCNConv(self.mpg_dim, self.output_dim)
        else:
            self.output_proj = nn.Linear(self.mpg_dim, self.output_dim)

        # Store config for data loader
        self._config = config

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, data, **kwargs) -> Tuple[torch.Tensor, Any]:
        """Forward pass

        Args:
            data: PyG Batch object containing:
                - x: Node features (atom features)
                - edge_index: Edge connectivity
                - mpg_ft: Pre-computed MPG features (if freeze=True)
                - batch: Batch assignment vector

        Returns:
            Tuple of (node_embeddings, graph_data) for fusion module
            - node_embeddings: (total_nodes, output_dim)
            - graph_data: Original data for pooling in fusion
        """
        # Handle different input formats
        if isinstance(data, tuple):
            data = data[0]

        # Get node features
        if self.freeze and hasattr(data, 'mpg_ft'):
            # Use pre-computed MPG features
            x = data.mpg_ft
        else:
            # Compute MPG features
            x = self.mpg(data)

        # Apply output convolution or projection
        if self.use_conv:
            x = self.output_conv(x, data.edge_index)
        else:
            x = self.output_proj(x)

        return x, data


class DrugDataLoader:
    """DeepDR Drug Data Loader - Graph feature extraction

    Converts SMILES to molecular graphs with optional pre-computed MPG features.
    Following DeepDR's Graph drug feature pipeline.
    """

    def __init__(self, config: Dict, vocab_path: str = None, base_path: str = None):
        self.config = config
        self.vocab_path = vocab_path  # Path to pre-computed MPG features

        if base_path is None:
            base_path = config.get('data', {}).get('base_path', '/workspace/datasets/shared')
        self.base_path = base_path

        self.cache_dir = config.get('data', {}).get('preprocessing', {}).get('cache_dir', './cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Try to load pre-computed MPG features
        self.mpg_dict = None
        mpg_path = config.get('model', {}).get('drug_encoder', {}).get('architecture', {}).get('mpg_features_path', None)
        if mpg_path and os.path.exists(mpg_path):
            with open(mpg_path, 'rb') as f:
                self.mpg_dict = pickle.load(f)
            print(f"Loaded pre-computed MPG features for {len(self.mpg_dict)} drugs")

    def load_smiles(self) -> Dict[str, str]:
        """Load SMILES data from file"""
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
        """Extract Graph features for drugs

        Args:
            drug_names: List of drug names to process

        Returns:
            Tuple of (drug_features_dict, failed_drugs_list)
            Each feature is a PyG Data object with graph structure and optional MPG features
        """
        cache_file = os.path.join(self.cache_dir, 'drug_features_deepdr_graph.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                drug_dict = pickle.load(f)
            print(f"Loaded {len(drug_dict)} drug graph features from cache")
            return drug_dict, []

        drug_smiles = self.load_smiles()
        drug_dict = {}
        failed = []

        for drug_name in drug_names:
            if drug_name not in drug_smiles:
                failed.append(drug_name)
                continue

            try:
                graph = smiles_to_graph(drug_smiles[drug_name])
                if graph is not None:
                    # Add pre-computed MPG features if available
                    if self.mpg_dict and drug_name in self.mpg_dict:
                        graph.mpg_ft = torch.tensor(self.mpg_dict[drug_name], dtype=torch.float)
                    drug_dict[drug_name] = graph
                else:
                    failed.append(drug_name)
            except Exception as e:
                print(f"Failed to process {drug_name}: {e}")
                failed.append(drug_name)

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(drug_dict, f)

        print(f"Generated {len(drug_dict)} drug graph features, {len(failed)} failed")
        return drug_dict, failed
