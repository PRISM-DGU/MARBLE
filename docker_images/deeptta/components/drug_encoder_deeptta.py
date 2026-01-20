"""DeepTTA Transformer Drug Encoder"""

import os
import codecs
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from subword_nmt.apply_bpe import BPE


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.gamma * (x - u) / torch.sqrt(s + self.eps) + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_pos, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_pos, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(pos_ids)
        return self.dropout(self.LayerNorm(embeddings))


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, attn_dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask):
        B, L, _ = x.size()
        q = self.query(x).view(B, L, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k = self.key(x).view(B, L, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        v = self.value(x).view(B, L, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size) + mask
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_heads, attn_dropout, hidden_dropout):
        super().__init__()
        self.attention = SelfAttention(hidden_size, n_heads, attn_dropout)
        self.attn_dense = nn.Linear(hidden_size, hidden_size)
        self.attn_norm = LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(hidden_dropout)
        self.ffn = nn.Linear(hidden_size, intermediate_size)
        self.ffn_out = nn.Linear(intermediate_size, hidden_size)
        self.ffn_norm = LayerNorm(hidden_size)
        self.ffn_dropout = nn.Dropout(hidden_dropout)

    def forward(self, x, mask):
        attn = self.attention(x, mask)
        attn = self.attn_dropout(self.attn_dense(attn))
        x = self.attn_norm(x + attn)
        ffn = F.relu(self.ffn(x))
        ffn = self.ffn_dropout(self.ffn_out(ffn))
        return self.ffn_norm(x + ffn)


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, hidden_size, intermediate_size, n_heads, attn_dropout, hidden_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, intermediate_size, n_heads, attn_dropout, hidden_dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DrugEncoder(nn.Module):
    """Transformer-based Drug Encoder (DeepTTA style)"""

    def __init__(self, config: Dict, vocab_path: str = None):
        super().__init__()

        input_dim = config.get('input', {}).get('vocab_size', 3000)
        output_dim = config.get('output', {}).get('dim', 128)
        arch = config.get('architecture', {})

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

        emb_size = arch.get('embedding_dim', 128)
        n_layers = arch.get('n_layers', 2)
        n_heads = arch.get('n_heads', 8)
        intermediate_size = arch.get('intermediate_size', 512)
        dropout = arch.get('dropout', 0.1)
        attn_dropout = arch.get('attention_dropout', 0.1)
        hidden_dropout = arch.get('hidden_dropout', 0.1)
        max_position = arch.get('max_position', 100)

        self.emb = Embeddings(input_dim, emb_size, max_position, dropout)
        self.encoder = TransformerEncoder(n_layers, emb_size, intermediate_size, n_heads, attn_dropout, hidden_dropout)

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, v, **kwargs):
        e = v[0].long().to(self.emb.word_embeddings.weight.device)
        e_mask = v[1].long().to(self.emb.word_embeddings.weight.device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded = self.encoder(emb.float(), ex_e_mask.float())
        return encoded[:, 0]


class SMILESEncoder:
    """SMILES to BPE token encoder"""

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


class DrugDataLoader:
    """DeepTTA Drug Data Loader"""

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
        import pickle
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
            except:
                failed.append(drug_name)

        with open(cache_file, 'wb') as f:
            pickle.dump(drug_dict, f)

        print(f"Generated {len(drug_dict)} drug features, {len(failed)} failed")
        return drug_dict, failed
