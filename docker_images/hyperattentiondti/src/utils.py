"""Utility functions for HyperAttentionDTI"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

from components.drug_encoder_hyperattentiondti import CHARISOSMISET, DrugDataLoader
from components.protein_encoder_hyperattentiondti import CHARPROTSET, ProteinDataLoader


def set_seed(seed: int = 1234):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def mkdir(path: str):
    """Create directory if not exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def label_smiles(smiles: str, max_length: int = 100) -> np.ndarray:
    """Integer encoding for SMILES string"""
    encoding = np.zeros(max_length, dtype=np.int64)
    for idx, ch in enumerate(smiles[:max_length]):
        encoding[idx] = CHARISOSMISET.get(ch, 0)
    return encoding


def label_sequence(sequence: str, max_length: int = 1000) -> np.ndarray:
    """Integer encoding for protein sequence"""
    encoding = np.zeros(max_length, dtype=np.int64)
    for idx, ch in enumerate(sequence[:max_length]):
        encoding[idx] = CHARPROTSET.get(ch, 0)
    return encoding


class DTIDataset(Dataset):
    """Drug-Target Interaction Dataset

    Loads data from text file format:
    drug_id protein_id SMILES protein_sequence label
    """

    def __init__(self, data_list: List[str], drug_max_len: int = 100, protein_max_len: int = 1000):
        self.data_list = data_list
        self.drug_max_len = drug_max_len
        self.protein_max_len = protein_max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch_data: List[str], drug_max_len: int = 100, protein_max_len: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader

    Args:
        batch_data: List of data strings (drug_id protein_id SMILES protein label)
        drug_max_len: Maximum SMILES length
        protein_max_len: Maximum protein length

    Returns:
        Tuple of (drug_tensor, protein_tensor, label_tensor)
    """
    N = len(batch_data)
    drug_tensor = torch.zeros((N, drug_max_len), dtype=torch.long)
    protein_tensor = torch.zeros((N, protein_max_len), dtype=torch.long)
    label_tensor = torch.zeros(N, dtype=torch.long)

    for i, pair in enumerate(batch_data):
        parts = pair.strip().split()
        # Format: drug_id protein_id SMILES protein_sequence label
        smiles = parts[-3]
        protein = parts[-2]
        label = parts[-1]

        drug_tensor[i] = torch.from_numpy(label_smiles(smiles, drug_max_len))
        protein_tensor[i] = torch.from_numpy(label_sequence(protein, protein_max_len))
        label_tensor[i] = int(float(label))

    return drug_tensor, protein_tensor, label_tensor


def create_collate_fn(drug_max_len: int = 100, protein_max_len: int = 1000):
    """Create collate function with specific max lengths"""
    def _collate_fn(batch_data):
        return collate_fn(batch_data, drug_max_len, protein_max_len)
    return _collate_fn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path: str = None, patience: int = 50, verbose: bool = True, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, epoch: int):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        if self.save_path:
            torch.save(model.state_dict(), os.path.join(self.save_path, 'best_checkpoint.pth'))
        self.val_loss_min = val_loss


def load_data(file_path: str) -> List[str]:
    """Load data from text file"""
    with open(file_path, 'r') as f:
        data = f.read().strip().split('\n')
    return data


def shuffle_dataset(dataset: List, seed: int) -> List:
    """Shuffle dataset with given seed"""
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def get_class_weights(dataset_name: str, config: Dict) -> torch.Tensor:
    """Get class weights for imbalanced dataset"""
    weights_config = config.get('data', {}).get('class_weights', {})
    weights = weights_config.get(dataset_name, None)

    if weights is not None:
        return torch.FloatTensor(weights)
    return None
