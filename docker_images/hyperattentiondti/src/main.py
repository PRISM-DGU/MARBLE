"""HyperAttentionDTI Main Entry Point

Usage:
    python main.py --config ../config.yaml --epochs 100
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from pathlib import Path
from typing import Dict, Tuple, List
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    precision_recall_curve, auc
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import HyperAttentionDTI
from utils import (
    set_seed, mkdir, DTIDataset, create_collate_fn,
    EarlyStopping, load_data, shuffle_dataset, get_class_weights
)


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """HyperAttentionDTI Trainer"""

    def __init__(self, model: HyperAttentionDTI, config: Dict, device: torch.device, class_weights: torch.Tensor = None):
        self.model = model.to(device)
        self.config = config
        self.device = device

        training_config = config.get('training', {})
        self.epochs = 50  # Hardcoded to 50 epochs
        self.lr = training_config.get('learning_rate', 5e-5)
        self.batch_size = training_config.get('batch_size', 32)
        self.weight_decay = training_config.get('weight_decay', 1e-4)
        self.patience = training_config.get('patience', 50)
        self.use_cyclic_lr = training_config.get('use_cyclic_lr', True)
        self.max_lr_multiplier = training_config.get('max_lr_multiplier', 10)

        # Initialize weights with Xavier
        self._init_weights()

        # Optimizer with weight decay only for non-bias parameters
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)

        self.optimizer = optim.AdamW([
            {'params': weight_p, 'weight_decay': self.weight_decay},
            {'params': bias_p, 'weight_decay': 0}
        ], lr=self.lr)

        # Loss function with optional class weights
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        output_config = config.get('output', {})
        self.output_dir = output_config.get('result_path', './results')
        mkdir(self.output_dir)

        self.best_model_state = None
        self.best_auprc = 0
        self.best_epoch = 0

    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_scheduler(self, train_size: int):
        """Create cyclic learning rate scheduler"""
        if self.use_cyclic_lr:
            step_size = train_size // self.batch_size
            return optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.lr,
                max_lr=self.lr * self.max_lr_multiplier,
                step_size_up=step_size,
                cycle_momentum=False
            )
        return None

    def train_epoch(self, train_loader: DataLoader, scheduler=None) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for drugs, proteins, labels in pbar:
            drugs = drugs.to(self.device)
            proteins = proteins.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(drugs, proteins)
            loss = self.loss_fn(outputs, labels)

            loss.backward()
            self.optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(data_loader)

        y_true, y_pred, y_scores = [], [], []

        with torch.no_grad():
            for drugs, proteins, labels in data_loader:
                drugs = drugs.to(self.device)
                proteins = proteins.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(drugs, proteins)
                loss = self.loss_fn(outputs, labels)

                total_loss += loss.item()

                # Get predictions
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                scores = probs[:, 1]  # Probability of positive class

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)
                y_scores.extend(scores)

        # Calculate metrics
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'auroc': roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.0,
        }

        # AUPRC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        metrics['auprc'] = auc(recall_curve, precision_curve)

        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader = None) -> Dict:
        """Full training loop"""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\n")

        # Create scheduler
        train_size = len(train_loader.dataset)
        scheduler = self._create_scheduler(train_size)

        # Early stopping
        early_stopping = EarlyStopping(
            save_path=self.output_dir,
            patience=self.patience,
            verbose=True
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader, scheduler)
            val_metrics = self.evaluate(val_loader)

            # Track best model
            if val_metrics['auprc'] >= self.best_auprc:
                self.best_auprc = val_metrics['auprc']
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch

            print(f"Epoch {epoch}/{self.epochs} - "
                  f"Train Loss: {train_loss:.5f}, "
                  f"Val Loss: {val_metrics['loss']:.5f}, "
                  f"Val AUPRC: {val_metrics['auprc']:.5f}, "
                  f"Val AUROC: {val_metrics['auroc']:.5f}, "
                  f"Val Acc: {val_metrics['accuracy']:.5f}")

            # Early stopping check
            early_stopping(val_metrics['loss'], self.model, epoch)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model for final test
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Final test evaluation
        result = {}
        if test_loader is not None:
            test_metrics = self.evaluate(test_loader)
            result = test_metrics

            print(f"\n{'='*60}")
            print(f"Training completed! Best Epoch: {self.best_epoch}")
            print(f"Test AUPRC: {test_metrics['auprc']:.5f}, "
                  f"Test AUROC: {test_metrics['auroc']:.5f}, "
                  f"Test Acc: {test_metrics['accuracy']:.5f}")
            print(f"{'='*60}\n")

        # Save best model
        torch.save(self.best_model_state, os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))

        result['best_epoch'] = self.best_epoch
        return result


# ============================================================================
# Main
# ============================================================================

def _get_default_config_path():
    """Get default config path relative to script location (supports build_N iterations)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    return os.path.join(parent_dir, 'config.yaml')


def main():
    parser = argparse.ArgumentParser(description='HyperAttentionDTI Training')
    parser.add_argument('--config', type=str, default=None)  # Dynamic default
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dataset', type=str, default=None, help='Dataset: KIBA, Davis, DrugBank')
    args = parser.parse_args()

    # Dynamic config path resolution
    config_path = args.config if args.config else _get_default_config_path()
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        alt_paths = [
            os.path.join(parent_dir, 'config.yaml'),
            './config.yaml',
        ]
        for p in alt_paths:
            if os.path.exists(p):
                config_path = p
                break

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.dataset:
        config['data']['dataset'] = args.dataset

    # Setup
    seed = config.get('training', {}).get('seed', 1234)
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_config = config.get('data', {})
    base_path = data_config.get('base_path', '/workspace/datasets/hyperattentiondti')
    dataset_name = data_config.get('dataset', 'KIBA')

    data_file = os.path.join(base_path, f'{dataset_name}.txt')
    print(f"Loading dataset: {dataset_name} from {data_file}")

    data_list = load_data(data_file)
    data_list = shuffle_dataset(data_list, seed)
    print(f"Total samples: {len(data_list)}")

    # Split data (80% train, 10% val, 10% test)
    n = len(data_list)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Get max lengths from config
    drug_max_len = config.get('model', {}).get('drug_encoder', {}).get('input', {}).get('max_length', 100)
    protein_max_len = config.get('model', {}).get('protein_encoder', {}).get('input', {}).get('max_length', 1000)

    # Create datasets
    train_dataset = DTIDataset(train_data, drug_max_len, protein_max_len)
    val_dataset = DTIDataset(val_data, drug_max_len, protein_max_len)
    test_dataset = DTIDataset(test_data, drug_max_len, protein_max_len)

    batch_size = config.get('training', {}).get('batch_size', 32)
    collate = create_collate_fn(drug_max_len, protein_max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=4)

    # Get class weights for imbalanced data
    class_weights = get_class_weights(dataset_name, config)
    if class_weights is not None:
        print(f"Using class weights: {class_weights.tolist()}")

    # Create model
    model = HyperAttentionDTI(config)
    print(f"Model created with {model.get_num_params():,} parameters")

    # Train
    trainer = Trainer(model, config, device, class_weights)
    result = trainer.train(train_loader, val_loader, test_loader)

    return result


if __name__ == '__main__':
    result = main()
