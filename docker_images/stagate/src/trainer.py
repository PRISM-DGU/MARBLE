import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from runtime_validator import ErrorDiagnostics, RuntimeValidator


class MetricTracker:
    def __init__(self, config: Dict, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.csv_path = self.output_dir / 'training_metrics.csv'

    def log_epoch(self, epoch: int, loss: float, **metrics):
        entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'loss': loss,
            **metrics
        }
        self.metrics_history.append(entry)

        if epoch % 100 == 0:
            self.save()

    def save(self):
        if not self.metrics_history:
            return

        fieldnames = list(self.metrics_history[0].keys())

        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(self.metrics_history)

        print(f"Metrics saved to {self.csv_path}")


class Trainer:
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        training_config = config['training']
        self.epochs = training_config.get('epochs', 1000)
        self.learning_rate = training_config.get('learning_rate', 0.001)
        self.weight_decay = training_config.get('weight_decay', 0.0001)
        self.gradient_clipping = training_config.get('gradient_clipping', 5.0)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        output_path = config['output'].get('result_path', './')
        self.metric_tracker = MetricTracker(config, output_path)

    def _safe_forward(self, x: torch.Tensor, edge_index: torch.Tensor, adata=None, **kwargs):
        """
        Safe forward pass with error diagnostics.

        Catches common errors and provides detailed diagnostics for debugging.
        """
        try:
            return self.model(x=x, edge_index=edge_index, adata=adata, **kwargs)
        except RuntimeError as e:
            error_msg = str(e).lower()

            # CUDA index error
            if 'cuda' in error_msg and ('index' in error_msg or 'assert' in error_msg):
                print("\n" + "=" * 60)
                print("CUDA INDEX ERROR DETECTED - Running diagnostics...")
                print("=" * 60)
                print(ErrorDiagnostics.diagnose_cuda_index_error(x, edge_index, self.model))

                # Try to fix and retry
                print("\n[RuntimeValidator] Attempting automatic fix...")
                edge_index = RuntimeValidator.validate_edge_index(edge_index, x.shape[0])
                RuntimeValidator.fix_gatconv_dimensions(self.model)

                try:
                    return self.model(x=x, edge_index=edge_index, adata=adata, **kwargs)
                except Exception as retry_error:
                    print(f"[RuntimeValidator] Retry failed: {retry_error}")
                    raise

            # Shape mismatch error
            elif 'size mismatch' in error_msg or 'shape' in error_msg:
                print("\n" + "=" * 60)
                print("TENSOR SHAPE ERROR DETECTED - Running diagnostics...")
                print("=" * 60)
                print(ErrorDiagnostics.diagnose_tensor_mismatch({
                    'x': x,
                    'edge_index': edge_index
                }))
                raise

            else:
                raise

    def train_epoch(self, x: torch.Tensor, edge_index: torch.Tensor = None, adata=None, **kwargs) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        z, recon = self._safe_forward(x=x, edge_index=edge_index, adata=adata, **kwargs)
        loss = F.mse_loss(x, recon)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

        return loss.item()

    def train(self, x: torch.Tensor, edge_index: torch.Tensor = None, adata=None, **kwargs) -> np.ndarray:
        x = x.to(self.device)
        if edge_index is not None:
            edge_index = edge_index.to(self.device)

        # Pre-training validation
        print("[RuntimeValidator] Running pre-training validation...")
        edge_index = RuntimeValidator.validate_edge_index(edge_index, x.shape[0])

        print(f'Begin training for {self.epochs} epochs...')

        for epoch in tqdm(range(1, self.epochs + 1)):
            loss = self.train_epoch(x=x, edge_index=edge_index, adata=adata, **kwargs)
            self.metric_tracker.log_epoch(epoch, loss)

        self.metric_tracker.save()
        print("Optimization finished!")

        self.model.eval()
        with torch.no_grad():
            z, recon = self._safe_forward(x=x, edge_index=edge_index, adata=adata, **kwargs)

        return z.cpu().numpy()
