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
        self.pre_epochs = training_config.get('pre_epochs', 500)
        self.epochs = training_config.get('epochs', 500)
        self.learning_rate = training_config.get('learning_rate', 0.001)
        self.weight_decay = training_config.get('weight_decay', 0.0001)
        self.gradient_clipping = training_config.get('gradient_clipping', 5.0)

        loss_weights = training_config.get('loss_weights', {})
        self.mse_weight = loss_weights.get('mse', 10.0)
        self.bce_kld_weight = loss_weights.get('bce_kld', 0.1)
        self.kl_weight = loss_weights.get('kl', 1.0)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        output_path = config['output'].get('result_path', './')
        self.metric_tracker = MetricTracker(config, output_path)

    def compute_loss(
        self,
        x: torch.Tensor,
        de_feat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        gnn_z: torch.Tensor,
        adj_label: torch.Tensor,
        norm: float,
    ) -> torch.Tensor:
        mse_loss = F.mse_loss(de_feat, x)

        adj_pred = torch.sigmoid(torch.mm(gnn_z, gnn_z.t()))
        bce_loss = norm * F.binary_cross_entropy(adj_pred, adj_label)

        n_nodes = x.shape[0]
        kld_loss = -0.5 / n_nodes * torch.mean(
            torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), dim=1)
        )

        total_loss = (
            self.mse_weight * mse_loss +
            self.bce_kld_weight * (bce_loss + kld_loss)
        )

        return total_loss

    def train_epoch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adj_label: torch.Tensor,
        norm: float,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        z, mu, logvar, de_feat, q, feat_x, gnn_z = self.model(x=x, edge_index=edge_index)

        loss = self.compute_loss(x, de_feat, mu, logvar, gnn_z, adj_label, norm)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

        return loss.item()

    def train_epoch_with_kl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adj_label: torch.Tensor,
        norm: float,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        z, mu, logvar, de_feat, q, feat_x, gnn_z = self.model(x=x, edge_index=edge_index)

        base_loss = self.compute_loss(x, de_feat, mu, logvar, gnn_z, adj_label, norm)

        p = self.model.target_distribution(q)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

        total_loss = base_loss + self.kl_weight * kl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

        return total_loss.item()

    def train(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adj_label: torch.Tensor,
        norm: float,
        adata=None,
    ) -> np.ndarray:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        adj_label = adj_label.to(self.device)

        print(f'Pre-training for {self.pre_epochs} epochs...')
        for epoch in tqdm(range(1, self.pre_epochs + 1)):
            loss = self.train_epoch(x, edge_index, adj_label, norm)
            self.metric_tracker.log_epoch(epoch, loss, phase='pretrain')

        print(f'Training with KL for {self.epochs} epochs...')
        for epoch in tqdm(range(1, self.epochs + 1)):
            loss = self.train_epoch_with_kl(x, edge_index, adj_label, norm)
            self.metric_tracker.log_epoch(self.pre_epochs + epoch, loss, phase='train')

        self.metric_tracker.save()
        print("Optimization finished!")

        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(x=x, edge_index=edge_index)

        return z.cpu().numpy()
