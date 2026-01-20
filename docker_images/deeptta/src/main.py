import os
import copy
import csv
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import DeepTTAModel
from utils import calculate_metrics

# Registry에서 Data Loader 동적으로 가져오기
from components import get_cell_data_loader, get_drug_data_loader


# ============================================================================
# Data Type Detection
# ============================================================================

GRAPH_CELL_ENCODERS = {'cell_encoder_graph'}
GRAPH_DRUG_ENCODERS = {'drug_encoder_graph'}
TENSOR_CELL_ENCODERS = {'cell_encoder_deeptta', 'cell_encoder_other'}
TOKEN_DRUG_ENCODERS = {'drug_encoder_deeptta', 'drug_encoder_other'}


def is_graph_cell_encoder(encoder_type: str) -> bool:
    return encoder_type in GRAPH_CELL_ENCODERS


def is_graph_drug_encoder(encoder_type: str) -> bool:
    return encoder_type in GRAPH_DRUG_ENCODERS


# ============================================================================
# Metric Tracker
# ============================================================================

class MetricTracker:
    """Tracks and saves training metrics to CSV"""

    def __init__(self, model_name: str, total_epochs: int, output_dir: str,
                 dataset_type: str, fold: int, variant: str = ''):
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.output_dir = output_dir
        self.dataset_type = dataset_type
        self.fold = fold
        self.variant = variant
        self.metrics_history = []

        # Create output directory structure
        if variant:
            self.save_dir = Path(output_dir) / model_name / variant
        else:
            self.save_dir = Path(output_dir) / model_name / 'baseline'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.save_dir / f'fold_{fold}_metrics.csv'

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  mse: float = None, rmse: float = None, mae: float = None,
                  pcc: float = None, pcc_pval: float = None,
                  scc: float = None, scc_pval: float = None,
                  r2: float = None, ci: float = None):
        """Log metrics for an epoch"""
        timestamp = datetime.now().isoformat()

        metrics = {
            'epoch': epoch,
            'timestamp': timestamp,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        if mse is not None:
            metrics['mse'] = mse
        if rmse is not None:
            metrics['rmse'] = rmse
        if mae is not None:
            metrics['mae'] = mae
        if pcc is not None:
            metrics['pcc'] = pcc
        if pcc_pval is not None:
            metrics['pcc_pval'] = pcc_pval
        if scc is not None:
            metrics['scc'] = scc
        if scc_pval is not None:
            metrics['scc_pval'] = scc_pval
        if r2 is not None:
            metrics['r2'] = r2
        if ci is not None:
            metrics['ci'] = ci

        self.metrics_history.append(metrics)

        # Auto-save every 10 epochs
        if epoch % 10 == 0 or epoch == self.total_epochs:
            self.save()

    def save(self):
        """Save metrics to CSV file"""
        if not self.metrics_history:
            return

        # Get all unique keys from metrics history
        fieldnames = ['epoch', 'timestamp', 'train_loss', 'val_loss', 'mse', 'rmse',
                      'mae', 'pcc', 'pcc_pval', 'scc', 'scc_pval', 'r2', 'ci']

        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(self.metrics_history)

        print(f"Metrics saved to {self.csv_path}")


# ============================================================================
# Cold Split Loader
# ============================================================================

class ColdSplitLoader:

    def __init__(self, dataset_type: str = 'dataset_4_strict_blind', fold: int = 1, base_path: str = None):
        self.dataset_type = dataset_type
        self.fold = fold

        if base_path is None:
            candidate_paths = ['/workspace/datasets/shared/dragent_coldsplt', '/workspace/datasets/shared']
            for candidate in candidate_paths:
                if os.path.exists(os.path.join(candidate, dataset_type)):
                    base_path = candidate
                    break
            if base_path is None:
                base_path = candidate_paths[0]

        self.base_path = base_path
        self.split_dir = os.path.join(base_path, dataset_type)

    def load_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_path = os.path.join(self.split_dir, f'fold_{self.fold}_train.csv')
        val_path = os.path.join(self.split_dir, f'fold_{self.fold}_val.csv')
        test_path = os.path.join(self.split_dir, 'test.csv')

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        print(f"Loaded cold split {self.dataset_type} fold {self.fold}")
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df


# ============================================================================
# Unified Dataset - 모든 encoder type 지원
# ============================================================================

class UnifiedDataset(Dataset):
    """Drug-Cell 페어 데이터셋 (Graph/Tensor/Token 모두 지원)"""

    def __init__(
        self,
        drug_dict: Dict,
        cell_dict: Dict,
        data_df: pd.DataFrame,
        drug_encoder_type: str,
        cell_encoder_type: str,
        edge_index: Optional[torch.Tensor] = None
    ):
        self.drug = drug_dict
        self.cell = cell_dict
        self.drug_encoder_type = drug_encoder_type
        self.cell_encoder_type = cell_encoder_type
        self.edge_index = edge_index

        self.is_graph_cell = is_graph_cell_encoder(cell_encoder_type)
        self.is_graph_drug = is_graph_drug_encoder(drug_encoder_type)

        self.samples = []
        for _, row in data_df.iterrows():
            drug_name = row.get('drug_name')
            cosmic_id = row.get('COSMIC_ID')
            ic50 = row.get('LN_IC50', row.get('IC50'))

            if pd.notna(cosmic_id):
                cosmic_id = int(cosmic_id)
                if drug_name in drug_dict and cosmic_id in cell_dict:
                    self.samples.append((drug_name, cosmic_id, ic50))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        drug_name, cosmic_id, ic50 = self.samples[index]

        drug_data = self.drug[drug_name]
        cell_data = self.cell[cosmic_id]

        if self.is_graph_cell and self.edge_index is not None:
            if isinstance(cell_data, Data):
                cell_data = cell_data.clone()
                cell_data.edge_index = self.edge_index

        return drug_data, cell_data, ic50


def create_collate_fn(drug_encoder_type: str, cell_encoder_type: str):
    """Encoder type에 맞는 collate_fn 생성"""

    is_graph_drug = is_graph_drug_encoder(drug_encoder_type)
    is_graph_cell = is_graph_cell_encoder(cell_encoder_type)

    def collate_fn(batch):
        drug_list = [item[0] for item in batch]
        cell_list = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.float)

        # Drug batching
        if is_graph_drug:
            drug_batch = Batch.from_data_list(drug_list)
        else:
            if isinstance(drug_list[0], tuple):
                tokens = torch.tensor(np.array([d[0] for d in drug_list]), dtype=torch.long)
                masks = torch.tensor(np.array([d[1] for d in drug_list]), dtype=torch.long)
                drug_batch = (tokens, masks)
            else:
                drug_batch = torch.stack([torch.tensor(d, dtype=torch.float) for d in drug_list])

        # Cell batching
        if is_graph_cell:
            cell_batch = Batch.from_data_list(cell_list)
        else:
            if isinstance(cell_list[0], np.ndarray):
                cell_batch = torch.tensor(np.array(cell_list), dtype=torch.float)
            elif isinstance(cell_list[0], torch.Tensor):
                cell_batch = torch.stack(cell_list)
            else:
                cell_batch = torch.tensor(cell_list, dtype=torch.float)

        return drug_batch, cell_batch, labels

    return collate_fn


# ============================================================================
# Main Data Loader
# ============================================================================

class MainDataLoader:

    def __init__(self, config: Dict, vocab_path: str, dataset_type: str = 'dataset_4_strict_blind', fold: int = 1):
        self.config = config
        self.vocab_path = vocab_path
        self.dataset_type = dataset_type
        self.fold = fold

        data_config = config.get('data', {})
        self.root_dir = data_config.get('base_path', '/workspace/datasets/shared')

        model_config = config.get('model', {})
        self.cell_encoder_type = model_config.get('cell_encoder', {}).get('type', 'cell_encoder_deeptta')
        self.drug_encoder_type = model_config.get('drug_encoder', {}).get('type', 'drug_encoder_deeptta')

        print(f"Using cell_encoder: {self.cell_encoder_type}")
        print(f"Using drug_encoder: {self.drug_encoder_type}")

    def prepare_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        DrugDataLoaderClass = get_drug_data_loader(self.drug_encoder_type)
        CellDataLoaderClass = get_cell_data_loader(self.cell_encoder_type)

        # All drug encoders use the same BPE vocab path
        drug_loader = DrugDataLoaderClass(self.config, self.vocab_path, self.root_dir)

        cell_loader = CellDataLoaderClass(self.config, self.root_dir)

        unique_drugs = set()
        unique_cells = set()
        for df in [train_df, val_df, test_df]:
            unique_drugs.update(df['drug_name'].unique())
            unique_cells.update(df['COSMIC_ID'].dropna().astype(int).unique())

        drug_features_dict, failed_drugs = drug_loader.get_drug_features(list(unique_drugs))

        cell_result = cell_loader.get_cell_features(list(unique_cells))
        if len(cell_result) == 3:
            cell_features_dict, failed_cells, _ = cell_result
        else:
            cell_features_dict, failed_cells = cell_result

        print(f"Loaded {len(drug_features_dict)} drug features, {len(cell_features_dict)} cell features")

        # edge_index는 graph cell encoder일 때만 필요
        edge_index = None

        train_dataset = UnifiedDataset(
            drug_features_dict, cell_features_dict, train_df,
            self.drug_encoder_type, self.cell_encoder_type, edge_index
        )
        val_dataset = UnifiedDataset(
            drug_features_dict, cell_features_dict, val_df,
            self.drug_encoder_type, self.cell_encoder_type, edge_index
        )
        test_dataset = UnifiedDataset(
            drug_features_dict, cell_features_dict, test_df,
            self.drug_encoder_type, self.cell_encoder_type, edge_index
        )

        print(f"TRAIN dataset: {len(train_dataset)} samples")
        print(f"VAL dataset: {len(val_dataset)} samples")
        print(f"TEST dataset: {len(test_dataset)} samples")

        return train_dataset, val_dataset, test_dataset

    def get_dataloaders(self, train_dataset, val_dataset, test_dataset, batch_size: int = 32):
        collate_fn = create_collate_fn(self.drug_encoder_type, self.cell_encoder_type)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        return train_loader, val_loader, test_loader


# ============================================================================
# Trainer
# ============================================================================

class Trainer:

    def __init__(self, model: DeepTTAModel, config: Dict, device='cuda', dataset_type: str = 'dataset_4_strict_blind',
                 fold: int = None, config_path: str = None):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.dataset_type = dataset_type
        self.fold = fold
        self.config_path = config_path

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0)
        )

        self.criterion = nn.MSELoss()

        # Initialize metric tracker
        output_dir = '/workspace/output/execution'
        model_name = 'deeptta'
        variant = self._extract_variant_from_config(config_path) if config_path else ''
        epochs = config['training'].get('epochs', 100)

        self.metric_tracker = MetricTracker(
            model_name=model_name,
            total_epochs=epochs,
            output_dir=output_dir,
            dataset_type=dataset_type,
            fold=fold if fold is not None else 1,
            variant=variant
        )

    def _extract_variant_from_config(self, config_path: str) -> str:
        """Extract variant name from config file path"""
        if not config_path:
            return ''
        config_name = Path(config_path).stem
        if 'degraded' in config_name:
            if 'drug_encoder' in config_name:
                return 'degraded_drug_encoder'
            elif 'cell_encoder' in config_name:
                return 'degraded_cell_encoder'
            elif 'decoder' in config_name:
                return 'degraded_decoder'
        return 'baseline'

    def _to_device(self, data):
        if isinstance(data, Batch):
            return data.to(self.device)
        elif isinstance(data, tuple):
            return tuple(d.to(self.device) for d in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data

    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate detailed metrics for predictions"""
        metrics = {}

        # MSE
        metrics['mse'] = float(mean_squared_error(targets, predictions))
        # RMSE
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        # MAE
        metrics['mae'] = float(mean_absolute_error(targets, predictions))
        # R2
        metrics['r2'] = float(r2_score(targets, predictions))

        # Pearson correlation
        try:
            pcc, pcc_pval = pearsonr(targets, predictions)
            metrics['pcc'] = float(pcc)
            metrics['pcc_pval'] = float(pcc_pval)
        except:
            metrics['pcc'] = 0.0
            metrics['pcc_pval'] = 1.0

        # Spearman correlation
        try:
            scc, scc_pval = spearmanr(targets, predictions)
            metrics['scc'] = float(scc)
            metrics['scc_pval'] = float(scc_pval)
        except:
            metrics['scc'] = 0.0
            metrics['scc_pval'] = 1.0

        # Confidence interval (95% CI for RMSE)
        try:
            residuals = targets - predictions
            ci = 1.96 * np.std(residuals) / np.sqrt(len(residuals))
            metrics['ci'] = float(ci)
        except:
            metrics['ci'] = 0.0

        return metrics

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0

        for drug_batch, cell_batch, labels in tqdm(train_loader, desc="Training", leave=False):
            drug_batch = self._to_device(drug_batch)
            cell_batch = self._to_device(cell_batch)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(drug_batch, cell_batch)
            loss = self.criterion(predictions.squeeze(), labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for drug_batch, cell_batch, labels in tqdm(val_loader, desc="Evaluating", leave=False):
                drug_batch = self._to_device(drug_batch)
                cell_batch = self._to_device(cell_batch)
                labels = labels.to(self.device)

                predictions = self.model(drug_batch, cell_batch)
                loss = self.criterion(predictions.squeeze(), labels)
                total_loss += loss.item()

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / len(val_loader)
        metrics = calculate_metrics(all_labels, all_predictions)

        return avg_loss, metrics

    def train(self, train_loader, val_loader) -> None:
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\n")

        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training'].get('early_stopping_patience', 10)
        epochs = self.config['training']['epochs']
        best_model_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            # Log metrics to tracker
            self.metric_tracker.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                mse=val_metrics.get('mse'),
                rmse=val_metrics.get('rmse'),
                mae=val_metrics.get('mae'),
                pcc=val_metrics.get('pcc'),
                pcc_pval=val_metrics.get('pcc_pval'),
                scc=val_metrics.get('scc'),
                scc_pval=val_metrics.get('scc_pval'),
                r2=val_metrics.get('r2'),
                ci=val_metrics.get('ci')
            )

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"PCC: {val_metrics.get('pcc', 0):.4f}, SCC: {val_metrics.get('scc', 0):.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Final save of metrics
        self.metric_tracker.save()

        print(f"\n{'='*60}")
        print(f"Training completed! Best Val Loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")


# ============================================================================
# Training Functions
# ============================================================================

def save_predictions_csv(trainer: Trainer, test_loader, test_df: pd.DataFrame,
                         dataset_type: str, fold: int):
    """Save test set predictions to CSV"""
    trainer.model.eval()
    all_predictions = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        batch_idx = 0
        for drug_batch, cell_batch, labels in tqdm(test_loader, desc="Predicting", leave=False):
            drug_batch = trainer._to_device(drug_batch)
            cell_batch = trainer._to_device(cell_batch)
            labels = labels.to(trainer.device)

            predictions = trainer.model(drug_batch, cell_batch)

            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Track sample indices for this batch
            batch_size = len(labels)
            for i in range(batch_size):
                sample_idx = batch_idx * test_loader.batch_size + i
                if sample_idx < len(test_df):
                    all_indices.append(sample_idx)
            batch_idx += 1

    # Create prediction dataframe
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Determine number of samples
    num_samples = min(len(all_predictions), len(test_df))

    predictions_df = pd.DataFrame({
        'cell_line_name': [test_df.iloc[i]['COSMIC_ID'] if pd.notna(test_df.iloc[i].get('COSMIC_ID')) else ''
                           for i in range(num_samples)],
        'drug_name': [test_df.iloc[i]['drug_name'] for i in range(num_samples)],
        'true': all_labels[:num_samples],
        'predicted': all_predictions[:num_samples]
    })

    # Save to CSV
    output_dir = trainer.metric_tracker.save_dir
    predictions_csv_path = output_dir / f'fold_{fold}_predictions.csv'
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"Predictions saved to {predictions_csv_path}")


def train_single_fold(config: Dict, fold: int, dataset_type: str = 'dataset_4_strict_blind', vocab_path: str = None, config_path: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_path = vocab_path or config['data'].get('vocab_dir')

    cold_split_loader = ColdSplitLoader(dataset_type, fold)
    train_df, val_df, test_df = cold_split_loader.load_split_data()

    main_loader = MainDataLoader(config, vocab_path, dataset_type, fold)
    train_dataset, val_dataset, test_dataset = main_loader.prepare_datasets(train_df, val_df, test_df)
    train_loader, val_loader, test_loader = main_loader.get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config['training']['batch_size']
    )

    model = DeepTTAModel(config, vocab_path=vocab_path)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model, config, dataset_type=dataset_type, fold=fold, config_path=config_path)
    trainer.train(train_loader, val_loader)

    if test_loader:
        test_loss, test_metrics = trainer.evaluate(test_loader)
        print(f"Fold {fold} Test Loss: {test_loss:.4f}")
        print(f"  PCC: {test_metrics.get('pcc', 0):.4f}, SCC: {test_metrics.get('scc', 0):.4f}")
        # Save predictions to CSV
        save_predictions_csv(trainer, test_loader, test_df, dataset_type, fold)


def summarize_fold_results(output_dir: str, model_name: str, variant: str):
    """Summarize results from all 5 folds and save to CSV"""
    from pathlib import Path

    results_dir = Path(output_dir) / model_name / variant

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Metrics to aggregate
    metrics_to_aggregate = ['rmse', 'mae', 'pcc', 'scc', 'r2', 'ci']

    # Collect results from all folds
    fold_results = {}
    for metric in metrics_to_aggregate:
        fold_results[metric] = []

    for fold in range(1, 6):
        csv_path = results_dir / f'fold_{fold}_metrics.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Get the last row (final epoch)
            last_row = df.iloc[-1]
            for metric in metrics_to_aggregate:
                if metric in df.columns:
                    fold_results[metric].append(float(last_row[metric]))

    # Calculate mean and std for each metric
    summary_data = {}
    for metric in metrics_to_aggregate:
        if fold_results[metric]:
            mean_val = np.mean(fold_results[metric])
            std_val = np.std(fold_results[metric])
            summary_data[metric] = (mean_val, std_val)

    # Create summary dataframe
    summary_row = {}
    summary_row['model'] = model_name
    summary_row['variant'] = variant

    for metric in metrics_to_aggregate:
        if metric in summary_data:
            mean_val, std_val = summary_data[metric]
            summary_row[f'{metric}_mean'] = mean_val
            summary_row[f'{metric}_std'] = std_val

    # Save to summary CSV
    summary_csv_path = results_dir / 'summary_results.csv'
    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"\n{'='*60}")
    print("5-FOLD SUMMARY RESULTS")
    print(f"{'='*60}")
    for metric in metrics_to_aggregate:
        if metric in summary_data:
            mean_val, std_val = summary_data[metric]
            print(f"{metric.upper():6s}: {mean_val:.4f} ± {std_val:.4f}")
    print(f"Summary saved to {summary_csv_path}")
    print(f"{'='*60}\n")

    return summary_df


def _extract_variant_from_config_path(config_path: str) -> str:
    """Extract variant name from config file path"""
    if not config_path:
        return 'baseline'
    config_name = Path(config_path).stem
    if 'degraded' in config_name:
        if 'drug_encoder' in config_name:
            return 'degraded_drug_encoder'
        elif 'cell_encoder' in config_name:
            return 'degraded_cell_encoder'
        elif 'decoder' in config_name:
            return 'degraded_decoder'
    return 'baseline'


def train_all_folds(config: Dict, dataset_type: str = 'dataset_4_strict_blind', vocab_path: str = None, config_path: str = None):
    print(f"1-FOLD COLD SPLIT TRAINING (dataset: {dataset_type})")

    for fold in range(1, 2):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/1")
        print(f"{'='*60}")
        train_single_fold(config, fold, dataset_type, vocab_path, config_path)

    print(f"\n{'='*60}")
    print("ALL FOLDS COMPLETED")
    print(f"{'='*60}")

    # Summarize results
    output_dir = '/workspace/output/execution'
    model_name = 'deeptta'
    variant = _extract_variant_from_config_path(config_path) if config_path else 'baseline'
    summarize_fold_results(output_dir, model_name, variant)


def _get_default_config_path():
    """Get default config path relative to script location (supports build_N iterations)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # experiments/build_N or docker_images/deeptta
    return os.path.join(parent_dir, 'config.yaml')


def main():
    parser = argparse.ArgumentParser(description='DeepTTA Training')
    parser.add_argument('--config', type=str, default=None)  # Dynamic default
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--vocab_path', type=str, default=None)
    parser.add_argument('--split-method', type=str, default='cold-all',
                       choices=['random', 'cold-all', 'cold-1', 'cold-2', 'cold-3', 'cold-4', 'cold-5'])
    parser.add_argument('--dataset-type', type=str, default='dataset_1_random',
                       choices=['dataset_1_random', 'dataset_2_drug_blind', 'dataset_3_cell_blind', 'dataset_4_strict_blind'])

    args = parser.parse_args()

    # Dynamic config path resolution
    config_path = args.config if args.config else _get_default_config_path()
    if not os.path.exists(config_path):
        # Try alternative locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        alt_paths = [
            os.path.join(parent_dir, 'config.yaml'),  # Script-relative (most reliable)
            './config.yaml',
        ]
        found = None
        for p in alt_paths:
            if p and os.path.exists(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please provide --config path to a valid YAML file.")
        config_path = found

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    vocab_path = args.vocab_path or config['data'].get('vocab_dir')

    if args.split_method == 'cold-all':
        train_all_folds(config, args.dataset_type, vocab_path, config_path)
    elif args.split_method.startswith('cold-'):
        fold_num = int(args.split_method.split('-')[1])
        train_single_fold(config, fold_num, args.dataset_type, vocab_path, config_path)
    else:
        raise NotImplementedError("Random split mode not implemented")


if __name__ == '__main__':
    main()
