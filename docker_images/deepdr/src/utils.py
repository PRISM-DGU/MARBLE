"""DeepDR Utility Functions"""

import yaml
import numpy as np
import pandas as pd
import logging
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from typing import Dict, Tuple, Optional, List, Any


def setup_logger(name: str = "DeepDR",
                 level: str = "INFO",
                 log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger with console and optional file output"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, config_path: str) -> None:
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def calculate_metrics(y_true, y_pred, binary: bool = False) -> Dict:
    """Calculate evaluation metrics

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        binary: If True, calculate classification metrics

    Returns:
        Dictionary of calculated metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if binary:
        y_pred_binary = (y_pred > 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true, y_pred_binary, zero_division=0)
        }

        if len(np.unique(y_true)) > 1:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred)
            except:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0

    else:
        # Regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        # Correlation metrics
        try:
            pearson, pearson_p = pearsonr(y_true, y_pred)
        except:
            pearson, pearson_p = 0.0, 1.0

        try:
            spearman, spearman_p = spearmanr(y_true, y_pred)
        except:
            spearman, spearman_p = 0.0, 1.0

        try:
            ci = concordance_index(y_true, y_pred)
        except:
            ci = 0.5

        try:
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        except:
            r2 = 0.0

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'pcc': pearson,
            'pcc_pval': pearson_p,
            'scc': spearman,
            'scc_pval': spearman_p,
            'r2': r2,
            'ci': ci
        }

    return metrics


def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_str: str):
    """Get torch device from string"""
    import torch

    if device_str.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            return torch.device('cpu')
    return torch.device('cpu')


def format_metrics(metrics: Dict, prefix: str = '') -> str:
    """Format metrics dictionary for display"""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return '\n'.join(lines)
