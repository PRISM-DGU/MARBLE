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


def setup_logger(name: str = "DeepTTA",
                 level: str = "INFO",
                 log_file: Optional[str] = None) -> logging.Logger:

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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, config_path: str) -> None:
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def calculate_metrics(y_true, y_pred, binary: bool = False) -> Dict:
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
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        pearson, pearson_p = pearsonr(y_true, y_pred)
        spearman, spearman_p = spearmanr(y_true, y_pred)
        ci = concordance_index(y_true, y_pred)
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

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


def set_random_seed(seed):
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_str):
    import torch

    if device_str.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            return torch.device('cpu')
    return torch.device('cpu')

