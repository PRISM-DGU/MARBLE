import os
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import yaml

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

fix_seed(0)

from model import STAGATE
from data_loader import SpatialDataLoader
from trainer import Trainer
from evaluator import Evaluator
from utils import Transfer_pytorch_Data


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_device(config: dict) -> torch.device:
    device_name = config['training'].get('device', 'cuda')
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device


def _get_default_config_path():
    """Get default config path relative to script location (supports build_N iterations)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    return os.path.join(parent_dir, 'config.yaml')


def run_single_seed(config, seed, device):
    """Run training and evaluation with a single seed."""
    import copy
    config = copy.deepcopy(config)
    fix_seed(seed)

    loader = SpatialDataLoader(config)
    adata = loader.load_and_preprocess(config['data']['base_path'])

    if 'highly_variable' in adata.var.columns:
        adata_input = adata[:, adata.var['highly_variable']]
    else:
        adata_input = adata

    pyg_data = Transfer_pytorch_Data(adata_input)

    config['model']['encoder']['architecture']['in_dim'] = pyg_data.x.shape[1]
    config['model']['decoder']['architecture']['out_dim'] = pyg_data.x.shape[1]

    model = STAGATE(config)
    trainer = Trainer(model, config, device)
    embeddings = trainer.train(x=pyg_data.x, edge_index=pyg_data.edge_index, adata=adata)

    embedding_key = config['output'].get('embedding_key', 'STAGATE')
    adata.obsm[embedding_key] = embeddings

    evaluator = Evaluator(config)
    adata = evaluator.cluster(adata)
    metrics = evaluator.compute_metrics(adata)

    return metrics


def main(config_path: str = None):
    if config_path is None:
        config_path = _get_default_config_path()
    config = load_config(config_path)

    device = setup_device(config)

    # 5 seeds for robust evaluation
    seeds = [0, 42, 123, 456, 789]
    all_metrics = []

    print(f"=" * 60)
    print(f"STAGATE - Running {len(seeds)} seeds for robust evaluation")
    print(f"=" * 60)

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {i+1}/{len(seeds)}: {seed} ---")
        metrics = run_single_seed(config, seed, device)
        all_metrics.append(metrics)
        print(f"Seed {seed}: ARI={metrics.get('ari', 0):.4f}, NMI={metrics.get('nmi', 0):.4f}")

    # Calculate mean metrics
    mean_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if values:
            mean_metrics[key] = sum(values) / len(values)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY (Mean of {len(seeds)} seeds)")
    print(f"{'=' * 60}")
    print(f"ARI: {mean_metrics.get('ari', 0):.4f}")
    print(f"NMI: {mean_metrics.get('nmi', 0):.4f}")
    if 'silhouette' in mean_metrics:
        print(f"Silhouette: {mean_metrics.get('silhouette', 0):.4f}")
    print(f"{'=' * 60}")

    return mean_metrics


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
