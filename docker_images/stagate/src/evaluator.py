import numpy as np
import scanpy as sc
from anndata import AnnData
from typing import Dict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from utils import mclust_R


class Evaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.clustering_config = config.get('clustering', {})
        self.eval_config = config.get('evaluation', {})
        self.embedding_key = config['output'].get('embedding_key', 'STAGATE')

    def _search_resolution(self, adata: AnnData, target_clusters: int, seed: int, method: str = 'leiden') -> float:
        """Binary search for resolution that gives approximately target_clusters."""
        res_low, res_high = 0.1, 2.0
        best_res, best_diff = 1.0, float('inf')

        for _ in range(10):
            res_mid = (res_low + res_high) / 2
            if method == 'leiden':
                sc.tl.leiden(adata, resolution=res_mid, random_state=seed, key_added='_temp_cluster')
            else:
                sc.tl.louvain(adata, resolution=res_mid, random_state=seed, key_added='_temp_cluster')

            n_clusters = adata.obs['_temp_cluster'].nunique()
            diff = abs(n_clusters - target_clusters)

            if diff < best_diff:
                best_diff = diff
                best_res = res_mid

            if n_clusters == target_clusters:
                break
            elif n_clusters < target_clusters:
                res_low = res_mid
            else:
                res_high = res_mid

        print(f'Found resolution {best_res:.4f} giving ~{target_clusters} clusters')
        return best_res

    def cluster(self, adata: AnnData) -> AnnData:
        method = self.clustering_config.get('method', 'mclust')
        n_clusters = self.clustering_config.get('n_clusters', 7)
        seed = self.config['training'].get('seed', 0)

        print(f'Performing {method} clustering with {n_clusters} clusters...')

        if method == 'mclust':
            adata = mclust_R(adata, num_cluster=n_clusters, used_obsm=self.embedding_key, random_seed=seed)
            adata.obs['domain'] = adata.obs['mclust']
        elif method == 'leiden':
            sc.pp.neighbors(adata, use_rep=self.embedding_key, random_state=seed)
            # Search for resolution that gives target n_clusters
            resolution = self._search_resolution(adata, n_clusters, seed, method='leiden')
            sc.tl.leiden(adata, resolution=resolution, random_state=seed)
            adata.obs['domain'] = adata.obs['leiden']
        elif method == 'louvain':
            sc.pp.neighbors(adata, use_rep=self.embedding_key, random_state=seed)
            resolution = self._search_resolution(adata, n_clusters, seed, method='louvain')
            sc.tl.louvain(adata, resolution=resolution, random_state=seed)
            adata.obs['domain'] = adata.obs['louvain']

        return adata

    def compute_metrics(self, adata: AnnData, ground_truth_col: str = 'ground_truth') -> Dict[str, float]:
        metrics = {}
        eval_metrics = self.eval_config.get('metrics', ['ari'])

        if ground_truth_col in adata.obs.columns:
            gt = adata.obs[ground_truth_col]
            pred = adata.obs['domain']
            mask = gt.notna()

            if mask.sum() > 0:
                gt_valid = gt[mask]
                pred_valid = pred[mask]

                if 'ari' in eval_metrics:
                    metrics['ari'] = adjusted_rand_score(gt_valid, pred_valid)
                    print(f'ARI: {metrics["ari"]:.4f}')

                if 'nmi' in eval_metrics:
                    metrics['nmi'] = normalized_mutual_info_score(gt_valid, pred_valid)
                    print(f'NMI: {metrics["nmi"]:.4f}')
        else:
            print(f'Ground truth column "{ground_truth_col}" not found, skipping ARI/NMI calculation')

        if 'silhouette' in eval_metrics:
            embedding = adata.obsm[self.embedding_key]
            n_labels = adata.obs['domain'].nunique()
            if n_labels >= 2:
                metrics['silhouette'] = silhouette_score(embedding, adata.obs['domain'])
                print(f'Silhouette: {metrics["silhouette"]:.4f}')
            else:
                print(f'Silhouette: skipped (only {n_labels} cluster found)')

        return metrics
