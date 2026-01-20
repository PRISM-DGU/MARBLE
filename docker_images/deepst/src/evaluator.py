import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import distance
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score
from typing import Dict

from utils import mclust_R


class Evaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.clustering_config = config.get('clustering', {})
        self.eval_config = config.get('evaluation', {})
        self.embedding_key = config['output'].get('embedding_key', 'DeepST_embed')

    def _optimize_resolution(self, adata: sc.AnnData) -> float:
        resolution_range = np.arange(0.1, 2.5, 0.01)
        scores = []

        for r in resolution_range:
            sc.tl.leiden(adata, resolution=r, flavor="igraph", n_iterations=2, directed=False)
            s = calinski_harabasz_score(adata.obsm[self.embedding_key], adata.obs["leiden"])
            scores.append(s)

        best_idx = np.argmax(scores)
        best_res = resolution_range[best_idx]
        print(f"Optimal resolution: {best_res:.2f}")
        return best_res

    def _priori_resolution(self, adata: sc.AnnData, n_clusters: int) -> float:
        for res in sorted(np.arange(0.1, 2.5, 0.01), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res, flavor="igraph", n_iterations=2, directed=False)
            if len(adata.obs['leiden'].unique()) == n_clusters:
                print(f"Found resolution: {res:.2f} for {n_clusters} clusters")
                return res
        return 1.0

    def _refine_clusters(
        self,
        adata: sc.AnnData,
        cluster_key: str,
        output_key: str,
        shape: str = 'hexagon'
    ) -> sc.AnnData:
        from collections import Counter

        adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')

        if shape == 'hexagon':
            n_neighbors = 6
        else:
            n_neighbors = 4

        refined = []
        for i in range(adata.n_obs):
            distances = adj_2d[i]
            neighbor_indices = np.argsort(distances)[1:n_neighbors + 1]

            neighbor_labels = [adata.obs[cluster_key].iloc[j] for j in neighbor_indices]
            current_label = adata.obs[cluster_key].iloc[i]

            label_counts = Counter(neighbor_labels + [current_label])
            most_common = label_counts.most_common(1)[0][0]
            refined.append(most_common)

        adata.obs[output_key] = refined
        return adata

    def cluster(self, adata: sc.AnnData) -> sc.AnnData:
        method = self.clustering_config.get('method', 'leiden')
        n_clusters = self.clustering_config.get('n_clusters', 7)
        seed = self.config['training'].get('seed', 0)
        do_refine = self.clustering_config.get('refine', True)
        refine_shape = self.clustering_config.get('refine_shape', 'hexagon')

        print(f'Performing {method} clustering with {n_clusters} clusters...')

        sc.pp.neighbors(adata, use_rep=self.embedding_key, random_state=seed)

        if method == 'mclust':
            adata = mclust_R(adata, num_cluster=n_clusters, used_obsm=self.embedding_key, random_seed=seed)
            adata.obs['domain'] = adata.obs['mclust']
        elif method == 'leiden':
            res = self._priori_resolution(adata, n_clusters)
            sc.tl.leiden(adata, key_added='domain', resolution=res, flavor="igraph", n_iterations=2, directed=False)
        elif method == 'louvain':
            sc.tl.louvain(adata, resolution=1.0, random_state=seed)
            adata.obs['domain'] = adata.obs['louvain']

        if do_refine and 'spatial' in adata.obsm:
            print('Refining clusters based on spatial neighbors...')
            adata = self._refine_clusters(adata, 'domain', 'domain_refined', shape=refine_shape)
        else:
            adata.obs['domain_refined'] = adata.obs['domain']

        return adata

    def compute_metrics(self, adata: sc.AnnData, ground_truth_col: str = 'Ground Truth') -> Dict[str, float]:
        metrics = {}
        eval_metrics = self.eval_config.get('metrics', ['ari'])

        pred_col = 'domain_refined' if 'domain_refined' in adata.obs.columns else 'domain'

        if ground_truth_col in adata.obs.columns:
            gt = adata.obs[ground_truth_col]
            pred = adata.obs[pred_col]
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
            metrics['silhouette'] = silhouette_score(embedding, adata.obs[pred_col])
            print(f'Silhouette: {metrics["silhouette"]:.4f}')

        return metrics
