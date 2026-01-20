import os
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Dict

from utils import Cal_Spatial_Net


class SpatialDataLoader:
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get('data', {})
        self.preprocessing = self.data_config.get('preprocessing', {})

    def _find_h5_file(self, data_path: str) -> str:
        """Find the h5 matrix file in data_path."""
        h5_file = os.path.join(data_path, 'filtered_feature_bc_matrix.h5')
        if os.path.exists(h5_file):
            return h5_file
        # Try to find *_filtered_feature_bc_matrix.h5
        import glob
        pattern = os.path.join(data_path, '*_filtered_feature_bc_matrix.h5')
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No h5 matrix file found in {data_path}")

    def load_data(self, data_path: str) -> AnnData:
        if data_path.endswith('.h5ad'):
            print(f'Loading h5ad file: {data_path}')
            adata = sc.read_h5ad(data_path)
        else:
            print(f'Loading 10X Visium data from: {data_path}')
            h5_file = self._find_h5_file(data_path)
            adata = sc.read_10x_h5(h5_file)

            spatial_dir = os.path.join(data_path, 'spatial')

            tissue_positions = os.path.join(spatial_dir, 'tissue_positions_list.csv')
            if os.path.exists(tissue_positions):
                positions = pd.read_csv(tissue_positions, header=None,
                                       names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col'],
                                       index_col=0)
                positions = positions.loc[positions.index.isin(adata.obs_names)]
                adata = adata[positions.index].copy()
                adata.obsm['spatial'] = positions[['pxl_row', 'pxl_col']].values

            metadata_file = os.path.join(data_path, 'metadata.tsv')
            if os.path.exists(metadata_file):
                metadata = pd.read_csv(metadata_file, sep='\t', index_col=0)
                common_idx = adata.obs_names.intersection(metadata.index)
                if len(common_idx) > 0:
                    adata = adata[common_idx].copy()
                    adata.obs['ground_truth'] = metadata.loc[common_idx, 'cluster'].values
                    mask = adata.obs['ground_truth'].notna()
                    adata = adata[mask].copy()
                    print(f'Loaded ground truth, removed {(~mask).sum()} unlabeled spots')

            adata.var_names_make_unique()

        return adata

    def preprocess(self, adata: AnnData) -> AnnData:
        if 'highly_variable' not in adata.var.columns:
            print('Preprocessing and selecting highly variable genes...')
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            n_top_genes = self.preprocessing.get('n_top_genes', 3000)
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        return adata

    def build_spatial_network(self, adata: AnnData) -> AnnData:
        if 'Spatial_Net' not in adata.uns.keys():
            print('Constructing spatial network...')
            spatial_config = self.data_config.get('spatial_network', {})
            method = spatial_config.get('method', 'knn').upper()

            if method == 'RADIUS':
                rad_cutoff = spatial_config.get('rad_cutoff', 150)
                Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff, model='Radius')
            else:
                k_cutoff = spatial_config.get('k_cutoff', 6)
                Cal_Spatial_Net(adata, k_cutoff=k_cutoff, model='KNN')
        return adata

    def load_and_preprocess(self, data_path: str) -> AnnData:
        adata = self.load_data(data_path)
        adata = self.preprocess(adata)
        adata = self.build_spatial_network(adata)
        return adata
