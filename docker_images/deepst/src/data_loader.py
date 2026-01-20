import os
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict
import json

from utils import Cal_Spatial_Net, augment_adata
from his_feat import image_crop, ImageFeatureExtractor


class SpatialDataLoader:
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get('data', {})
        self.preprocessing = self.data_config.get('preprocessing', {})
        self.augmentation = self.data_config.get('augmentation', {})

    def load_data(self, data_path: str) -> sc.AnnData:
        platform = self.data_config.get('platform', 'Visium')

        if data_path.endswith('.h5ad'):
            print(f'Loading h5ad file: {data_path}')
            adata = sc.read_h5ad(data_path)
        else:
            print(f'Loading {platform} data from: {data_path}')
            adata = self._load_by_platform(data_path, platform)

        return adata

    def _load_by_platform(self, data_path: str, platform: str) -> sc.AnnData:
        if platform == 'Visium':
            return self._load_visium(data_path)
        elif platform == 'MERFISH':
            return self._load_merfish(data_path)
        elif platform == 'slideSeq':
            return self._load_slideseq(data_path)
        elif platform == 'stereoSeq':
            return self._load_stereoseq(data_path)
        else:
            raise ValueError(f"Unknown platform: {platform}")

    def _find_h5_file(self, data_path: str) -> str:
        """Find the h5 matrix file in data_path."""
        import glob
        h5_file = os.path.join(data_path, 'filtered_feature_bc_matrix.h5')
        if os.path.exists(h5_file):
            return h5_file
        pattern = os.path.join(data_path, '*_filtered_feature_bc_matrix.h5')
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No h5 matrix file found in {data_path}")

    def _load_visium(self, data_path: str) -> sc.AnnData:
        h5_file = self._find_h5_file(data_path)
        adata = sc.read_10x_h5(h5_file)

        spatial_dir = os.path.join(data_path, 'spatial')
        tissue_positions = os.path.join(spatial_dir, 'tissue_positions_list.csv')

        if os.path.exists(tissue_positions):
            positions = pd.read_csv(
                tissue_positions, header=None,
                names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col'],
                index_col=0
            )
            positions = positions.loc[positions.index.isin(adata.obs_names)]
            adata = adata[positions.index].copy()
            adata.obsm['spatial'] = positions[['pxl_row', 'pxl_col']].values
            # Store imagerow/imagecol for image cropping
            adata.obs['imagerow'] = positions['pxl_row'].values
            adata.obs['imagecol'] = positions['pxl_col'].values
            adata.obs['array_row'] = positions['array_row'].values
            adata.obs['array_col'] = positions['array_col'].values

        # Load spatial image for morphological features
        self._load_spatial_image(adata, data_path)

        metadata_file = os.path.join(data_path, 'metadata.tsv')
        if os.path.exists(metadata_file):
            metadata = pd.read_csv(metadata_file, sep='\t', index_col=0)
            common_idx = adata.obs_names.intersection(metadata.index)
            if len(common_idx) > 0:
                adata = adata[common_idx].copy()
                adata.obs['Ground Truth'] = metadata.loc[common_idx, 'cluster'].values
                mask = adata.obs['Ground Truth'].notna()
                adata = adata[mask].copy()
                print(f'Loaded ground truth, removed {(~mask).sum()} unlabeled spots')

        adata.var_names_make_unique()
        return adata

    def _load_spatial_image(self, adata: sc.AnnData, data_path: str):
        """Load H&E image and scalefactors for Visium data."""
        spatial_dir = os.path.join(data_path, 'spatial')

        # Load scalefactors
        scalefactors_file = os.path.join(spatial_dir, 'scalefactors_json.json')
        if os.path.exists(scalefactors_file):
            with open(scalefactors_file, 'r') as f:
                scalefactors = json.load(f)
        else:
            scalefactors = {
                'tissue_hires_scalef': 1.0,
                'tissue_lowres_scalef': 1.0,
                'spot_diameter_fullres': 100
            }

        # Initialize spatial dict
        library_id = os.path.basename(data_path.rstrip('/'))
        adata.uns['spatial'] = {library_id: {'scalefactors': scalefactors, 'images': {}}}

        from PIL import Image

        # Try multiple image paths in order of priority
        image_paths = [
            # 1. image.tif/png/jpg in data_path root
            (os.path.join(data_path, 'image.tif'), 'hires'),
            (os.path.join(data_path, 'image.png'), 'hires'),
            (os.path.join(data_path, 'image.jpg'), 'hires'),
            # 2. Standard Visium spatial folder structure
            (os.path.join(spatial_dir, 'tissue_hires_image.png'), 'hires'),
            (os.path.join(spatial_dir, 'tissue_lowres_image.png'), 'lowres'),
            (os.path.join(spatial_dir, 'tissue_hires_image.tif'), 'hires'),
            (os.path.join(spatial_dir, 'tissue_lowres_image.tif'), 'lowres'),
        ]

        image_loaded = False
        for img_path, quality in image_paths:
            if os.path.exists(img_path):
                try:
                    img = np.array(Image.open(img_path).convert('RGB'))
                    adata.uns['spatial'][library_id]['images'][quality] = img
                    adata.uns['spatial'][library_id]['use_quality'] = quality
                    print(f'Loaded image: {img_path} (shape: {img.shape})')
                    image_loaded = True
                    break
                except Exception as e:
                    print(f'WARNING: Failed to load {img_path}: {e}')
                    continue

        if not image_loaded:
            print('WARNING: No spatial image found. Morphological features will not be available.')

    def _load_merfish(self, data_path: str) -> sc.AnnData:
        counts_file = os.path.join(data_path, 'counts.h5ad')
        if os.path.exists(counts_file):
            return sc.read_h5ad(counts_file)
        raise FileNotFoundError(f"MERFISH data not found at {data_path}")

    def _load_slideseq(self, data_path: str) -> sc.AnnData:
        counts_file = os.path.join(data_path, 'counts.h5ad')
        if os.path.exists(counts_file):
            return sc.read_h5ad(counts_file)
        raise FileNotFoundError(f"slideSeq data not found at {data_path}")

    def _load_stereoseq(self, data_path: str) -> sc.AnnData:
        counts_file = os.path.join(data_path, 'counts.h5ad')
        if os.path.exists(counts_file):
            return sc.read_h5ad(counts_file)
        raise FileNotFoundError(f"stereoSeq data not found at {data_path}")

    def preprocess(self, adata: sc.AnnData) -> sc.AnnData:
        if 'highly_variable' not in adata.var.columns:
            print('Preprocessing and selecting highly variable genes...')
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            n_top_genes = self.preprocessing.get('n_top_genes', 3000)
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        return adata

    def build_spatial_network(self, adata: sc.AnnData) -> sc.AnnData:
        if 'Spatial_Net' not in adata.uns.keys():
            print('Constructing spatial network...')
            spatial_config = self.data_config.get('spatial_network', {})
            method = spatial_config.get('method', 'KDTree')
            k = spatial_config.get('k', 12)
            rad_cutoff = spatial_config.get('rad_cutoff', 150)

            Cal_Spatial_Net(adata, method=method, k=k, rad_cutoff=rad_cutoff)

        return adata

    def extract_image_features(self, adata: sc.AnnData, data_path: str) -> sc.AnnData:
        """Extract CNN-based image features from H&E image."""
        image_config = self.augmentation.get('image', {})

        if not self.augmentation.get('use_morphological', False):
            print('Skipping image feature extraction (use_morphological=False)')
            return adata

        # Check if image is available
        if 'spatial' not in adata.uns or not adata.uns['spatial']:
            print('WARNING: No spatial image available. Skipping image features.')
            return adata

        library_id = list(adata.uns['spatial'].keys())[0]
        if 'images' not in adata.uns['spatial'][library_id] or not adata.uns['spatial'][library_id]['images']:
            print('WARNING: No images in adata.uns. Skipping image features.')
            return adata

        print('Extracting image features...')

        # Step 1: Crop spot images
        crop_save_path = os.path.join(data_path, 'spot_crops')
        crop_size = image_config.get('crop_size', 50)
        target_size = image_config.get('target_size', 224)

        adata = image_crop(
            adata,
            save_path=crop_save_path,
            crop_size=crop_size,
            target_size=target_size,
            verbose=True
        )

        # Step 2: Extract CNN features
        cnn_type = image_config.get('cnn_type', 'ResNet50')
        pca_components = image_config.get('pca_components', 50)

        extractor = ImageFeatureExtractor(
            adata,
            pca_components=pca_components,
            cnn_type=cnn_type,
            verbose=True
        )
        adata = extractor.extract_features()

        print(f'Image features extracted: {adata.obsm["image_feat_pca"].shape}')
        return adata

    def augment(self, adata: sc.AnnData) -> sc.AnnData:
        if not self.augmentation.get('enabled', True):
            return adata

        print('Augmenting data...')
        adata = augment_adata(
            adata,
            spatial_type=self.augmentation.get('spatial_type', 'BallTree'),
            use_morphological=self.augmentation.get('use_morphological', False),
            adjacent_weight=self.augmentation.get('adjacent_weight', 0.3),
            neighbour_k=self.augmentation.get('neighbour_k', 4),
            spatial_k=self.augmentation.get('spatial_k', 30),
        )
        return adata

    def process_data(self, adata: sc.AnnData) -> np.ndarray:
        pca_n_comps = self.preprocessing.get('pca_n_comps', 200)

        adata.raw = adata

        if 'augment_gene_data' in adata.obsm:
            adata.X = adata.obsm['augment_gene_data'].astype(np.float64)
        else:
            if hasattr(adata.X, 'toarray'):
                adata.X = adata.X.toarray().astype(np.float64)
            else:
                adata.X = adata.X.astype(np.float64)

        data = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
        data = sc.pp.log1p(data)
        data = sc.pp.scale(data)
        data = sc.pp.pca(data, n_comps=pca_n_comps)

        return data

    def load_and_preprocess(self, data_path: str) -> tuple:
        adata = self.load_data(data_path)
        adata = self.preprocess(adata)
        adata = self.build_spatial_network(adata)

        # Extract image features BEFORE augmentation (if use_morphological=True)
        adata = self.extract_image_features(adata, data_path)

        adata = self.augment(adata)
        data = self.process_data(adata)

        return adata, data
