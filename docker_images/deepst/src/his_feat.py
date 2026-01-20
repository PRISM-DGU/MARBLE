#!/usr/bin/env python
"""
Image Feature Extraction Module for DeepST

Extracts CNN-based features from H&E stained histology images.

Functions:
- image_crop: Crop spot images from whole tissue image
- image_feature: Extract CNN features using pretrained models (ResNet50, VGG, etc.)
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
import random

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms


class ImageFeatureExtractor:
    """Extract CNN-based features from spot images."""

    def __init__(
        self,
        adata,
        pca_components=50,
        cnn_type='ResNet50',
        verbose=False,
        seed=88,
    ):
        """
        Initialize image feature extractor.

        Args:
            adata: AnnData object with slices_path in obs
            pca_components: Number of PCA components for dimensionality reduction
            cnn_type: CNN model type ('ResNet50', 'Resnet152', 'Vgg19', 'Vgg16', 'DenseNet121')
            verbose: Whether to print progress
            seed: Random seed for PCA
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seed = seed
        self.cnn_type = cnn_type

    def load_cnn_model(self):
        """Load pretrained CNN model."""
        model_map = {
            'ResNet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
            'Resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT),
            'Vgg19': (models.vgg19, models.VGG19_Weights.DEFAULT),
            'Vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT),
            'DenseNet121': (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        }

        if self.cnn_type not in model_map:
            raise ValueError(f"{self.cnn_type} is not valid. Options: {list(model_map.keys())}")

        model_func, weights = model_map[self.cnn_type]
        model = model_func(weights=weights)
        model.to(self.device)
        model.eval()
        return model

    def extract_features(self):
        """
        Extract image features from cropped spot images.

        Returns:
            Updated adata with image_feat and image_feat_pca in obsm
        """
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        img_to_tensor = transforms.Compose(transform_list)

        features_list = []
        spot_names = []

        model = self.load_cnn_model()

        if "slices_path" not in self.adata.obs.keys():
            raise ValueError("Please run image_crop() first to generate spot images")

        with tqdm(total=len(self.adata),
                  desc="Extracting image features",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]",
                  disable=not self.verbose) as pbar:
            for spot, slice_path in self.adata.obs['slices_path'].items():
                spot_slice = Image.open(slice_path).convert('RGB').resize((224, 224))
                spot_slice = np.asarray(spot_slice, dtype=np.float32) / 255.0

                tensor = img_to_tensor(spot_slice).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = model(Variable(tensor)).cpu().numpy().ravel()

                features_list.append(features)
                spot_names.append(spot)
                pbar.update(1)

        feat_array = np.array(features_list)
        self.adata.obsm["image_feat"] = feat_array

        # PCA reduction
        pca = PCA(n_components=min(self.pca_components, feat_array.shape[1]), random_state=self.seed)
        self.adata.obsm["image_feat_pca"] = pca.fit_transform(feat_array)

        if self.verbose:
            print(f"Image features shape: {feat_array.shape}")
            print(f"PCA-reduced features shape: {self.adata.obsm['image_feat_pca'].shape}")

        return self.adata


def image_crop(
    adata,
    save_path,
    library_id=None,
    crop_size=50,
    target_size=224,
    verbose=False,
):
    """
    Crop spot images from the whole tissue H&E image.

    Args:
        adata: AnnData object with spatial image in uns['spatial']
        save_path: Directory to save cropped images
        library_id: Library ID for spatial data (auto-detected if None)
        crop_size: Size of crop around each spot (in pixels)
        target_size: Target size for resizing (224 for CNN input)
        verbose: Whether to print progress

    Returns:
        Updated adata with slices_path in obs
    """
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    # Get the tissue image
    spatial_data = adata.uns["spatial"][library_id]
    use_quality = spatial_data.get("use_quality", "hires")

    if use_quality not in spatial_data["images"]:
        use_quality = list(spatial_data["images"].keys())[0]

    image = spatial_data["images"][use_quality]

    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    img_pillow = Image.fromarray(image)

    os.makedirs(save_path, exist_ok=True)
    tile_names = []

    # Get scale factor for coordinate conversion
    scale_factor = spatial_data["scalefactors"].get(f"tissue_{use_quality}_scalef", 1.0)

    with tqdm(total=len(adata),
              desc="Cropping spot images",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]",
              disable=not verbose) as pbar:

        for idx, (spot_id, row) in enumerate(adata.obs.iterrows()):
            # Get spot coordinates
            if "imagerow" in adata.obs.columns and "imagecol" in adata.obs.columns:
                imagerow = row["imagerow"]
                imagecol = row["imagecol"]
            else:
                # Use spatial coordinates with scale factor
                spatial_coord = adata.obsm['spatial'][idx]
                imagecol = spatial_coord[0] * scale_factor
                imagerow = spatial_coord[1] * scale_factor

            # Crop the spot region
            left = imagecol - crop_size / 2
            upper = imagerow - crop_size / 2
            right = imagecol + crop_size / 2
            lower = imagerow + crop_size / 2

            tile = img_pillow.crop((left, upper, right, lower))
            tile = tile.resize((target_size, target_size))

            tile_name = f"{spot_id}_{crop_size}.png"
            out_path = Path(save_path) / tile_name
            tile.save(out_path, "PNG")
            tile_names.append(str(out_path))

            pbar.update(1)

    adata.obs["slices_path"] = tile_names

    if verbose:
        print(f"Cropped {len(tile_names)} spot images to {save_path}")

    return adata
