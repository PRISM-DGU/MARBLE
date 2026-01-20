import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.neighbors
import torch
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, BallTree, KDTree

try:
    import scanpy as sc
except ImportError:
    sc = None


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        torch.use_deterministic_algorithms(True, warn_only=True)


def Transfer_pytorch_Data(adata, data: np.ndarray = None):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix(
        (np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])),
        shape=(adata.n_obs, adata.n_obs)
    )
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)

    if data is not None:
        x = torch.FloatTensor(data)
    elif type(adata.X) == np.ndarray:
        x = torch.FloatTensor(adata.X)
    else:
        x = torch.FloatTensor(adata.X.todense())

    pyg_data = Data(
        edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
        x=x
    )

    adj_label = torch.FloatTensor(G.toarray())
    norm = adj_label.shape[0] * adj_label.shape[0] / float((adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) * 2)

    return pyg_data, adj_label, norm


def Cal_Spatial_Net(adata, method='KDTree', k=12, rad_cutoff=150, verbose=True):
    if verbose:
        print(f'------Calculating spatial graph using {method}...')

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if method == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices[it].shape[0],
                indices[it],
                distances[it]
            )))
    else:
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1],
                indices[it, :],
                distances[it, :]
            )))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0, ]

    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index)))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net


def cal_spatial_weight(data, spatial_k=50, spatial_type="BallTree"):
    """Calculate binary spatial weight matrix based on k-nearest neighbors."""
    if spatial_type == "BallTree":
        tree = BallTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    elif spatial_type == "KDTree":
        tree = KDTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    else:
        nbrs = NearestNeighbors(n_neighbors=spatial_k + 1, algorithm='ball_tree').fit(data)
        _, indices = nbrs.kneighbors(data)

    indices = indices[:, 1:]
    spatial_weight = np.zeros((data.shape[0], data.shape[0]))
    for i in range(indices.shape[0]):
        spatial_weight[i, indices[i]] = 1

    return spatial_weight


def cal_gene_weight(data, n_components=50, gene_dist_type="cosine"):
    """Calculate gene expression similarity matrix."""
    from sklearn.metrics import pairwise_distances
    from scipy.sparse import csr_matrix

    if isinstance(data, csr_matrix):
        data = data.toarray()

    if data.shape[1] > 500:
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)

    return 1 - pairwise_distances(data, metric=gene_dist_type)


def cal_weight_matrix(
    adata,
    md_dist_type="cosine",
    gb_dist_type="correlation",
    n_components=50,
    use_morphological=True,
    spatial_k=30,
    spatial_type="BallTree",
    verbose=False,
):
    """
    Calculate combined weight matrix: spatial * gene * morphological.

    If use_morphological=True, requires adata.obsm['image_feat_pca'].
    """
    from sklearn.metrics import pairwise_distances

    # Spatial weights
    physical_distance = cal_spatial_weight(
        adata.obsm['spatial'],
        spatial_k=spatial_k,
        spatial_type=spatial_type
    )
    print(f"Spatial weights calculated. Avg neighbors: {physical_distance.sum()/adata.shape[0]:.1f}")

    # Gene expression weights
    gene_correlation = cal_gene_weight(
        data=adata.X.copy(),
        gene_dist_type=gb_dist_type,
        n_components=n_components
    )
    print("Gene expression weights calculated.")

    if verbose:
        adata.obsm["gene_correlation"] = gene_correlation
        adata.obsm["physical_distance"] = physical_distance

    # Morphological weights (if enabled and available)
    if use_morphological:
        if "image_feat_pca" not in adata.obsm:
            print("WARNING: use_morphological=True but image_feat_pca not found. Skipping morphological weights.")
            adata.obsm["weights_matrix_all"] = gene_correlation * physical_distance
        else:
            morphological_similarity = 1 - pairwise_distances(
                np.array(adata.obsm["image_feat_pca"]),
                metric=md_dist_type
            )
            morphological_similarity[morphological_similarity < 0] = 0
            print("Morphological weights calculated.")

            if verbose:
                adata.obsm["morphological_similarity"] = morphological_similarity

            # Combine all three: spatial * gene * morphological
            adata.obsm["weights_matrix_all"] = (
                physical_distance *
                gene_correlation *
                morphological_similarity
            )
    else:
        # Combine only spatial and gene
        adata.obsm["weights_matrix_all"] = gene_correlation * physical_distance

    print("Final weight matrix stored in adata.obsm['weights_matrix_all']")
    return adata


def find_adjacent_spot(adata, use_data="raw", neighbour_k=4, verbose=False):
    """Identify neighboring spots and calculate weighted contributions."""
    from scipy.sparse import csr_matrix
    from tqdm import tqdm

    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            gene_matrix = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            gene_matrix = adata.X
        else:
            gene_matrix = np.array(adata.X)
    else:
        gene_matrix = adata.obsm[use_data]

    weights_list = []
    final_coordinates = []

    for i in range(adata.shape[0]):
        current_spot = adata.obsm['weights_matrix_all'][i].argsort()[-neighbour_k:][:neighbour_k-1]
        spot_weight = adata.obsm['weights_matrix_all'][i][current_spot]
        spot_matrix = gene_matrix[current_spot]

        if spot_weight.sum() > 0:
            spot_weight_scaled = spot_weight / spot_weight.sum()
            weights_list.append(spot_weight_scaled)
            spot_matrix_scaled = spot_weight_scaled.reshape(-1, 1) * spot_matrix
            spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
        else:
            spot_matrix_final = np.zeros(gene_matrix.shape[1])
            weights_list.append(np.zeros(len(current_spot)))

        final_coordinates.append(spot_matrix_final)

    adata.obsm['adjacent_data'] = np.array(final_coordinates)
    if verbose:
        adata.obsm['adjacent_weight'] = np.array(weights_list)

    return adata


def augment_gene_data(adata, adjacent_weight=0.2):
    """Augment gene expression using neighboring spot information."""
    from scipy.sparse import csr_matrix

    if isinstance(adata.X, np.ndarray):
        augmented = adata.X + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
    elif isinstance(adata.X, csr_matrix):
        augmented = adata.X.toarray() + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
    else:
        augmented = np.array(adata.X) + adjacent_weight * adata.obsm["adjacent_data"].astype(float)

    adata.obsm["augment_gene_data"] = augmented
    return adata


def augment_adata(
    adata,
    spatial_type='BallTree',
    use_morphological=False,
    adjacent_weight=0.3,
    neighbour_k=4,
    spatial_k=30,
    md_dist_type='cosine',
    gb_dist_type='correlation',
    n_components=50,
    use_data='raw',
):
    """
    Complete pipeline for spatial transcriptomics data augmentation.

    If use_morphological=True:
        - Requires adata.obsm['image_feat_pca'] (from his_feat.py)
        - Uses spatial * gene * morphological weights

    If use_morphological=False:
        - Uses only spatial * gene weights
    """
    print('Augmenting data...')

    # Step 1: Calculate combined weight matrix
    adata = cal_weight_matrix(
        adata,
        md_dist_type=md_dist_type,
        gb_dist_type=gb_dist_type,
        n_components=n_components,
        use_morphological=use_morphological,
        spatial_k=spatial_k,
        spatial_type=spatial_type,
    )

    # Step 2: Find neighboring spots
    adata = find_adjacent_spot(
        adata,
        use_data=use_data,
        neighbour_k=neighbour_k,
    )

    # Step 3: Augment gene expression
    adata = augment_gene_data(
        adata,
        adjacent_weight=adjacent_weight,
    )

    print('Data augmentation completed')
    return adata


def graph(spatial_coords, distType='BallTree', k=12, rad_cutoff=150):
    if distType == 'BallTree':
        tree = BallTree(spatial_coords)
        distances, indices = tree.query(spatial_coords, k=k + 1)
    elif distType == 'KDTree':
        tree = KDTree(spatial_coords)
        distances, indices = tree.query(spatial_coords, k=k + 1)
    else:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(spatial_coords)
        distances, indices = nbrs.kneighbors(spatial_coords)

    n_cells = spatial_coords.shape[0]
    adj = sp.lil_matrix((n_cells, n_cells))

    for i in range(n_cells):
        for j in indices[i, 1:]:
            adj[i, j] = 1
            adj[j, i] = 1

    adj = adj.tocsr()
    adj = adj + sp.eye(adj.shape[0])

    adj_norm = normalize_adj(adj)
    adj_label = adj.toarray()

    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    return {
        'adj_norm': adj_norm,
        'adj_label': torch.FloatTensor(adj_label),
        'norm_value': norm
    }


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='DeepST_embed', random_seed=0):
    try:
        os.environ['R_HOME'] = '/usr/lib/R'
    except:
        pass

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    robjects.r.library("mclust")
    robjects.r['set.seed'](random_seed)

    run_mclust = robjects.r('''
        function(data, G, modelNames) {
            res <- Mclust(data, G=G, modelNames=modelNames)
            return(res$classification)
        }
    ''')

    with localconverter(robjects.default_converter + numpy2ri.converter):
        embedding = adata.obsm[used_obsm]
        mclust_res = np.array(run_mclust(embedding, num_cluster, modelNames))

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    return adata
