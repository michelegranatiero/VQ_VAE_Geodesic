"""
Utility functions for geodesic quantization.

Contains helper functions for:
- Subsampling large datasets
- Building k-NN graphs for manifold structure (now uses PyTorch for W2)
- Computing medoids for cluster centers
"""
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def w2_squared_distance(mu1, sigma1, mu2, sigma2):
    """
    Compute the squared Wasserstein-2 distance between diagonal Gaussians.
    """
    mu_term = torch.sum((mu1 - mu2) ** 2, dim=-1)
    sigma_term = torch.sum((sigma1 - sigma2) ** 2, dim=-1)
    return mu_term + sigma_term

def subsample_indices(N, max_pts=5000, seed=42):
    """
    Randomly subsample indices for computational efficiency.

    Args:
        N: Total number of points
        max_pts: Maximum number of points to keep
        seed: Random seed for reproducibility

    Returns:
        indices: Array of selected indices (length min(N, max_pts))
    """
    if N <= max_pts:
        return torch.arange(N, dtype=torch.long)
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=max_pts, replace=False)
    return torch.from_numpy(idx).long()


def build_knn_graph_w2(points: torch.Tensor, sigmas: torch.Tensor, k=20):
    """
    Build k-NN graph where edge weight = W2 distance between diagonal Gaussians.
    Args:
        points: (N, d) torch.Tensor (means)
        sigmas: (N, d) torch.Tensor (stddevs)
        k: number of neighbors
    Returns:
        A: scipy.sparse adjacency matrix (N, N) with W2 distances
    """
    N, d = points.shape

    # Use sklearn NearestNeighbors to get kNN indices (on CPU, fast enough)
    # Get indices of k nearest neighbors for each point using Euclidean distance (only on means)
    points_np = points.detach().cpu().numpy()
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points_np)
    _, indices = nn.kneighbors(points_np)  # (N, k)

    # Prepare for vectorized computation
    device = points.device
    indices_torch = torch.from_numpy(indices).to(device)  # (N, k)

    # Gather neighbor points/sigmas for each point
    points_neighbors = points[indices_torch]  # (N, k, d)
    sigmas_neighbors = sigmas[indices_torch]  # (N, k, d)

    # Expand points/sigmas for broadcasting
    points_exp = points.unsqueeze(1)  # (N, 1, d)
    sigmas_exp = sigmas.unsqueeze(1)  # (N, 1, d)

    # Usa la funzione w2_squared_distance
    w = w2_squared_distance(points_exp, sigmas_exp, points_neighbors, sigmas_neighbors)  # (N, k)

    # Prepare rows, cols, data for sparse matrix
    rows = np.repeat(np.arange(N), k)
    cols = indices.flatten()
    data = w.detach().cpu().numpy().flatten()

    # symmetrize
    rows2 = np.concatenate([rows, cols])
    cols2 = np.concatenate([cols, rows])
    data2 = np.concatenate([data, data])
    A = csr_matrix((data2, (rows2, cols2)), shape=(N, N))
    return A




def build_knn_graph_w2_full(points: torch.Tensor, sigmas: torch.Tensor, k=20, batch_size=500):
    """
    Build k-NN graph where both neighbor search and edge weights use W2 distance between diagonal Gaussians.
    Uses batching to reduce memory usage.
    Args:
        points: (N, d) torch.Tensor (means)
        sigmas: (N, d) torch.Tensor (stddevs)
        k: number of neighbors
        batch_size: batch size for distance computation
    Returns:
        A: scipy.sparse adjacency matrix (N, N) with W2 distances
    """
    N, d = points.shape

    # Preallocate for indices and distances (torch)
    knn_indices = torch.zeros((N, k), dtype=torch.long, device=points.device)
    knn_dists = torch.zeros((N, k), dtype=points.dtype, device=points.device)

    # Compute distances in batches
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_points = points[start:end]  # (B, d)
        batch_sigmas = sigmas[start:end]  # (B, d)

        # Expand for broadcasting
        batch_points1 = batch_points.unsqueeze(1).expand(end-start, N, d)  # (B, N, d)
        batch_points2 = points.unsqueeze(0).expand(end-start, N, d)        # (B, N, d)
        batch_sigmas1 = batch_sigmas.unsqueeze(1).expand(end-start, N, d)
        batch_sigmas2 = sigmas.unsqueeze(0).expand(end-start, N, d)

        dists = w2_squared_distance(batch_points1, batch_sigmas1, batch_points2, batch_sigmas2)  # (B, N)
        # Exclude self (set to +inf)
        eye_idx = torch.arange(start, end, device=points.device)
        dists[torch.arange(end-start), eye_idx] = float('inf')
        # Take the k smallest
        vals, idxs = torch.topk(dists, k, dim=1, largest=False)
        knn_indices[start:end] = idxs
        knn_dists[start:end] = vals

    # Move to CPU for sparse matrix
    knn_indices_np = knn_indices.cpu().numpy()
    knn_dists_np = knn_dists.cpu().numpy()
    rows = np.repeat(np.arange(N), k)
    cols = knn_indices_np.flatten()
    data = knn_dists_np.flatten()

    # symmetrize
    rows2 = np.concatenate([rows, cols])
    cols2 = np.concatenate([cols, rows])
    data2 = np.concatenate([data, data])
    A = csr_matrix((data2, (rows2, cols2)), shape=(N, N))
    return A