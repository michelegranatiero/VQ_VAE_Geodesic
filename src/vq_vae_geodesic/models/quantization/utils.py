"""
Utility functions for geodesic quantization.

Contains helper functions for:
- Subsampling large datasets
- Building k-NN graphs for manifold structure
- Computing medoids for cluster centers
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


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
        return np.arange(N)
    rng = np.random.RandomState(seed)
    return rng.choice(N, size=max_pts, replace=False)


def build_knn_graph(Z, k=20):
    """
    Build a k-nearest neighbors graph for manifold approximation.

    Connects each point to its k nearest neighbors with edges
    weighted by Euclidean distance. The graph is symmetrized.

    Args:
        Z: Data points (N, D)
        k: Number of nearest neighbors

    Returns:
        A: Sparse adjacency matrix (N, N) with distances as weights
    """
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(Z)
    distances, indices = nn.kneighbors(Z)

    rows, cols, data = [], [], []
    N = Z.shape[0]
    for i in range(N):
        for j_idx, dist in zip(indices[i], distances[i]):
            rows.append(i)
            cols.append(j_idx)
            data.append(dist)

    # Symmetrize the graph (make undirected)
    rows2 = rows + cols
    cols2 = cols + rows
    data2 = data + data
    A = csr_matrix((data2, (rows2, cols2)), shape=(N, N))
    return A


def compute_medoids(Z, labels, n_clusters):
    """
    Compute cluster medoids (most representative actual points).

    For each cluster, finds the point that minimizes the sum of
    squared distances to all other points in the cluster.
    Medoids are better than centroids for preserving data distribution.

    Args:
        Z: Data points (N, D)
        labels: Cluster assignments (N,)
        n_clusters: Number of clusters

    Returns:
        medoid_indices: Array of medoid indices (n_clusters,)
        medoids: Medoid points (n_clusters, D)
    """
    medoid_indices = []
    medoids = []

    for c in range(n_clusters):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            # Empty cluster: use zero vector
            medoid_indices.append(None)
            medoids.append(np.zeros(Z.shape[1]))
            continue

        # Get all points in this cluster
        Zc = Z[idxs]

        # Compute pairwise squared distances
        dists = np.sum((Zc[:, None, :] - Zc[None, :, :])**2, axis=2)

        # Find point with minimum sum of distances (medoid)
        medoid_local = np.argmin(dists.sum(axis=1))
        medoid_idx = idxs[medoid_local]

        medoid_indices.append(medoid_idx)
        medoids.append(Z[medoid_idx])

    return np.array(medoid_indices), np.vstack(medoids)
