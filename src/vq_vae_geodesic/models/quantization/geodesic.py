"""
Geodesic Quantizer for VAE latent spaces.

Performs a posteriori vector quantization using geodesic distances 
instead of Euclidean distances, respecting the manifold structure 
of the latent space.
"""
import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from .utils import subsample_indices, build_knn_graph, compute_medoids


class GeodesicQuantizer:
    """
    Quantizes VAE latent vectors using geodesic distances.

    Pipeline:
    1. Split latent vectors into chunks
    2. Build k-NN graph on subsampled chunks
    3. Compute geodesic distances via shortest paths
    4. Embed in lower-dimensional space with MDS
    5. Cluster with K-means to find codebook centroids
    6. Use medoids as final codewords
    """

    def __init__(self, n_codewords=256, n_chunks=None, chunk_size=None, use_var=False, k=20, mds_dim=64, random_state=0):
        """
        Args:
            n_codewords: Size of the codebook (K centroids)
            n_chunks: Number of chunks to split each latent vector into (auto-computed if None)
            chunk_size: Dimensionality of each chunk (auto-computed if None)
            use_var: Whether to concatenate variance features
            k: Number of neighbors for k-NN graph
            mds_dim: Dimensionality for MDS embedding
            random_state: Random seed for reproducibility

        Note: If n_chunks and chunk_size are not provided, they will be computed
        automatically during fit() based on the latent dimension.
        """
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.n_codewords = n_codewords
        self.use_var = use_var
        self.k = k
        self.mds_dim = mds_dim
        self.random_state = random_state
        self.codebook_chunks = None  # Final codebook of chunk centroids
        self.medoids_idx_global = None  # Indices of medoids in original data
        self.idx_sub = None  # Indices of subsampled points

    def chunk_latents(self, mu, logvar=None):
        """
        Split latent vectors into chunks for finer-grained quantization.

        Instead of quantizing the full D-dimensional latent vector,
        we split it into L chunks of size chunk_size, allowing
        independent quantization of each chunk.

        Args:
            mu: Latent means (N, D)
            logvar: Latent log-variances (N, D), optional

        Returns:
            mus_chunks: Reshaped means (N, n_chunks, chunk_size)
            logvar_chunks: Reshaped log-variances (N, n_chunks, chunk_size) or None
        """
        N, D = mu.shape
        assert D % self.n_chunks == 0, "LATENT_DIM must be divisible by n_chunks"
        mus_chunks = mu.reshape(N, self.n_chunks, self.chunk_size)
        if logvar is not None:
            logvar_chunks = logvar.reshape(N, self.n_chunks, self.chunk_size)
        else:
            logvar_chunks = None
        return mus_chunks, logvar_chunks

    def flatten_chunks(self, mus_chunks, logvar_chunks=None):
        """
        Flatten chunks into a single array of points for clustering.

        Transforms (N, n_chunks, chunk_size) -> (N*n_chunks, chunk_size)
        so we can treat each chunk independently.

        Args:
            mus_chunks: Chunked means (N, n_chunks, chunk_size)
            logvar_chunks: Chunked log-variances (N, n_chunks, chunk_size) or None

        Returns:
            points: Flattened chunks (N*n_chunks, chunk_size)
            logvars_pts: Flattened log-variances (N*n_chunks, chunk_size) or None
        """
        points = mus_chunks.reshape(-1, self.chunk_size)
        if logvar_chunks is not None:
            logvars_pts = logvar_chunks.reshape(-1, self.chunk_size)
        else:
            logvars_pts = None
        return points, logvars_pts

    def build_features(self, points, logvars_pts=None):
        """
        Build feature vectors for clustering.

        Optionally augments chunk vectors with variance information
        to account for uncertainty in the latent space.

        Args:
            points: Chunk vectors (N*n_chunks, chunk_size)
            logvars_pts: Log-variances (N*n_chunks, chunk_size) or None

        Returns:
            features: Feature vectors (N*n_chunks, chunk_size or 2*chunk_size)
        """
        if self.use_var and logvars_pts is not None:
            # Concatenate mean and std for richer representation
            features = np.concatenate([points, np.sqrt(np.exp(logvars_pts))], axis=1)
        else:
            features = points.copy()
        return features

    def fit(self, mu, logvar=None):
        """
        Fit the geodesic quantizer to training latents.

        Pipeline:
        1. Chunk latents into smaller vectors
        2. Subsample for computational efficiency
        3. Build k-NN graph connecting nearby chunks
        4. Compute geodesic distances (shortest paths on graph)
        5. Embed in lower-dim space via MDS to preserve geodesic structure
        6. Cluster with K-means in MDS space
        7. Select medoids (actual data points) as codebook entries

        Args:
            mu: Latent means (N, D)
            logvar: Latent log-variances (N, D), optional

        Returns:
            self: Fitted quantizer with codebook
        """
        # Auto-compute n_chunks and chunk_size if not provided
        latent_dim = mu.shape[1]
        if self.n_chunks is None or self.chunk_size is None:
            # Default: split latent into 8 chunks
            self.n_chunks = 8
            self.chunk_size = latent_dim // self.n_chunks
            print(
                f"Auto-computed: n_chunks={self.n_chunks}, chunk_size={self.chunk_size} (latent_dim={latent_dim})")

        # Step 1: Split into chunks
        mus_chunks, logvar_chunks = self.chunk_latents(mu, logvar)
        points, logvars_pts = self.flatten_chunks(mus_chunks, logvar_chunks)
        features = self.build_features(points, logvars_pts)

        # Step 2: Subsample for efficiency (max 5000 points)
        self.idx_sub = subsample_indices(features.shape[0], max_pts=5000)
        X_sub = features[self.idx_sub]

        # Step 3: Build k-NN graph (captures local manifold structure)
        A = build_knn_graph(X_sub, k=self.k)

        # Step 4: Compute geodesic distances via shortest paths
        # This respects the manifold geometry instead of using Euclidean distance
        D_geo = shortest_path(A, method='D', directed=False)

        # Step 5: MDS embedding to preserve geodesic distances in lower dimensions
        mds = MDS(n_components=self.mds_dim, dissimilarity='precomputed',
                  random_state=self.random_state, n_init=1, max_iter=300)
        X_mds = mds.fit_transform(D_geo)

        # Step 6: K-means clustering in MDS space
        kmeans = KMeans(n_clusters=self.n_codewords, random_state=self.random_state).fit(X_mds)
        labels_sub = kmeans.labels_

        # Step 7: Compute medoids (most representative actual points per cluster)
        # Medoids are better than centroids for preserving original data distribution
        points_sub = points[self.idx_sub]
        medoid_idxs_local, medoids_sub = compute_medoids(points_sub, labels_sub, n_clusters=self.n_codewords)
        self.medoids_idx_global = self.idx_sub[medoid_idxs_local]
        self.codebook_chunks = points[self.medoids_idx_global]

        return self

    def assign(self, mu):
        """
        Assign each latent vector to nearest codebook entries.

        For each chunk of each latent vector, finds the closest
        codebook entry using Euclidean distance.

        Args:
            mu: Latent means to quantize (N, D)

        Returns:
            codes_per_image: Codebook indices (N, n_chunks)
                Each row contains n_chunks indices into the codebook
        """
        N, D = mu.shape
        # Split into chunks
        mus_chunks, _ = self.chunk_latents(mu)
        points, _ = self.flatten_chunks(mus_chunks)

        # Find nearest codebook entry for each chunk
        assigned_point_idx, _ = pairwise_distances_argmin_min(points, self.codebook_chunks)

        # Reshape to (N, n_chunks) format
        codes_per_image = assigned_point_idx.reshape(N, self.n_chunks)
        return codes_per_image

    def save(self, path, codes_per_image=None, codes_grid=None):
        """
        Save codebook and assignments to disk.

        Args:
            path: Output file path (.npz)
            codes_per_image: Optional assigned codes (N, n_chunks)
            codes_grid: Optional reshaped codes for visualization (N, H, W)
        """
        # Save quantizer parameters along with codebook
        np.savez_compressed(path,
                            codebook_chunks=self.codebook_chunks,  # The learned codebook (K, chunk_size)
                            medoid_idxs_global=self.medoids_idx_global,  # Indices of medoids
                            subset_indices=self.idx_sub,  # Subsampled indices used for fitting
                            codes_per_image=codes_per_image,  # Assigned codes per image
                            codes_grid=codes_grid,  # Codes reshaped as spatial grid
                            # Save parameters for reconstruction
                            n_chunks=self.n_chunks,
                            chunk_size=self.chunk_size,
                            n_codewords=self.n_codewords,
                            use_var=self.use_var
                            )

    @classmethod
    def load(cls, path):
        """
        Load a saved quantizer from disk.

        Args:
            path: Path to saved .npz file

        Returns:
            GeodesicQuantizer: Loaded quantizer with fitted codebook
        """
        data = np.load(path)

        # Create quantizer with saved parameters
        quantizer = cls(
            n_codewords=int(data['n_codewords']),
            n_chunks=int(data['n_chunks']),
            chunk_size=int(data['chunk_size']),
            use_var=bool(data['use_var'])
        )

        # Load fitted codebook and metadata
        quantizer.codebook_chunks = data['codebook_chunks']
        quantizer.medoids_idx_global = data['medoid_idxs_global']
        quantizer.idx_sub = data['subset_indices']

        return quantizer
