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
# from sklearn_extra.cluster import KMedoids
from kmedoids import KMedoids
from sklearn.metrics import pairwise_distances_argmin_min

from .utils import build_knn_graph_w2, subsample_indices, build_knn_graph, compute_medoids


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

    def __init__(self, n_codewords=256, n_chunks=None, chunk_size=None, k=20, mds_dim=64, random_state=42):
        """
        Args:
            n_codewords: Size of the codebook (K centroids)
            n_chunks: Number of chunks to split each latent vector into (auto-computed if None)
            chunk_size: Dimensionality of each chunk (auto-computed if None)
            k: Number of neighbors for k-NN graph
            mds_dim: Dimensionality for MDS embedding
            random_state: Random seed for reproducibility

        Note: If n_chunks and chunk_size are not provided, they will be computed
        automatically during fit() based on the latent dimension.
        """
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.n_codewords = n_codewords
        self.k = k
        self.mds_dim = mds_dim
        self.random_state = random_state
        self.codebook_chunks = None  # Final codebook of chunk centroids
        self.codebook_sigmas = None  # Final codebook of chunk sigmas (stddev)
        self.medoids_idx_global = None  # Indices of medoids in original data
        self.idx_sub = None  # Indices of subsampled points

        self.mu_std = None
        self.mu_mean = None
        self.sigma_std = None
        self.sigma_mean = None
        self.D_geo = None  # Geodesic distance matrix on subsampled points

    def chunk_latents(self, mu, logvar):
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
        logvar_chunks = logvar.reshape(N, self.n_chunks, self.chunk_size)
        return mus_chunks, logvar_chunks

    def flatten_chunks(self, mus_chunks, logvar_chunks):
        """
        Flatten chunks into a single array of points for clustering.

        Transforms (N, n_chunks, chunk_size) -> (N*n_chunks, chunk_size)
        so we can treat each chunk independently.
        """
        points = mus_chunks.reshape(-1, self.chunk_size)
        logvars_pts = logvar_chunks.reshape(-1, self.chunk_size)
        return points, logvars_pts

    def build_features(self, points, sigmas):
        """
        Build feature vectors for clustering.
        """
        # Concatenate mean and std for richer representation
        features = np.concatenate([points, sigmas], axis=1)

        return features
    
    
    def _fit_normalization_params(self, points, sigmas):
        """
        Calcola e salva i parametri di normalizzazione per points e sigmas.
        """
        self.mu_mean = points.mean(axis=0, keepdims=True)
        self.mu_std = points.std(axis=0, keepdims=True) + 1e-9
        self.sigma_mean = sigmas.mean(axis=0, keepdims=True)
        self.sigma_std = sigmas.std(axis=0, keepdims=True) + 1e-9
    
    def _normalize(self, points, sigmas):
        """
        Normalizza points e sigmas usando i parametri salvati.
        """
        points_norm = (points - self.mu_mean) / self.mu_std
        sigmas_norm = (sigmas - self.sigma_mean) / self.sigma_std
        return points_norm, sigmas_norm

    def fit(self, mu, logvar):
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

        # Convert logvars to stddevs (sigmas)
        sigmas = np.sqrt(np.exp(logvars_pts))

        # features = self.build_features(points, sigmas)

        # Stima e salva i parametri di normalizzazione
        self._fit_normalization_params(points, sigmas)
        points_normalized, sigmas_normalized = self._normalize(points, sigmas)

        # Step 2: Subsample for efficiency (max 5000 points)
        self.idx_sub = subsample_indices(points_normalized.shape[0], max_pts=5000)
        points_sub = points_normalized[self.idx_sub]
        sigmas_sub = sigmas_normalized[self.idx_sub]

        # Step 3: Build k-NN graph (captures local manifold structure)
        # A = build_knn_graph(points_sub, k=self.k)
        print("Building k-NN graph with W-2 distances...")
        A = build_knn_graph_w2(points_sub, sigmas_sub, k=self.k)

        # Step 4: Compute geodesic distances via shortest paths
        # This respects the manifold geometry instead of using Euclidean distance
        print("Computing shortest paths for geodesic distances...")
        D_geo = shortest_path(A, method='D', directed=False)  # Outputs a matrix with shortest path distances
        self.D_geo = D_geo  # Salva la matrice delle distanze geodetiche


        # Step 5: K-medoids clustering directly on geodesic distance matrix
        print("Clustering with K-medoids on geodesic distances (kmedoids package)...")
        # kmedoids expects a distance matrix and returns medoid indices
        km = KMedoids(n_clusters=self.n_codewords, metric='precomputed', init='random', random_state=self.random_state)
        labels_sub = km.fit_predict(D_geo)
        medoid_idxs_local = km.medoid_indices_
        # Medoids in original space: map local idx -> global index; then pick from original points (not normalized)
        self.medoids_idx_global = self.idx_sub[medoid_idxs_local]
        self.codebook_chunks = points[self.medoids_idx_global]  # original scale
        self.codebook_sigmas = sigmas[self.medoids_idx_global]  # salva anche le sigmas dei medoids


        # # Step 5: Multi-dimensional scaling to preserve geodesic distances in lower dimensions
        # print("Performing MDS embedding...")
        # mds = MDS(n_components=self.mds_dim, dissimilarity='precomputed',
        #           random_state=self.random_state, n_init=4, max_iter=300)
        # X_mds = mds.fit_transform(D_geo)

        # # Step 6: K-means clustering in MDS space
        # print("Clustering with K-means...")
        # kmeans = KMeans(n_clusters=self.n_codewords, random_state=self.random_state).fit(X_mds)
        # labels_sub = kmeans.labels_

        # # Step 7: Compute medoids (most representative actual points per cluster)
        # # Medoids are better than centroids for preserving original data distribution
        # points_sub = points[self.idx_sub]
        # print("Computing medoids...")
        # medoid_idxs_local, medoids_sub = compute_medoids(points_sub, labels_sub, n_clusters=self.n_codewords)
        # self.medoids_idx_global = self.idx_sub[medoid_idxs_local]
        # self.codebook_chunks = points[self.medoids_idx_global]

        if self.codebook_chunks is None or self.codebook_chunks.shape[0] != self.n_codewords:
            raise RuntimeError("Codebook construction failed or produced wrong shape.")

        return self

    def assign(self, mu, logvar):
        """
        Assign each latent vector to nearest codebook entries.

        For training data (same as fit): uses precomputed D_geo.
        For new data (e.g., validation): builds a new k-NN graph including new chunks and medoids,
        computes geodesic distances, and assigns each chunk to the nearest medoid.

        Args:
            mu: Latent means to quantize (N, D)
            logvar: Latent log-variances (N, D)

        Returns:
            codes_per_image: Codebook indices (N, n_chunks)
                Each row contains n_chunks indices into the codebook
        """
        N, D = mu.shape
        mus_chunks, logvar_chunks = self.chunk_latents(mu, logvar)
        points, logvars_pts = self.flatten_chunks(mus_chunks, logvar_chunks)
        sigmas = np.sqrt(np.exp(logvars_pts))

        # Normalizza points e sigmas con i parametri salvati
        points_norm, sigmas_norm = self._normalize(points, sigmas)

        # Generalized: always build k-NN graph on [input chunks + medoids], normalize both
        medoids = self.codebook_chunks
        medoids_sigmas = self.codebook_sigmas if self.codebook_sigmas is not None else np.zeros_like(medoids)
        medoids_norm, medoids_sigmas_norm = self._normalize(medoids, medoids_sigmas)

        all_points = np.concatenate([points_norm, medoids_norm], axis=0)
        all_sigmas = np.concatenate([sigmas_norm, medoids_sigmas_norm], axis=0)

        from .utils import build_knn_graph_w2
        k_graph = self.k
        print("Building k-NN graph for assignment (chunks+medoids)...")
        A = build_knn_graph_w2(all_points, all_sigmas, k=k_graph)

        print("Computing geodesic distances for assignment...")
        D_geo = shortest_path(A, method='D', directed=False)

        n_chunks = points.shape[0]
        n_medoids = medoids.shape[0]
        chunk_idx = np.arange(n_chunks)
        medoid_idx = np.arange(n_chunks, n_chunks + n_medoids)

        D_to_medoids = D_geo[chunk_idx[:, None], medoid_idx[None, :]]  # shape (n_chunks, n_medoids)
        assigned_idx = np.argmin(D_to_medoids, axis=1)
        codes_per_image = assigned_idx.reshape(N, self.n_chunks)
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
                            codebook_sigmas=self.codebook_sigmas,  # The learned codebook sigmas (K, chunk_size)
                            medoid_idxs_global=self.medoids_idx_global,  # Indices of medoids
                            subset_indices=self.idx_sub,  # Subsampled indices used for fitting
                            codes_per_image=codes_per_image,  # Assigned codes per image
                            codes_grid=codes_grid,  # Codes reshaped as spatial grid
                            # Save parameters for reconstruction
                            n_chunks=self.n_chunks,
                            chunk_size=self.chunk_size,
                            n_codewords=self.n_codewords,
                            mu_mean=self.mu_mean,
                            mu_std=self.mu_std,
                            sigma_mean=self.sigma_mean,
                            sigma_std=self.sigma_std,
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
        )

        # Load fitted codebook and metadata
        quantizer.codebook_chunks = data['codebook_chunks']
        quantizer.codebook_sigmas = data['codebook_sigmas'] if 'codebook_sigmas' in data else None
        quantizer.medoids_idx_global = data['medoid_idxs_global']
        quantizer.idx_sub = data['subset_indices']
        quantizer.mu_mean = data['mu_mean']
        quantizer.mu_std = data['mu_std']
        quantizer.sigma_mean = data['sigma_mean']
        quantizer.sigma_std = data['sigma_std']

        return quantizer
