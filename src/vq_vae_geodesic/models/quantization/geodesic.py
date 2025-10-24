"""
Geodesic Quantizer for VAE latent spaces.

Performs a posteriori vector quantization using geodesic distances 
instead of Euclidean distances, respecting the manifold structure 
of the latent space.
"""
import torch
from scipy.sparse.csgraph import shortest_path
from kmedoids import KMedoids

from .utils import build_knn_graph_w2, subsample_indices, w2_squared_distance


class GeodesicQuantizer:
    """
    Quantizes VAE latent vectors using geodesic distances.

    Pipeline:
    1. Split latent vectors into chunks
    2. Build k-NN graph on subsampled chunks
    3. Compute geodesic distances via shortest paths (requires numpy)
    4. Cluster with K-medoids (requires numpy)
    5. Use medoids as final codewords (stored as torch.Tensor)
    """

    def __init__(self, n_codewords=256, n_chunks=None, chunk_size=None, k=20, random_state=42):
        """
        Args:
            n_codewords: Size of the codebook (K centroids)
            n_chunks: Number of chunks to split each latent vector into (auto-computed if None)
            chunk_size: Dimensionality of each chunk (auto-computed if None), ITS ALSO THE CODEWORD SIZE
            k: Number of neighbors for k-NN graph
            random_state: Random seed for reproducibility
        """
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.n_codewords = n_codewords
        self.k = k
        self.random_state = random_state
        self.codebook_chunks = None  # Final codebook of chunk centroids
        self.codebook_sigmas = None  # Final codebook of chunk sigmas (stddev)
        self.medoids_idx_global = None  # Indices of medoids in original data
        self.idx_sub = None  # Indices of subsampled points


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
        mus_chunks = mu.view(N, self.n_chunks, self.chunk_size)
        logvar_chunks = logvar.view(N, self.n_chunks, self.chunk_size)
        return mus_chunks, logvar_chunks

    def flatten_chunks(self, mus_chunks, logvar_chunks):
        """
        Flatten chunks into a single array of points for clustering.

        Transforms (N, n_chunks, chunk_size) -> (N*n_chunks, chunk_size)
        """
        points = mus_chunks.view(-1, self.chunk_size)
        logvars_pts = logvar_chunks.view(-1, self.chunk_size)
        return points, logvars_pts

    def fit(self, mu, logvar):
        """
        Fit the geodesic quantizer to training latents.

        Args:
            mu: Latent means (N, D)
            logvar: Latent log-variances (N, D)

        Returns:
            self: Fitted quantizer with codebook
        """
        # Auto-compute chunk_size
        latent_dim = mu.shape[1]
        self.chunk_size = latent_dim // self.n_chunks
        print(f"Auto-computed: n_chunks={self.n_chunks}, chunk_size={self.chunk_size} (latent_dim={latent_dim})")

        # Step 1: Split into chunks
        mus_chunks, logvar_chunks = self.chunk_latents(mu, logvar)
        points, logvars_pts = self.flatten_chunks(mus_chunks, logvar_chunks)

        # Convert logvars to stddevs (sigmas)
        sigmas = torch.sqrt(torch.exp(logvars_pts))

        # Step 2: Subsample for efficiency (max 10000 points)
        self.idx_sub = subsample_indices(points.shape[0], max_pts=10000, seed=self.random_state)
        points_sub = points[self.idx_sub]
        sigmas_sub = sigmas[self.idx_sub]

        # Step 3: Build k-NN graph (captures local manifold structure)
        print("Building k-NN graph with W-2 distances (no normalization)...")
        A = build_knn_graph_w2(points_sub, sigmas_sub, k=self.k)

        # Step 4: Compute geodesic distances via shortest paths
        print("Computing shortest paths for geodesic distances...")
        D_geo = shortest_path(A, method='D', directed=False)  # numpy array

        # Step 5: K-medoids clustering directly on geodesic distance matrix
        print("Clustering with K-medoids on geodesic distances (kmedoids package)...")
        km = KMedoids(n_clusters=self.n_codewords, metric='precomputed', init='random', random_state=self.random_state)
        km.fit(D_geo)
        medoid_idxs_local = km.medoid_indices_
        # Medoids in original space: map local idx -> global index; then pick from original points
        self.medoids_idx_global = torch.as_tensor(self.idx_sub[medoid_idxs_local], dtype=torch.long)
        self.codebook_chunks = points[self.medoids_idx_global].clone()  # torch.Tensor
        self.codebook_sigmas = sigmas[self.medoids_idx_global].clone()  # torch.Tensor

        if self.codebook_chunks is None or self.codebook_chunks.shape[0] != self.n_codewords:
            raise RuntimeError("Codebook construction failed or produced wrong shape.")
        return self

    def assign(self, mu, logvar, batch_size=10000):
        """
        Assign each latent vector to nearest codebook entries using direct Wasserstein-2 (squared) distance.

        Args:
            mu: Latent means to quantize (N, D)
            logvar: Latent log-variances (N, D)
            batch_size: Number of chunks to process at once (to limit RAM usage)
            device: torch.device su cui eseguire l'assegnazione (CPU o GPU). Se None, usa device di mu.

        Returns:
            codes_per_image: Codebook indices (N, n_chunks)
                Each row contains n_chunks indices into the codebook
        """
        N, D = mu.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mus_chunks, logvar_chunks = self.chunk_latents(mu, logvar) # (N, n_chunks, chunk_size)
        points, logvars_pts = self.flatten_chunks(mus_chunks, logvar_chunks) # (N*n_chunks, chunk_size)
        sigmas = torch.sqrt(torch.exp(logvars_pts))

        # Move to device
        points = points.to(device)
        sigmas = sigmas.to(device)
        medoids = self.codebook_chunks.to(device)
        medoids_sigmas = self.codebook_sigmas.to(device)

        assigned_idx = []
        for start in range(0, points.shape[0], batch_size):
            end = min(start + batch_size, points.shape[0])
            # For each chunk in batch, compute distance to all medoids (codewords)
            # (batch of points (chunks), n_codewords)
            dists = w2_squared_distance(
                points[start:end].unsqueeze(1), sigmas[start:end].unsqueeze(1),
                medoids.unsqueeze(0), medoids_sigmas.unsqueeze(0)
            )
            # Find and assign the closest medoid (codeword) index for each chunk
            assigned_idx.append(torch.argmin(dists, dim=1))
        # Concatenate back all assigned indices over batches
        assigned_idx = torch.cat(assigned_idx, dim=0)
        # Reshape to (N, n_chunks) (order is preserved)
        # Each latent vector is represented by n_chunks codeword indices
        codes_per_image = assigned_idx.view(N, self.n_chunks) # (N, n_chunks) codeword indices
        return codes_per_image

    def save(self, path):
        """
        Save codebook to disk.
        """
        # Save quantizer parameters along with codebook
        save_dict = {
            "codebook_chunks": self.codebook_chunks,
            "codebook_sigmas": self.codebook_sigmas,
            "medoid_idxs_global": self.medoids_idx_global,
            "subset_indices": self.idx_sub,
            "n_chunks": self.n_chunks,
            "chunk_size": self.chunk_size,
            "n_codewords": self.n_codewords,
        }
        torch.save(save_dict, path)
