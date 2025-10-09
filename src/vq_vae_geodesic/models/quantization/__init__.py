"""Quantization methods for latent spaces."""
from .geodesic import GeodesicQuantizer
from .utils import subsample_indices, build_knn_graph, compute_medoids

__all__ = ["GeodesicQuantizer", "subsample_indices", "build_knn_graph", "compute_medoids"]
