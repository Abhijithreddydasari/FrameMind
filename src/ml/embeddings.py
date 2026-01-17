"""Embedding cache and similarity operations.

Provides utilities for caching frame embeddings and performing
vector similarity operations for efficient retrieval.
"""
import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.logging import get_logger
from src.core.config import settings

logger = get_logger(__name__)

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


def compute_embedding_hash(embedding: NDArray[np.float32]) -> str:
    """Compute a hash for an embedding for cache key generation.
    
    Args:
        embedding: Embedding vector
        
    Returns:
        Hex digest of the embedding hash
    """
    # Use first/last elements + shape for fast hashing
    key_data = f"{embedding.shape}:{embedding[:4].tobytes()}:{embedding[-4:].tobytes()}"
    return hashlib.md5(key_data.encode()).hexdigest()[:16]


def cosine_similarity(
    query: NDArray[np.float32],
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute cosine similarity between query and embeddings.
    
    Args:
        query: Query embedding (1D array)
        embeddings: Matrix of embeddings (2D array, each row is an embedding)
        
    Returns:
        Array of similarity scores
    """
    # Normalize
    query_norm = query / (np.linalg.norm(query) + 1e-7)
    emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7)

    # Dot product for cosine similarity
    similarities = emb_norms @ query_norm

    return similarities.astype(np.float32)


def top_k_similar(
    query: NDArray[np.float32],
    embeddings: NDArray[np.float32],
    k: int,
    indices: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Find top-k most similar embeddings.
    
    Args:
        query: Query embedding
        embeddings: Matrix of embeddings
        k: Number of results to return
        indices: Optional list of indices corresponding to embeddings
        
    Returns:
        List of (index, similarity) tuples, sorted by similarity descending
    """
    similarities = cosine_similarity(query, embeddings)

    # Get top-k indices
    top_indices = np.argsort(similarities)[-k:][::-1]

    if indices is None:
        indices = list(range(len(embeddings)))

    results = [
        (indices[i], float(similarities[i]))
        for i in top_indices
    ]

    return results


def cluster_embeddings(
    embeddings: NDArray[np.float32],
    n_clusters: int,
) -> tuple[list[int], list[list[int]]]:
    """Cluster embeddings and return cluster assignments.
    
    Uses K-means clustering to group similar frames.
    
    Args:
        embeddings: Matrix of embeddings
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (centroid_indices, cluster_members)
    """
    from sklearn.cluster import KMeans

    n_samples = len(embeddings)
    n_clusters = min(n_clusters, n_samples)

    if n_clusters <= 1:
        return [0], [list(range(n_samples))]

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",
    )
    labels = kmeans.fit_predict(embeddings)

    # Find centroid (closest to cluster center) for each cluster
    centroid_indices: list[int] = []
    cluster_members: list[list[int]] = [[] for _ in range(n_clusters)]

    for i, label in enumerate(labels):
        cluster_members[label].append(i)

    for cluster_id in range(n_clusters):
        members = cluster_members[cluster_id]
        if not members:
            continue

        # Find member closest to centroid
        cluster_embeddings = embeddings[members]
        center = kmeans.cluster_centers_[cluster_id]

        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        centroid_idx = members[np.argmin(distances)]
        centroid_indices.append(centroid_idx)

    return centroid_indices, cluster_members


def embedding_to_json(embedding: NDArray[np.float32]) -> str:
    """Serialize embedding to JSON string for storage.
    
    Args:
        embedding: Embedding vector
        
    Returns:
        JSON string representation
    """
    return json.dumps(embedding.tolist())


def embedding_from_json(json_str: str) -> NDArray[np.float32]:
    """Deserialize embedding from JSON string.
    
    Args:
        json_str: JSON string representation
        
    Returns:
        Embedding vector
    """
    data = json.loads(json_str)
    return np.array(data, dtype=np.float32)


class EmbeddingIndex:
    """In-memory embedding index for fast similarity search.
    
    Provides efficient nearest neighbor search for small to medium
    collections. For larger collections, consider FAISS or similar.
    """

    def __init__(self) -> None:
        self._embeddings: list[NDArray[np.float32]] = []
        self._metadata: list[dict[str, Any]] = []
        self._matrix: NDArray[np.float32] | None = None

    def add(
        self,
        embedding: NDArray[np.float32],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add an embedding to the index.
        
        Args:
            embedding: Embedding vector
            metadata: Optional metadata to associate with embedding
            
        Returns:
            Index of the added embedding
        """
        idx = len(self._embeddings)
        self._embeddings.append(embedding)
        self._metadata.append(metadata or {})
        self._matrix = None  # Invalidate cache
        return idx

    def add_batch(
        self,
        embeddings: list[NDArray[np.float32]],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add multiple embeddings to the index.
        
        Args:
            embeddings: List of embedding vectors
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of indices for added embeddings
        """
        start_idx = len(self._embeddings)
        self._embeddings.extend(embeddings)

        if metadata_list:
            self._metadata.extend(metadata_list)
        else:
            self._metadata.extend([{}] * len(embeddings))

        self._matrix = None
        return list(range(start_idx, start_idx + len(embeddings)))

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """Search for similar embeddings.
        
        Args:
            query: Query embedding
            k: Number of results
            
        Returns:
            List of (index, similarity, metadata) tuples
        """
        if not self._embeddings:
            return []

        # Build matrix cache if needed
        if self._matrix is None:
            self._matrix = np.stack(self._embeddings)

        similarities = cosine_similarity(query, self._matrix)
        top_k = min(k, len(self._embeddings))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            (int(i), float(similarities[i]), self._metadata[i])
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._embeddings)

    def clear(self) -> None:
        """Clear the index."""
        self._embeddings.clear()
        self._metadata.clear()
        self._matrix = None


class FaissEmbeddingIndex:
    """FAISS-backed embedding index for fast similarity search."""

    def __init__(self, dim: int | None = None) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to use this index.")
        self.dim = dim
        self._index = None
        self._metadata: list[dict[str, Any]] = []
        self._ids: list[int] = []

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            # Use cosine similarity via inner product on normalized vectors
            self._index = faiss.IndexFlatIP(dim)

    def add(
        self,
        embedding: NDArray[np.float32],
        metadata: dict[str, Any] | None = None,
        idx: int | None = None,
    ) -> int:
        """Add an embedding to the FAISS index."""
        if embedding.ndim != 1:
            raise ValueError("Embedding must be a 1D vector")
        dim = embedding.shape[0]
        self._ensure_index(dim)
        vector = embedding / (np.linalg.norm(embedding) + 1e-7)
        self._index.add(vector.reshape(1, -1).astype(np.float32))

        assigned_idx = idx if idx is not None else len(self._ids)
        self._ids.append(assigned_idx)
        self._metadata.append(metadata or {})
        return assigned_idx

    def add_batch(
        self,
        embeddings: NDArray[np.float32],
        metadata_list: list[dict[str, Any]] | None = None,
        ids: list[int] | None = None,
    ) -> list[int]:
        """Add multiple embeddings to the FAISS index."""
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")
        dim = embeddings.shape[1]
        self._ensure_index(dim)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7
        vectors = embeddings / norms
        self._index.add(vectors.astype(np.float32))

        count = embeddings.shape[0]
        if ids is None:
            ids = list(range(len(self._ids), len(self._ids) + count))

        if metadata_list is None:
            metadata_list = [{} for _ in range(count)]

        self._ids.extend(ids)
        self._metadata.extend(metadata_list)
        return ids

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """Search for nearest embeddings."""
        if self._index is None or len(self._ids) == 0:
            return []

        query_norm = query / (np.linalg.norm(query) + 1e-7)
        scores, indices = self._index.search(query_norm.reshape(1, -1).astype(np.float32), k)
        results: list[tuple[int, float, dict[str, Any]]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            emb_id = self._ids[idx]
            metadata = self._metadata[idx]
            results.append((emb_id, float(score), metadata))
        return results

    def __len__(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        """Clear the index."""
        self._index = None
        self._metadata.clear()
        self._ids.clear()


class DualStreamIndex:
    """Dual-stream index for spatial + temporal embeddings.
    
    Manages separate FAISS indexes for spatial (CLIP) and temporal (X-CLIP)
    embeddings, with fused search capability.
    
    Example:
        index = DualStreamIndex()
        
        # Add embeddings
        index.add_spatial(frame_embedding, {"frame_index": 0})
        index.add_temporal(clip_embedding, {"clip_index": 0})
        
        # Search with fusion
        results = index.search_fused(query_embedding, k=10, alpha=0.5)
    """

    def __init__(
        self,
        spatial_dim: int = 512,
        temporal_dim: int = 512,
        use_gpu: bool = False,
    ) -> None:
        self.spatial_index = FaissEmbeddingIndex(spatial_dim)
        self.temporal_index = FaissEmbeddingIndex(temporal_dim)

    def add_spatial(
        self,
        embedding: NDArray[np.float32],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a spatial (frame) embedding."""
        return self.spatial_index.add(embedding, metadata)

    def add_spatial_batch(
        self,
        embeddings: list[NDArray[np.float32]],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add multiple spatial embeddings."""
        matrix = np.stack(embeddings).astype(np.float32)
        return self.spatial_index.add_batch(matrix, metadata_list)

    def add_temporal(
        self,
        embedding: NDArray[np.float32],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a temporal (clip) embedding."""
        return self.temporal_index.add(embedding, metadata)

    def add_temporal_batch(
        self,
        embeddings: list[NDArray[np.float32]],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add multiple temporal embeddings."""
        matrix = np.stack(embeddings).astype(np.float32)
        return self.temporal_index.add_batch(matrix, metadata_list)

    def search_spatial(
        self,
        query: NDArray[np.float32],
        k: int = 10,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """Search spatial index only."""
        return self.spatial_index.search(query, k)

    def search_temporal(
        self,
        query: NDArray[np.float32],
        k: int = 10,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """Search temporal index only."""
        return self.temporal_index.search(query, k)

    def search_fused(
        self,
        spatial_query: NDArray[np.float32],
        temporal_query: NDArray[np.float32] | None = None,
        k: int = 10,
        alpha: float = 0.5,
    ) -> list[tuple[str, int, float, dict[str, Any]]]:
        """Search both indexes and fuse results.
        
        Args:
            spatial_query: Query embedding for spatial search
            temporal_query: Query embedding for temporal search (optional)
            k: Number of results per stream
            alpha: Fusion weight (0=temporal only, 1=spatial only, 0.5=equal)
            
        Returns:
            List of (stream, index, fused_score, metadata) tuples
        """
        results: list[tuple[str, int, float, dict[str, Any]]] = []

        # Search spatial
        if len(self.spatial_index) > 0:
            spatial_results = self.spatial_index.search(spatial_query, k)
            for idx, score, meta in spatial_results:
                fused_score = score * alpha
                results.append(("spatial", idx, fused_score, meta))

        # Search temporal
        if temporal_query is not None and len(self.temporal_index) > 0:
            temporal_results = self.temporal_index.search(temporal_query, k)
            for idx, score, meta in temporal_results:
                fused_score = score * (1 - alpha)
                results.append(("temporal", idx, fused_score, meta))

        # Sort by fused score
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:k * 2]  # Return more results for diversity

    def __len__(self) -> int:
        return len(self.spatial_index) + len(self.temporal_index)

    @property
    def spatial_count(self) -> int:
        return len(self.spatial_index)

    @property
    def temporal_count(self) -> int:
        return len(self.temporal_index)

    def clear(self) -> None:
        """Clear both indexes."""
        self.spatial_index.clear()
        self.temporal_index.clear()
