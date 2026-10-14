"""
Retrieval core and mathematical vector ranking module.
Implements pure NumPy cosine similarity and deterministic Top-K ranking
with zero external vector databases or retrieval frameworks.

=============================================================================
MATHEMATICAL FOUNDATIONS & COMPARATIVE ANALYSIS:
Cosine Similarity vs. Dot Product vs. Euclidean Distance
=============================================================================

1. Cosine Similarity:
   cos(u, v) = (u . v) / (||u||_2 * ||v||_2)
   - Range: [-1.0, 1.0]
   - Geometric Meaning: Measures the cosine of the angle between two vectors
     in d-dimensional space.
   - Invariance: Completely scale-invariant and length-invariant. Multiplying
     vector u by a constant scalar c > 0 yields identical similarity.
   - Information Retrieval Role: Preferred for text embeddings because text
     length or verbosity variations do not distort relevance alignment.
   - Limitation: Disregards vector magnitude entirely. When vector norm
     conveys signal (e.g., certainty or term significance), that information
     is lost.

2. Dot Product (Inner Product):
   <u, v> = u . v = ||u||_2 * ||v||_2 * cos(u, v) = sum(u_i * v_i)
   - Range: (-inf, +inf)
   - Geometric Meaning: Measures both the angular alignment and the magnitudes
     of the vectors.
   - Relationship to Cosine: If vectors are unit-normalized (||u||_2 = ||v||_2 = 1),
     the dot product is mathematically IDENTICAL to cosine similarity.
   - Limitation: For unnormalized vectors, documents with larger embedding norms
     (e.g., longer passages or repetitive tokens) receive artificially inflated scores,
     biasing ranking toward longer texts regardless of semantic relevance.

3. Euclidean Distance (L2 Distance):
   d(u, v) = ||u - v||_2 = sqrt( sum( (u_i - v_i)^2 ) )
   - Range: [0.0, +inf)
   - Geometric Meaning: Straight-line physical distance between vector coordinates.
   - Relationship to Cosine: For unit-normalized vectors:
       ||u - v||_2^2 = ||u||^2 + ||v||^2 - 2(u . v) = 1 + 1 - 2 cos(u, v) = 2 - 2 cos(u, v)
       d(u, v) = sqrt( 2 * (1 - cos(u, v)) )
     Thus, for normalized embeddings, Euclidean distance is monotonically inverse
     to cosine similarity (minimizing L2 distance is equivalent to maximizing cosine similarity).
   - Limitation: For unnormalized vectors, Euclidean distance penalizes magnitude differences
     even when vectors point in the identical direction (same semantic topic).
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from src.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    title: str
    text: str
    strategy: str
    token_count: int
    source_char_start: int
    source_char_end: int
    source_word_start: int
    source_word_end: int
    source_position_pct: float
    position_bucket: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_cosine_similarity(
    query_vector: np.ndarray,
    corpus_matrix: np.ndarray,
) -> np.ndarray:
    """
    Computes exact cosine similarity between a query vector and a matrix of corpus vectors
    using vectorized pure NumPy operations.

    Formula:
        sim(q, D_i) = (q . D_i) / (||q||_2 * ||D_i||_2)

    Edge Cases Handled:
    - Zero vectors: Returns 0.0 similarity without NaN or division by zero.
    - Dimension mismatch: Raises ValueError with informative message.
    - Non-finite values: Raises ValueError if inputs contain NaN or Inf.
    - Empty corpus matrix: Returns empty 1D float32 array.
    - Output bounds: Clamped to [-1.0, 1.0] to prevent floating point overflow errors.

    Args:
        query_vector: 1D array of shape (d,) or 2D array of shape (1, d).
        corpus_matrix: 2D array of shape (N, d).

    Returns:
        1D float32 array of shape (N,) containing cosine similarity scores.
    """
    # 1. Input validation
    q = np.asarray(query_vector, dtype=np.float32)
    D = np.asarray(corpus_matrix, dtype=np.float32)

    if q.ndim == 2 and q.shape[0] == 1:
        q = q.squeeze(axis=0)

    if q.ndim != 1:
        raise ValueError(f"query_vector must be 1-dimensional, got shape {query_vector.shape}.")

    if D.ndim != 2:
        raise ValueError(f"corpus_matrix must be 2-dimensional (N, d), got shape {corpus_matrix.shape}.")

    if D.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    if q.shape[0] != D.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: query vector has dimension {q.shape[0]}, "
            f"but corpus matrix has dimension {D.shape[1]}."
        )

    if not np.all(np.isfinite(q)):
        raise ValueError("query_vector contains non-finite values (NaN or Inf).")

    if not np.all(np.isfinite(D)):
        raise ValueError("corpus_matrix contains non-finite values (NaN or Inf).")

    # 2. Vectorized norm and dot product computation
    q_norm = float(np.linalg.norm(q))
    D_norms = np.linalg.norm(D, axis=1)

    # 3. Handle zero-magnitude edge cases
    if q_norm == 0.0:
        return np.zeros(D.shape[0], dtype=np.float32)

    denominators = q_norm * D_norms
    dot_products = np.dot(D, q)

    # Avoid division by zero where D_norms == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        similarities = np.where(denominators > 0.0, dot_products / denominators, 0.0)

    # 4. Clamp to strictly legal [-1.0, 1.0] mathematical range
    similarities = np.clip(similarities, -1.0, 1.0).astype(np.float32)

    return similarities


def retrieve_top_k(
    query_vector: np.ndarray,
    corpus_matrix: np.ndarray,
    chunks: list[Chunk],
    k: int = 5,
) -> list[RetrievalResult]:
    """
    Performs deterministic Top-K retrieval ranking over chunk embeddings using NumPy cosine similarity.

    Deterministic Tie-Breaking:
    Primary Sort: Cosine similarity score descending.
    Secondary Sort: Chunk identifier ascending (ensures strict determinism across runs).

    Args:
        query_vector: 1D array of shape (d,) or 2D (1, d).
        corpus_matrix: 2D array of shape (N, d).
        chunks: List of Chunk metadata objects corresponding to rows in corpus_matrix.
        k: Number of top results to return (must be >= 1).

    Returns:
        List of RetrievalResult objects ordered by rank (1 to k).
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer >= 1, got {k}.")

    N = corpus_matrix.shape[0]
    if len(chunks) != N:
        raise ValueError(f"Chunk count ({len(chunks)}) does not match corpus matrix rows ({N}).")

    if N == 0:
        return []

    # Compute cosine similarities via pure NumPy
    scores = compute_cosine_similarity(query_vector, corpus_matrix)

    # Effective K cannot exceed total available documents
    effective_k = min(k, N)

    # Deterministic sorting: sort by (-score, chunk_id, original_index)
    sorted_indices = sorted(
        range(N),
        key=lambda i: (-float(scores[i]), chunks[i].chunk_id, i),
    )[:effective_k]

    results: list[RetrievalResult] = []
    for rank, idx in enumerate(sorted_indices, 1):
        c = chunks[idx]
        results.append(
            RetrievalResult(
                rank=rank,
                score=round(float(scores[idx]), 6),
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                title=c.title,
                text=c.text,
                strategy=c.strategy,
                token_count=c.token_count,
                source_char_start=c.source_char_start,
                source_char_end=c.source_char_end,
                source_word_start=c.source_word_start,
                source_word_end=c.source_word_end,
                source_position_pct=c.source_position_pct,
                position_bucket=c.position_bucket,
            )
        )

    return results
