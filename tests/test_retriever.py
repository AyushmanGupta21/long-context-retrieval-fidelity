"""
Unit tests for pure NumPy retriever, cosine similarity mathematical accuracy,
and deterministic Top-K ranking.
"""

import numpy as np
import pytest
from src.chunking import Chunk
from src.retriever import compute_cosine_similarity, retrieve_top_k


def test_cosine_similarity_identical_vectors():
    u = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    corpus = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    sim = compute_cosine_similarity(u, corpus)
    assert np.isclose(sim[0], 1.0, atol=1e-6)


def test_cosine_similarity_scaled_vectors():
    # Scale invariance: u and 100 * u should have similarity 1.0
    u = np.array([2.5, -1.5, 4.0], dtype=np.float32)
    corpus = np.array([[250.0, -150.0, 400.0]], dtype=np.float32)
    sim = compute_cosine_similarity(u, corpus)
    assert np.isclose(sim[0], 1.0, atol=1e-6)


def test_cosine_similarity_orthogonal_vectors():
    u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    corpus = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    sim = compute_cosine_similarity(u, corpus)
    assert np.isclose(sim[0], 0.0, atol=1e-6)
    assert np.isclose(sim[1], 0.0, atol=1e-6)


def test_cosine_similarity_opposite_vectors():
    u = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    corpus = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
    sim = compute_cosine_similarity(u, corpus)
    assert np.isclose(sim[0], -1.0, atol=1e-6)


def test_cosine_similarity_zero_vectors():
    zero_q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    corpus = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    # Zero query
    sim_zero_q = compute_cosine_similarity(zero_q, corpus)
    assert np.all(sim_zero_q == 0.0)
    assert not np.any(np.isnan(sim_zero_q))

    # Zero document in corpus
    non_zero_q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = compute_cosine_similarity(non_zero_q, corpus)
    assert np.isclose(sim[0], 1.0 / np.sqrt(14.0), atol=1e-5)
    assert sim[1] == 0.0
    assert not np.any(np.isnan(sim))


def test_cosine_similarity_dimension_mismatch():
    q = np.array([1.0, 2.0], dtype=np.float32)
    corpus = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="dimension mismatch"):
        compute_cosine_similarity(q, corpus)


def test_cosine_similarity_non_finite_values():
    q_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    corpus = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        compute_cosine_similarity(q_nan, corpus)

    q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    corpus_inf = np.array([[1.0, np.inf, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        compute_cosine_similarity(q, corpus_inf)


def test_cosine_similarity_empty_corpus():
    q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    empty_corpus = np.zeros((0, 3), dtype=np.float32)
    sim = compute_cosine_similarity(q, empty_corpus)
    assert sim.shape == (0,)


def test_retrieve_top_k_ranking_and_metadata():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    corpus = np.array([
        [0.0, 1.0, 0.0],  # orthogonal (score 0.0)
        [0.9, 0.1, 0.0],  # high similarity
        [0.5, 0.5, 0.0],  # medium similarity
        [-1.0, 0.0, 0.0], # opposite (score -1.0)
    ], dtype=np.float32)

    chunks = [
        Chunk("chk_0", "doc1", "Doc 1", "Orthogonal text", "fixed", 10, 0, 15, 0, 2, 0.1, "beginning"),
        Chunk("chk_1", "doc1", "Doc 1", "High sim text", "fixed", 10, 16, 30, 3, 5, 0.5, "middle"),
        Chunk("chk_2", "doc1", "Doc 1", "Medium sim text", "fixed", 10, 31, 45, 6, 8, 0.7, "intermediate"),
        Chunk("chk_3", "doc1", "Doc 1", "Opposite text", "fixed", 10, 46, 60, 9, 11, 0.95, "end"),
    ]

    results = retrieve_top_k(query_vector=q, corpus_matrix=corpus, chunks=chunks, k=2)

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].chunk_id == "chk_1"
    assert results[0].text == "High sim text"
    assert results[0].position_bucket == "middle"

    assert results[1].rank == 2
    assert results[1].chunk_id == "chk_2"
    assert results[1].text == "Medium sim text"


def test_retrieve_top_k_deterministic_ties():
    # Exactly identical vectors with tied similarity scores
    q = np.array([1.0, 1.0], dtype=np.float32)
    corpus = np.array([
        [1.0, 1.0],  # score 1.0
        [1.0, 1.0],  # score 1.0
        [1.0, 1.0],  # score 1.0
    ], dtype=np.float32)

    chunks = [
        Chunk("chk_C", "doc1", "T", "Text C", "fixed", 5, 0, 10, 0, 2, 0.5, "middle"),
        Chunk("chk_A", "doc1", "T", "Text A", "fixed", 5, 0, 10, 0, 2, 0.5, "middle"),
        Chunk("chk_B", "doc1", "T", "Text B", "fixed", 5, 0, 10, 0, 2, 0.5, "middle"),
    ]

    # Secondary sort key (chunk_id) ensures chk_A comes first, then chk_B, then chk_C
    results = retrieve_top_k(query_vector=q, corpus_matrix=corpus, chunks=chunks, k=3)
    assert [r.chunk_id for r in results] == ["chk_A", "chk_B", "chk_C"]


def test_retrieve_top_k_invalid_k_and_edge_cases():
    q = np.array([1.0, 0.0], dtype=np.float32)
    corpus = np.array([[1.0, 0.0]], dtype=np.float32)
    chunks = [Chunk("c1", "d", "t", "txt", "fixed", 5, 0, 5, 0, 1, 0.1, "beginning")]

    with pytest.raises(ValueError, match="k must be a positive integer"):
        retrieve_top_k(query_vector=q, corpus_matrix=corpus, chunks=chunks, k=0)

    # When K > N, return all N available results
    results = retrieve_top_k(query_vector=q, corpus_matrix=corpus, chunks=chunks, k=10)
    assert len(results) == 1

    # Empty collection
    empty_results = retrieve_top_k(query_vector=q, corpus_matrix=np.zeros((0, 2), dtype=np.float32), chunks=[], k=3)
    assert empty_results == []
