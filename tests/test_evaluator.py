"""
Unit tests for evaluator module: relevance rule, metric calculations,
MRR, position breakdown, and handling queries with zero eligible evidence.
"""

import pytest
from src.chunking import Chunk
from src.evaluator import (
    BenchmarkQuery,
    compute_query_metrics,
    evaluate_retrieval,
    is_chunk_relevant,
    measure_semantic_completeness,
)
from src.retriever import RetrievalResult


def test_is_chunk_relevant():
    q = BenchmarkQuery(
        question_id="Q1",
        document_id="doc_a.txt",
        company="A Corp",
        filename="doc_a.txt",
        question="What is X?",
        answer="X is 42.",
        evidence="Exact evidence sentence for X is 42.",
        source_char_start=100,
        source_char_end=136,
        source_word_start=20,
        source_word_end=27,
        total_word_count=500,
        normalized_position=0.04,
        position_bucket="beginning",
    )

    # 1. Encompassing chunk in same doc -> relevant
    chunk_ok = Chunk(
        chunk_id="chk_1",
        document_id="doc_a.txt",
        title="A Corp",
        text="Prefix text. Exact evidence sentence for X is 42. Suffix text.",
        strategy="fixed",
        token_count=15,
        source_char_start=80,
        source_char_end=150,
        source_word_start=15,
        source_word_end=30,
        source_position_pct=0.04,
        position_bucket="beginning",
    )
    assert is_chunk_relevant(chunk_ok, q) is True

    # 2. Different document ID -> NOT relevant
    chunk_diff_doc = Chunk(
        chunk_id="chk_2",
        document_id="doc_b.txt",
        title="B Corp",
        text="Exact evidence sentence for X is 42.",
        strategy="fixed",
        token_count=10,
        source_char_start=80,
        source_char_end=150,
        source_word_start=15,
        source_word_end=30,
        source_position_pct=0.04,
        position_bucket="beginning",
    )
    assert is_chunk_relevant(chunk_diff_doc, q) is False

    # 3. Truncated chunk (span does not cover evidence and text does not contain evidence) -> NOT relevant
    chunk_truncated = Chunk(
        chunk_id="chk_3",
        document_id="doc_a.txt",
        title="A Corp",
        text="Prefix text. Exact evidence sentence",
        strategy="fixed",
        token_count=10,
        source_char_start=80,
        source_char_end=120,
        source_word_start=15,
        source_word_end=24,
        source_position_pct=0.04,
        position_bucket="beginning",
    )
    assert is_chunk_relevant(chunk_truncated, q) is False


def test_compute_query_metrics():
    # Relevant chunk IDs
    corpus_rel_ids = {"chk_rel_1", "chk_rel_2"}

    # Mock retrieval: chk_other at rank 1, chk_rel_1 at rank 2, chk_rel_2 at rank 3
    retrieved = [
        RetrievalResult(1, 0.9, "chk_other", "d", "T", "txt", "s", 10, 0, 10, 0, 1, 0.1, "beginning"),
        RetrievalResult(2, 0.8, "chk_rel_1", "d", "T", "txt", "s", 10, 11, 20, 2, 3, 0.2, "beginning"),
        RetrievalResult(3, 0.7, "chk_rel_2", "d", "T", "txt", "s", 10, 21, 30, 4, 5, 0.3, "beginning"),
        RetrievalResult(4, 0.6, "chk_other_2", "d", "T", "txt", "s", 10, 31, 40, 6, 7, 0.4, "middle"),
    ]

    metrics = compute_query_metrics(retrieved, corpus_rel_ids, k_values=[1, 3])

    # At K=1: rank 1 is chk_other (not relevant)
    assert metrics["k_1"]["hit_rate"] == 0.0
    assert metrics["k_1"]["precision"] == 0.0
    assert metrics["k_1"]["recall"] == 0.0
    assert metrics["k_1"]["mrr"] == 0.0

    # At K=3: rank 2 is chk_rel_1 (first relevant), rank 3 is chk_rel_2
    assert metrics["k_3"]["hit_rate"] == 1.0
    assert metrics["k_3"]["precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["k_3"]["recall"] == 1.0  # 2 of 2 in corpus retrieved
    assert metrics["k_3"]["mrr"] == 0.5     # 1 / rank 2 = 0.5


def test_query_with_no_eligible_evidence():
    # If a query has 0 relevant chunks in the corpus (split across seams)
    corpus_rel_ids = set()
    retrieved = [
        RetrievalResult(1, 0.9, "chk_1", "d", "T", "txt", "s", 10, 0, 10, 0, 1, 0.1, "beginning"),
    ]
    metrics = compute_query_metrics(retrieved, corpus_rel_ids, k_values=[1, 5])
    assert metrics["k_1"]["hit_rate"] == 0.0
    assert metrics["k_1"]["precision"] == 0.0
    assert metrics["k_1"]["recall"] == 0.0
    assert metrics["k_1"]["mrr"] == 0.0


def test_measure_semantic_completeness():
    chunks = [
        Chunk("c1", "d", "t", "This sentence ends with a period.", "s", 5, 0, 10, 0, 1, 0.1, "beginning"),
        Chunk("c2", "d", "t", "| Table Row | Value |", "s", 5, 0, 10, 0, 1, 0.1, "beginning"),
        Chunk("c3", "d", "t", "This chunk cuts off in the middle of a sent", "s", 5, 0, 10, 0, 1, 0.1, "beginning"),
    ]
    rate = measure_semantic_completeness(chunks)
    assert rate == pytest.approx(2.0 / 3.0, abs=1e-3)
