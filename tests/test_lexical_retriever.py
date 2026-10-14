"""
Unit tests for src/lexical_retriever.py:
Tests Okapi BM25 indexing, term matching, TF saturation, document length
normalization, deterministic tie-breaking, and edge case handling.
"""

import math
import pytest
from src.chunking import Chunk
from src.lexical_retriever import BM25Retriever, tokenize_text


def _create_mock_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc_test.txt",
        title="Test Company",
        text=text,
        strategy="sentence",
        token_count=len(text.split()),
        source_char_start=0,
        source_char_end=len(text),
        source_word_start=0,
        source_word_end=len(text.split()),
        source_position_pct=0.5,
        position_bucket="middle",
    )


def test_tokenize_text():
    assert tokenize_text("") == []
    assert tokenize_text("   \n\t  ") == []
    assert tokenize_text("Alphabet Inc. reported $307.4 billion in 2025.") == [
        "alphabet",
        "inc",
        "reported",
        "307",
        "4",
        "billion",
        "in",
        "2025",
    ]


def test_bm25_parameter_validation():
    chunks = [_create_mock_chunk("c1", "Some text.")]
    with pytest.raises(ValueError, match="k1"):
        BM25Retriever(chunks, k1=-0.5)

    with pytest.raises(ValueError, match="b"):
        BM25Retriever(chunks, b=1.5)

    with pytest.raises(ValueError, match="b"):
        BM25Retriever(chunks, b=-0.1)


def test_bm25_term_matching_and_ranking():
    c1 = _create_mock_chunk("c1", "Google Cloud AI infrastructure and TPU accelerators.")
    c2 = _create_mock_chunk("c2", "Amazon Web Services cloud computing and EC2 instances.")
    c3 = _create_mock_chunk("c3", "Netflix streaming media subscriber count growth.")

    retriever = BM25Retriever([c1, c2, c3], k1=1.5, b=0.75)

    # Query for "TPU accelerators"
    results = retriever.retrieve("TPU accelerators", k=3)
    assert len(results) == 3
    assert results[0].chunk_id == "c1"
    assert results[0].score > results[1].score
    assert results[0].rank == 1

    # Query for "Netflix subscriber"
    results_netflix = retriever.retrieve("Netflix subscriber", k=3)
    assert results_netflix[0].chunk_id == "c3"
    assert results_netflix[0].rank == 1


def test_bm25_tf_saturation():
    # c1 has 'revenue' once, c2 has 'revenue' 5 times in equal length docs
    c1 = _create_mock_chunk("c1", "Annual revenue reached expectations for operating segments.")
    c2 = _create_mock_chunk("c2", "Revenue revenue revenue revenue revenue for segments.")

    retriever = BM25Retriever([c1, c2], k1=1.5, b=0.75)
    scores = retriever.compute_scores("revenue")

    # c2 must have higher score due to TF, but not 5x higher (due to saturation)
    assert scores[1] > scores[0]
    assert scores[1] < scores[0] * 5.0


def test_bm25_document_length_normalization():
    # Both have word 'antitrust' once, but c1 is short (10 words) and c2 is long (50 words)
    c1 = _create_mock_chunk("c1", "Department of Justice filed an antitrust complaint against search.")
    c2 = _create_mock_chunk(
        "c2",
        "The company operates across multiple international business jurisdictions. "
        "Regulatory authorities in various countries examine corporate operations, "
        "including an antitrust complaint filed in court, while global compliance costs "
        "continue to increase over the course of the fiscal reporting year."
    )

    retriever = BM25Retriever([c1, c2], k1=1.5, b=0.75)
    scores = retriever.compute_scores("antitrust")

    # Length normalization b=0.75 penalizes the much longer document c2
    assert scores[0] > scores[1]


def test_bm25_deterministic_tie_breaking():
    # Two chunks with identical text and scores
    c_b = _create_mock_chunk("chunk_B", "General legal proceedings disclosure.")
    c_a = _create_mock_chunk("chunk_A", "General legal proceedings disclosure.")

    retriever = BM25Retriever([c_b, c_a])
    results = retriever.retrieve("legal proceedings", k=2)

    # Identical score, so chunk_A must precede chunk_B alphabetically
    assert results[0].score == pytest.approx(results[1].score)
    assert results[0].chunk_id == "chunk_A"
    assert results[1].chunk_id == "chunk_B"
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_bm25_empty_query_and_empty_corpus():
    # Empty corpus
    empty_retriever = BM25Retriever([])
    assert empty_retriever.retrieve("test", k=5) == []

    # Empty query
    c = _create_mock_chunk("c1", "Some content")
    retriever = BM25Retriever([c])
    results = retriever.retrieve("", k=1)
    assert len(results) == 1
    assert results[0].score == 0.0

    # Invalid k
    with pytest.raises(ValueError, match="k"):
        retriever.retrieve("content", k=0)
