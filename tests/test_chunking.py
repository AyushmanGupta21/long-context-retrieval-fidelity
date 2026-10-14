"""
Unit tests for chunking module: fixed-token overlapping chunker,
sentence-aware chunker, token limits, and source position metadata.
"""

import pytest
from src.chunking import (
    MAX_CONTENT_TOKENS,
    classify_position_bucket,
    fixed_token_chunking,
    sentence_aware_chunking,
)


def test_classify_position_bucket():
    assert classify_position_bucket(0.00) == "beginning"
    assert classify_position_bucket(0.05) == "beginning"
    assert classify_position_bucket(0.10) == "beginning"

    assert classify_position_bucket(0.25) == "intermediate"

    assert classify_position_bucket(0.40) == "middle"
    assert classify_position_bucket(0.50) == "middle"
    assert classify_position_bucket(0.60) == "middle"

    assert classify_position_bucket(0.75) == "intermediate"

    assert classify_position_bucket(0.90) == "end"
    assert classify_position_bucket(0.95) == "end"
    assert classify_position_bucket(1.00) == "end"


def test_fixed_token_chunking_basic_and_offsets():
    sample_text = (
        "Alphabet Inc. reported significant revenue growth in cloud computing. "
        "The segment saw strong enterprise customer adoption worldwide. "
        "Operating margins also improved substantially across major divisions."
    )
    chunks = fixed_token_chunking(
        text=sample_text,
        document_id="googl_test",
        title="Alphabet Test",
        chunk_size=15,
        overlap=5,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_id == "googl_test"
        assert chunk.title == "Alphabet Test"
        assert chunk.strategy == "fixed_token_overlap"
        assert chunk.token_count <= 15
        assert chunk.source_char_start >= 0
        assert chunk.source_char_end <= len(sample_text)
        # Verify slice in original document matches chunk text
        assert sample_text[chunk.source_char_start:chunk.source_char_end] == chunk.text
        assert chunk.source_word_start >= 0
        assert chunk.source_word_end >= chunk.source_word_start
        assert 0.0 <= chunk.source_position_pct <= 1.0


def test_fixed_token_chunking_overlap():
    sample_text = (
        "First sentence discussing technology and financial performance. "
        "Second sentence covering operating expenses and capital expenditures. "
        "Third sentence describing future growth strategy and guidance."
    )
    chunks = fixed_token_chunking(
        text=sample_text,
        document_id="overlap_test",
        chunk_size=12,
        overlap=4,
    )

    assert len(chunks) >= 2
    # Verify adjacent chunks share overlap in characters
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]
        assert c2.source_char_start < c1.source_char_end, "Adjacent chunks should overlap in character span"


def test_sentence_aware_chunking_boundaries_and_tables():
    sample_text = (
        "ITEM 1. BUSINESS.\n"
        "Alphabet is a global technology company providing digital products.\n\n"
        "| Segment | FY 2025 | FY 2024 |\n"
        "| Google Services | $304,000 | $270,000 |\n\n"
        "Our mission is to organize the world's information and make it accessible.\n"
        "We continue to innovate in artificial intelligence and infrastructure."
    )
    chunks = sentence_aware_chunking(
        text=sample_text,
        document_id="sent_test",
        title="Sentence Aware Test",
        target_tokens=40,
    )

    assert len(chunks) > 0
    all_chunk_texts = " ".join(c.text for c in chunks)
    # Verify table row intactness
    assert "| Segment | FY 2025 | FY 2024 |" in all_chunk_texts
    assert "| Google Services | $304,000 | $270,000 |" in all_chunk_texts

    for chunk in chunks:
        assert chunk.strategy == "sentence_aware"
        assert chunk.token_count <= 40
        # Text slice verification
        assert sample_text[chunk.source_char_start:chunk.source_char_end] == chunk.text


def test_token_limit_respects_model_max():
    with pytest.raises(ValueError, match="exceeds maximum allowable content tokens"):
        fixed_token_chunking(
            text="Some sample text",
            document_id="limit_test",
            chunk_size=300,  # exceeds MAX_CONTENT_TOKENS (254)
        )

    with pytest.raises(ValueError, match="exceeds maximum allowable content tokens"):
        sentence_aware_chunking(
            text="Some sample text",
            document_id="limit_test",
            target_tokens=300,  # exceeds MAX_CONTENT_TOKENS (254)
        )


def test_invalid_parameters():
    text = "Valid sample sentence."
    # Overlap >= chunk_size
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        fixed_token_chunking(text=text, document_id="inv", chunk_size=50, overlap=50)

    # Negative overlap
    with pytest.raises(ValueError, match="cannot be negative"):
        fixed_token_chunking(text=text, document_id="inv", chunk_size=50, overlap=-5)

    # Empty text returns empty list
    assert fixed_token_chunking(text="", document_id="inv") == []
    assert sentence_aware_chunking(text="   \n\t", document_id="inv") == []
