"""
Unit tests for src/mitigation.py:
Tests Reciprocal Rank Fusion (RRF) score mathematics, union of disjoint candidate pools,
handling missing candidates, deterministic tie-breaking, and edge cases.
"""

import pytest
from src.mitigation import fuse_dense_and_lexical, reciprocal_rank_fusion
from src.retriever import RetrievalResult


def _make_result(rank: int, chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        document_id="doc_a.txt",
        title="Company A",
        text=f"Text for {chunk_id}",
        strategy="sentence",
        token_count=20,
        source_char_start=0,
        source_char_end=100,
        source_word_start=0,
        source_word_end=20,
        source_position_pct=0.5,
        position_bucket="middle",
    )


def test_rrf_mathematical_scoring():
    # Rank 1 in dense, Rank 2 in lexical
    dense = [_make_result(1, "chunk_consensus")]
    lexical = [_make_result(2, "chunk_consensus")]

    fused = reciprocal_rank_fusion([dense, lexical], c=60.0, top_k=5)
    assert len(fused) == 1
    # Expected score = 1/(60 + 1) + 1/(60 + 2) = 1/61 + 1/62
    expected_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert fused[0].score == pytest.approx(expected_score, abs=1e-7)
    assert fused[0].chunk_id == "chunk_consensus"
    assert fused[0].rank == 1


def test_rrf_consensus_beats_single_system_leader():
    # chunk_A is rank 1 in dense ONLY (1/61 = ~0.01639)
    # chunk_B is rank 2 in lexical ONLY (1/62 = ~0.01613)
    # chunk_C is rank 5 in dense AND rank 5 in lexical (1/65 + 1/65 = 2/65 = ~0.03077)
    dense = [
        _make_result(1, "chunk_A"),
        _make_result(5, "chunk_C"),
    ]
    lexical = [
        _make_result(2, "chunk_B"),
        _make_result(5, "chunk_C"),
    ]

    fused = fuse_dense_and_lexical(dense, lexical, c=60.0, top_k=5)

    assert len(fused) == 3
    # Consensus candidate chunk_C MUST be rank 1
    assert fused[0].chunk_id == "chunk_C"
    assert fused[0].rank == 1
    assert fused[0].score == pytest.approx(2.0 / 65.0, abs=1e-7)

    # chunk_A was rank 1 in dense -> score 1/61
    assert fused[1].chunk_id == "chunk_A"
    assert fused[1].rank == 2
    assert fused[1].score == pytest.approx(1.0 / 61.0, abs=1e-7)

    # chunk_B was rank 2 in lexical -> score 1/62
    assert fused[2].chunk_id == "chunk_B"
    assert fused[2].rank == 3
    assert fused[2].score == pytest.approx(1.0 / 62.0, abs=1e-7)


def test_rrf_union_disjoint_candidates():
    # Dense has [c1, c2], Lexical has [c3, c4] (completely disjoint)
    dense = [_make_result(1, "c1"), _make_result(2, "c2")]
    lexical = [_make_result(1, "c3"), _make_result(2, "c4")]

    fused = fuse_dense_and_lexical(dense, lexical, c=60.0, top_k=10)
    assert len(fused) == 4
    all_ids = {r.chunk_id for r in fused}
    assert all_ids == {"c1", "c2", "c3", "c4"}


def test_rrf_deterministic_tie_breaking():
    # c_beta and c_alpha both at rank 1 in different single systems -> identical score 1/61
    dense = [_make_result(1, "c_beta")]
    lexical = [_make_result(1, "c_alpha")]

    fused = fuse_dense_and_lexical(dense, lexical, c=60.0, top_k=2)
    assert len(fused) == 2
    assert fused[0].score == pytest.approx(fused[1].score)
    # Alphabetical tie-breaking: c_alpha before c_beta
    assert fused[0].chunk_id == "c_alpha"
    assert fused[1].chunk_id == "c_beta"
    assert fused[0].rank == 1
    assert fused[1].rank == 2


def test_rrf_duplicate_handling_within_same_ranking():
    # If a bug caused duplicate chunk IDs in one ranking, only the first rank counts
    dense = [
        _make_result(1, "chunk_dup"),
        _make_result(4, "chunk_dup"),
    ]
    fused = reciprocal_rank_fusion([dense], c=60.0, top_k=5)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1.0 / 61.0)


def test_rrf_invalid_parameters():
    dense = [_make_result(1, "c1")]
    with pytest.raises(ValueError, match="c"):
        reciprocal_rank_fusion([dense], c=0.0)

    with pytest.raises(ValueError, match="c"):
        reciprocal_rank_fusion([dense], c=-5.0)

    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([dense], top_k=0)


def test_rrf_empty_rankings():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
