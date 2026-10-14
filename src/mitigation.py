"""
Mitigation module implementing Reciprocal Rank Fusion (RRF).
Combines dense semantic vector retrieval with transparent lexical BM25 retrieval
to mitigate long-context retrieval fidelity loss and boundary fragmentation.

=============================================================================
MATHEMATICAL FOUNDATIONS OF RECIPROCAL RANK FUSION (RRF):
=============================================================================
Reciprocal Rank Fusion (Cormack, Clarke, and Buettcher, SIGIR 2009) is an
unsupervised, parameter-stable rank aggregation technique that integrates
disparate ranking permutations without requiring calibrated score normalization
(e.g., MinMax or Z-score normalization across dissimilar distributions).

1. Core Formulation:
   Given a set of candidate document chunks D and a collection of ranking
   permutations R = {R_1, R_2, ..., R_m}, the fused RRF score for chunk d in D is:
       RRF(d) = sum_{r in R} ( 1.0 / (c + rank_r(d)) )

   where:
   - rank_r(d) is the 1-based rank position of chunk d in ranking r (1 <= rank_r(d) <= M).
   - c is a non-negative smoothing constant (default c = 60.0).

2. Role of the Smoothing Constant c:
   - Dampens the disparity between top-tier rank positions. With c = 60:
       Rank 1 score contribution: 1 / (60 + 1) = 0.016393
       Rank 2 score contribution: 1 / (60 + 2) = 0.016129 (ratio: 0.9839)
       Rank 10 score contribution: 1 / (60 + 10) = 0.014286 (ratio: 0.8714)
   - If c were 0, Rank 1 (1.0) would drastically dominate Rank 2 (0.5),
     artificially favoring single-system outliers over consensus candidates.
   - When a chunk appears in both Dense and Lexical top candidate pools, its
     combined score achieves consensus boost:
       RRF(d) = 1 / (c + rank_dense(d)) + 1 / (c + rank_lexical(d))
     A document ranked #5 in both lists (2 * 1/65 = 0.03077) decisively beats a
     document ranked #1 in only one list (1/61 = 0.01639).

3. Handling Missing Candidates & Union Fusion:
   - Deep candidate lists of depth M (e.g., M = 50) are retrieved from each system.
   - The candidate pool is the union of chunk IDs across all rankings.
   - If a chunk appears in ranking R_1 but is missing from ranking R_2, its
     contribution from R_2 is 0.0 (no arbitrary penalty; absence simply yields no reciprocal bonus).

4. Strict Evaluation Guardrail:
   No evaluation labels (gold answer, evidence offsets, source position bucket,
   or report word count) are ever accessed or utilized during retrieval ranking.
=============================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from src.retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Standard RRF smoothing constant
DEFAULT_RRF_C = 60.0


def reciprocal_rank_fusion(
    rankings: list[list[RetrievalResult]],
    c: float = DEFAULT_RRF_C,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """
    Combines multiple retrieval rankings using Reciprocal Rank Fusion (RRF).

    Args:
        rankings: List of ranked lists, where each list contains RetrievalResult objects
                  ordered from best (rank 1) to worst.
        c: RRF smoothing constant (must be > 0.0, default 60.0).
        top_k: Maximum number of fused results to return (must be >= 1).

    Returns:
        List of RetrievalResult objects ordered by fused RRF score descending,
        with ranks re-assigned from 1 to top_k.

    Raises:
        ValueError: If c <= 0 or top_k <= 0.
    """
    if c <= 0.0:
        raise ValueError(f"RRF constant c must be strictly positive (got {c}).")
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer >= 1 (got {top_k}).")

    if not rankings or all(len(r) == 0 for r in rankings):
        return []

    # Map chunk_id -> accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # Map chunk_id -> representative RetrievalResult template
    candidate_metadata: dict[str, RetrievalResult] = {}

    for ranking_idx, ranked_list in enumerate(rankings):
        seen_in_this_ranking: set[str] = set()

        for res in ranked_list:
            cid = res.chunk_id
            if cid in seen_in_this_ranking:
                # Discard duplicate occurrences within the same ranking
                continue
            seen_in_this_ranking.add(cid)

            # Store metadata template from the first time this chunk is seen
            if cid not in candidate_metadata:
                candidate_metadata[cid] = res

            # Accumulate 1-based reciprocal rank score
            rank_1based = res.rank
            score_increment = 1.0 / (c + float(rank_1based))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_increment

    if not rrf_scores:
        return []

    # Deterministic sorting:
    # Primary: -rrf_score (descending)
    # Secondary: chunk_id (ascending)
    sorted_candidate_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: (-rrf_scores[cid], cid),
    )

    effective_k = min(top_k, len(sorted_candidate_ids))
    fused_results: list[RetrievalResult] = []

    for rank, cid in enumerate(sorted_candidate_ids[:effective_k], start=1):
        meta = candidate_metadata[cid]
        fused_score = float(rrf_scores[cid])

        fused_results.append(
            RetrievalResult(
                rank=rank,
                score=fused_score,
                chunk_id=meta.chunk_id,
                document_id=meta.document_id,
                title=meta.title,
                text=meta.text,
                strategy=f"{meta.strategy}+rrf",
                token_count=meta.token_count,
                source_char_start=meta.source_char_start,
                source_char_end=meta.source_char_end,
                source_word_start=meta.source_word_start,
                source_word_end=meta.source_word_end,
                source_position_pct=meta.source_position_pct,
                position_bucket=meta.position_bucket,
            )
        )

    return fused_results


def fuse_dense_and_lexical(
    dense_results: list[RetrievalResult],
    lexical_results: list[RetrievalResult],
    c: float = DEFAULT_RRF_C,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """
    Convenience helper fusing two candidate rankings (dense and lexical) via RRF.

    Args:
        dense_results: Ranked candidates from dense vector retriever.
        lexical_results: Ranked candidates from lexical BM25 retriever.
        c: RRF smoothing constant (default 60.0).
        top_k: Number of final results to return (default 10).

    Returns:
        Fused list of RetrievalResult objects.
    """
    return reciprocal_rank_fusion([dense_results, lexical_results], c=c, top_k=top_k)
