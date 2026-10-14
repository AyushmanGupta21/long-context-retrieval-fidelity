"""
Evaluation harness and benchmark execution engine.
Implements rigorous, framework-free retrieval evaluation metrics:
Hit Rate@K, Precision@K, Recall@K, and Mean Reciprocal Rank (MRR),
with position-stratified breakdown (Beginning, Middle, End).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.chunking import Chunk
from src.embeddings import EmbeddingEngine
from src.retriever import RetrievalResult, retrieve_top_k

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuery:
    question_id: str
    document_id: str
    company: str
    filename: str
    question: str
    answer: str
    evidence: str
    source_char_start: int
    source_char_end: int
    source_word_start: int
    source_word_end: int
    total_word_count: int
    normalized_position: float
    position_bucket: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkQuery":
        return cls(
            question_id=data["question_id"],
            document_id=data["document_id"],
            company=data["company"],
            filename=data["filename"],
            question=data["question"],
            answer=data["answer"],
            evidence=data["evidence"],
            source_char_start=data["source_char_start"],
            source_char_end=data["source_char_end"],
            source_word_start=data["source_word_start"],
            source_word_end=data["source_word_end"],
            total_word_count=data["total_word_count"],
            normalized_position=data["normalized_position"],
            position_bucket=data["position_bucket"],
        )


def load_benchmark_queries(file_path: Path | str) -> list[BenchmarkQuery]:
    """
    Loads benchmark queries from JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [BenchmarkQuery.from_dict(q) for q in data["queries"]]


def is_chunk_relevant(chunk: Chunk, query: BenchmarkQuery) -> bool:
    """
    Documented Relevance Rule:
    A retrieved chunk is relevant if and only if:
    1. It belongs to the identical source document (document_id matches), AND
    2. It fully contains the annotated evidence passage:
       - Either the chunk's source character span encompasses the query's evidence character span, OR
       - The query's exact evidence text is a substring of the chunk text.
    """
    if chunk.document_id != query.document_id:
        return False

    # Check 1: Character span containment
    span_contains = (
        chunk.source_char_start <= query.source_char_start
        and chunk.source_char_end >= query.source_char_end
    )
    if span_contains:
        return True

    # Check 2: Verbatim substring containment
    if query.evidence in chunk.text:
        return True

    return False


def measure_semantic_completeness(chunks: list[Chunk]) -> float:
    """
    Empirical measure of semantic completeness:
    Computes the fraction of chunks that end cleanly on a natural boundary
    (sentence terminal punctuation '.', '!', '?', or structured table pipe '|').
    """
    if not chunks:
        return 0.0

    clean_count = 0
    for c in chunks:
        stripped = c.text.strip()
        if not stripped:
            continue
        if stripped[-1] in {".", "!", "?", "|", "\"", "'", "”"}:
            clean_count += 1

    return round(clean_count / len(chunks), 4)


def compute_query_metrics(
    retrieved_results: list[RetrievalResult],
    corpus_relevant_chunk_ids: set[str],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict[str, dict[str, float]]:
    """
    Computes retrieval metrics for a single query across multiple K values.
    """
    metrics_by_k: dict[str, dict[str, float]] = {}
    total_relevant_in_corpus = len(corpus_relevant_chunk_ids)

    # Find rank of first relevant chunk across entire retrieved list
    first_relevant_rank: Optional[int] = None
    for r in retrieved_results:
        if r.chunk_id in corpus_relevant_chunk_ids:
            first_relevant_rank = r.rank
            break

    for k in k_values:
        top_k = retrieved_results[:k]
        relevant_in_k = [r for r in top_k if r.chunk_id in corpus_relevant_chunk_ids]
        n_rel_in_k = len(relevant_in_k)

        # Hit Rate@K: 1 if at least one relevant chunk in top K, else 0
        hit = 1.0 if n_rel_in_k > 0 else 0.0

        # Precision@K: relevant in top K / K
        prec = n_rel_in_k / float(k)

        # Recall@K: relevant in top K / total relevant in corpus (0.0 if total is 0)
        rec = (n_rel_in_k / float(total_relevant_in_corpus)) if total_relevant_in_corpus > 0 else 0.0

        # MRR@K: 1 / rank if first relevant rank <= K, else 0.0
        if first_relevant_rank is not None and first_relevant_rank <= k:
            rr = 1.0 / float(first_relevant_rank)
        else:
            rr = 0.0

        metrics_by_k[f"k_{k}"] = {
            "hit_rate": hit,
            "precision": prec,
            "recall": rec,
            "mrr": rr,
        }

    return metrics_by_k


def evaluate_retrieval(
    queries: list[BenchmarkQuery],
    all_retrievals: list[list[RetrievalResult]],
    chunks: list[Chunk],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict[str, Any]:
    """
    Evaluates retrieval performance across all queries and position buckets.

    Handles queries with zero eligible chunks in the corpus without silently discarding them.
    """
    if len(queries) != len(all_retrievals):
        raise ValueError(f"Query count ({len(queries)}) does not match retrieval results ({len(all_retrievals)}).")

    # Map each query to its set of relevant chunk IDs in the indexed corpus
    chunk_by_id = {c.chunk_id: c for c in chunks}
    query_relevant_map: dict[str, set[str]] = {}
    eligible_query_count = 0

    for q in queries:
        rel_ids = {c.chunk_id for c in chunks if is_chunk_relevant(c, q)}
        query_relevant_map[q.question_id] = rel_ids
        if len(rel_ids) > 0:
            eligible_query_count += 1

    per_query_metrics: list[dict[str, Any]] = []
    bucket_groups: dict[str, list[dict[str, Any]]] = {"beginning": [], "middle": [], "end": []}

    for q, retrieved in zip(queries, all_retrievals):
        rel_ids = query_relevant_map[q.question_id]
        q_metrics = compute_query_metrics(retrieved, rel_ids, k_values=k_values)

        entry = {
            "question_id": q.question_id,
            "document_id": q.document_id,
            "company": q.company,
            "position_bucket": q.position_bucket,
            "normalized_position": q.normalized_position,
            "relevant_chunks_in_corpus": len(rel_ids),
            "metrics": q_metrics,
        }
        per_query_metrics.append(entry)
        bucket_groups[q.position_bucket].append(entry)

    # Function to aggregate metrics across a group of queries
    def _aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        if count == 0:
            return {}
        agg: dict[str, Any] = {"count": count}
        for k in k_values:
            k_key = f"k_{k}"
            hits = [item["metrics"][k_key]["hit_rate"] for item in group]
            precs = [item["metrics"][k_key]["precision"] for item in group]
            recs = [item["metrics"][k_key]["recall"] for item in group]
            mrrs = [item["metrics"][k_key]["mrr"] for item in group]

            agg[f"hit_rate@{k}"] = round(float(np.mean(hits)), 4)
            agg[f"precision@{k}"] = round(float(np.mean(precs)), 4)
            agg[f"recall@{k}"] = round(float(np.mean(recs)), 4)
            agg[f"mrr@{k}"] = round(float(np.mean(mrrs)), 4)
        return agg

    overall_metrics = _aggregate(per_query_metrics)
    position_breakdown = {
        bucket: _aggregate(bucket_groups[bucket]) for bucket in ["beginning", "middle", "end"]
    }

    # Chunk corpus statistics
    token_counts = [c.token_count for c in chunks] if chunks else [0]
    mean_chunk_tokens = round(float(np.mean(token_counts)), 2)
    semantic_completeness = measure_semantic_completeness(chunks)

    summary = {
        "total_queries_evaluated": len(queries),
        "queries_with_eligible_evidence": eligible_query_count,
        "queries_without_eligible_evidence": len(queries) - eligible_query_count,
        "total_corpus_chunks": len(chunks),
        "mean_chunk_tokens": mean_chunk_tokens,
        "semantic_completeness_rate": semantic_completeness,
        "overall_metrics": overall_metrics,
        "position_breakdown": position_breakdown,
        "per_query_details": per_query_metrics,
    }

    return summary
