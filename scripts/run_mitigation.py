#!/usr/bin/env python3
"""
Mitigation Experiment Runner for Long-Context Retrieval Fidelity Engine.
Evaluates Reciprocal Rank Fusion (RRF) combining dense semantic retrieval
(all-MiniLM-L6-v2 + pure NumPy cosine) and lexical retrieval (Okapi BM25).

Primary Comparison:
Apples-to-apples evaluation over the exact same sentence-aware chunk set (3,957 chunks)
and the exact same 30 ground-truth SEC Form 10-K benchmark questions:
1. Dense-only retrieval (MiniLM-L6-v2 + NumPy cosine)
2. Lexical-only retrieval (BM25Okapi, k1=1.5, b=0.75)
3. Hybrid RRF retrieval (candidate pool depth M=50, c=60.0)

Outputs:
- results/mitigated_metrics.json
- results/mitigation_comparison.json
- results/baseline_vs_mitigation.png
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import Chunk, sentence_aware_chunking
from src.embeddings import EmbeddingEngine, get_embedding_engine
from src.evaluator import BenchmarkQuery, evaluate_retrieval, load_benchmark_queries
from src.lexical_retriever import BM25Retriever
from src.mitigation import fuse_dense_and_lexical
from src.retriever import RetrievalResult, retrieve_top_k

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MitigationRunner")

MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_queries.json"
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "data" / ".cache"


def load_corpus() -> list[tuple[str, str, str]]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    corpus: list[tuple[str, str, str]] = []
    for d in manifest["documents"]:
        fn = d["local_filename"]
        fp = DOCS_DIR / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing document file: {fp}")
        text = fp.read_text(encoding="utf-8")
        corpus.append((fn, d["company"], text))
    return corpus


def get_or_compute_sentence_chunks_and_embeddings(
    corpus: list[tuple[str, str, str]],
    engine: EmbeddingEngine,
) -> tuple[list[Chunk], np.ndarray]:
    """
    Retrieves sentence-aware chunks and embeddings from disk cache if available,
    or computes and caches them to avoid redundant CPU encoding.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chunks_cache = CACHE_DIR / "sentence_aware_192_chunks.json"
    emb_cache = CACHE_DIR / "sentence_aware_192_embeddings.npy"

    if chunks_cache.exists() and emb_cache.exists():
        logger.info("Loading cached sentence-aware chunks and embeddings from %s...", CACHE_DIR)
        with open(chunks_cache, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                title=c["title"],
                text=c["text"],
                strategy=c["strategy"],
                token_count=c["token_count"],
                source_char_start=c["source_char_start"],
                source_char_end=c["source_char_end"],
                source_word_start=c["source_word_start"],
                source_word_end=c["source_word_end"],
                source_position_pct=c["source_position_pct"],
                position_bucket=c["position_bucket"],
            )
            for c in raw_chunks
        ]
        embeddings = np.load(emb_cache)
        if len(chunks) == len(embeddings):
            logger.info("Successfully loaded %d cached chunks and embeddings.", len(chunks))
            return chunks, embeddings
        logger.warning("Cache size mismatch, recomputing chunks and embeddings...")

    # Compute sentence-aware chunks
    logger.info("Chunking corpus with sentence_aware_chunking (target 192 tokens)...")
    t0 = time.time()
    chunks: list[Chunk] = []
    for doc_id, company, text in corpus:
        doc_chunks = sentence_aware_chunking(
            text=text,
            document_id=doc_id,
            title=company,
            target_tokens=192,
        )
        chunks.extend(doc_chunks)
    logger.info("Generated %d sentence-aware chunks in %.2fs", len(chunks), time.time() - t0)

    # Encode all chunks
    logger.info("Encoding %d chunks with %s...", len(chunks), engine.model_name)
    t_emb = time.time()
    chunk_texts = [c.text for c in chunks]
    embeddings = engine.encode(chunk_texts, batch_size=64, normalize_embeddings=True)
    logger.info("Encoded %d chunks in %.2fs", len(chunks), time.time() - t_emb)

    # Save to cache
    logger.info("Caching chunks and embeddings to %s...", CACHE_DIR)
    with open(chunks_cache, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f)
    np.save(emb_cache, embeddings)

    return chunks, embeddings


def plot_mitigation_comparison(
    dense_metrics: dict[str, Any],
    rrf_metrics: dict[str, Any],
    output_image_path: Path,
):
    """
    Generates a publication-quality comparative bar chart showing retrieval fidelity
    by position bucket (Beginning, Middle, End) for Dense-Only vs RRF.
    """
    buckets = ["beginning", "middle", "end"]
    labels = ["Beginning (<10%)", "Middle (40-60%)", "End (>=90%)"]

    dense_pos = dense_metrics["position_breakdown"]
    rrf_pos = rrf_metrics["position_breakdown"]

    hit1_dense = [dense_pos[b]["hit_rate@1"] for b in buckets]
    hit1_rrf = [rrf_pos[b]["hit_rate@1"] for b in buckets]

    hit5_dense = [dense_pos[b]["hit_rate@5"] for b in buckets]
    hit5_rrf = [rrf_pos[b]["hit_rate@5"] for b in buckets]

    mrr_dense = [dense_pos[b]["mrr@10"] for b in buckets]
    mrr_rrf = [rrf_pos[b]["mrr@10"] for b in buckets]

    x = np.arange(len(buckets))
    width = 0.14

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot grouped bars
    ax.bar(x - 2.5 * width, hit1_dense, width, label="Dense Hit@1", color="#aec7e8", edgecolor="#1f77b4")
    ax.bar(x - 1.5 * width, hit1_rrf, width, label="RRF Hit@1 (Mitigated)", color="#1f77b4")

    ax.bar(x - 0.5 * width, hit5_dense, width, label="Dense Hit@5", color="#98df8a", edgecolor="#2ca02c")
    ax.bar(x + 0.5 * width, hit5_rrf, width, label="RRF Hit@5 (Mitigated)", color="#2ca02c")

    ax.bar(x + 1.5 * width, mrr_dense, width, label="Dense MRR@10", color="#ff9896", edgecolor="#d62728")
    ax.bar(x + 2.5 * width, mrr_rrf, width, label="RRF MRR@10 (Mitigated)", color="#d62728")

    ax.set_title(
        "Retrieval Fidelity by Document Depth: Dense-Only vs. Reciprocal Rank Fusion (RRF)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Source Document Depth", fontsize=11, labelpad=10)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="semibold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, fontsize=9, loc="upper right", ncol=3)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Value labels
    for i in range(len(buckets)):
        ax.text(x[i] - 2.5 * width, hit1_dense[i] + 0.02, f"{hit1_dense[i]:.2f}", ha="center", fontsize=7.5)
        ax.text(x[i] - 1.5 * width, hit1_rrf[i] + 0.02, f"{hit1_rrf[i]:.2f}", ha="center", fontsize=7.5, fontweight="bold")
        ax.text(x[i] - 0.5 * width, hit5_dense[i] + 0.02, f"{hit5_dense[i]:.2f}", ha="center", fontsize=7.5)
        ax.text(x[i] + 0.5 * width, hit5_rrf[i] + 0.02, f"{hit5_rrf[i]:.2f}", ha="center", fontsize=7.5, fontweight="bold")
        ax.text(x[i] + 1.5 * width, mrr_dense[i] + 0.02, f"{mrr_dense[i]:.2f}", ha="center", fontsize=7.5)
        ax.text(x[i] + 2.5 * width, mrr_rrf[i] + 0.02, f"{mrr_rrf[i]:.2f}", ha="center", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()
    logger.info("Saved comparative visualization to %s", output_image_path)


def compute_relative_change(baseline_val: float, new_val: float) -> tuple[float, str]:
    delta = round(new_val - baseline_val, 4)
    if baseline_val > 0.0:
        pct = round(((new_val - baseline_val) / baseline_val) * 100.0, 2)
        pct_str = f"{pct:+.1f}%"
    else:
        pct_str = "N/A" if delta == 0.0 else "+inf"
    return delta, pct_str


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("Loading corpus and benchmark queries...")
    corpus = load_corpus()
    queries = load_benchmark_queries(BENCHMARK_PATH)
    logger.info("Loaded %d documents and %d benchmark queries.", len(corpus), len(queries))

    engine = get_embedding_engine()

    # Load or compute sentence-aware chunks & embeddings
    chunks, corpus_embeddings = get_or_compute_sentence_chunks_and_embeddings(corpus, engine)

    # Pre-encode all queries
    logger.info("Pre-encoding 30 benchmark queries with %s...", engine.model_name)
    query_texts = [q.question for q in queries]
    query_embeddings = engine.encode(query_texts, batch_size=32, normalize_embeddings=True)

    # Build BM25 index over the identical sentence-aware chunks
    logger.info("Indexing %d chunks with BM25Retriever (k1=1.5, b=0.75)...", len(chunks))
    t_bm25 = time.time()
    bm25 = BM25Retriever(chunks, k1=1.5, b=0.75)
    logger.info("BM25 indexing completed in %.2fs.", time.time() - t_bm25)

    # Retrieval candidate pool depth
    CANDIDATE_POOL_DEPTH = 50
    RRF_C = 60.0

    logger.info("Executing retrieval for %d queries (Candidate Pool M=%d, RRF c=%.1f)...", len(queries), CANDIDATE_POOL_DEPTH, RRF_C)

    dense_retrievals_top10: list[list[RetrievalResult]] = []
    lexical_retrievals_top10: list[list[RetrievalResult]] = []
    rrf_retrievals_top10: list[list[RetrievalResult]] = []

    for i, q in enumerate(queries):
        q_emb = query_embeddings[i : i + 1]

        # 1. Dense retrieval (top 50)
        dense_top50 = retrieve_top_k(
            query_vector=q_emb,
            corpus_matrix=corpus_embeddings,
            chunks=chunks,
            k=CANDIDATE_POOL_DEPTH,
        )

        # 2. Lexical retrieval (top 50)
        lexical_top50 = bm25.retrieve(
            query=q.question,
            k=CANDIDATE_POOL_DEPTH,
        )

        # 3. Hybrid RRF fusion (top 10 from union of top 50 candidates)
        fused_top10 = fuse_dense_and_lexical(
            dense_results=dense_top50,
            lexical_results=lexical_top50,
            c=RRF_C,
            top_k=10,
        )

        # Re-slice top 10 for pure Dense-only and Lexical-only baselines
        dense_top10 = [
            RetrievalResult(
                rank=r.rank,
                score=r.score,
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                title=r.title,
                text=r.text,
                strategy=r.strategy,
                token_count=r.token_count,
                source_char_start=r.source_char_start,
                source_char_end=r.source_char_end,
                source_word_start=r.source_word_start,
                source_word_end=r.source_word_end,
                source_position_pct=r.source_position_pct,
                position_bucket=r.position_bucket,
            )
            for r in dense_top50[:10]
        ]

        lexical_top10 = [
            RetrievalResult(
                rank=r.rank,
                score=r.score,
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                title=r.title,
                text=r.text,
                strategy=r.strategy,
                token_count=r.token_count,
                source_char_start=r.source_char_start,
                source_char_end=r.source_char_end,
                source_word_start=r.source_word_start,
                source_word_end=r.source_word_end,
                source_position_pct=r.source_position_pct,
                position_bucket=r.position_bucket,
            )
            for r in lexical_top50[:10]
        ]

        dense_retrievals_top10.append(dense_top10)
        lexical_retrievals_top10.append(lexical_top10)
        rrf_retrievals_top10.append(fused_top10)

    # Evaluate all three retrieval rankings on the exact same chunks
    logger.info("Evaluating Dense-Only baseline...")
    eval_dense = evaluate_retrieval(queries, dense_retrievals_top10, chunks, k_values=[1, 3, 5, 10])

    logger.info("Evaluating Lexical BM25-Only baseline...")
    eval_bm25 = evaluate_retrieval(queries, lexical_retrievals_top10, chunks, k_values=[1, 3, 5, 10])

    logger.info("Evaluating Mitigated RRF (Dense + BM25)...")
    eval_rrf = evaluate_retrieval(queries, rrf_retrievals_top10, chunks, k_values=[1, 3, 5, 10])

    # Load fixed token baseline for context
    fixed_token_baseline = None
    baseline_comp_path = RESULTS_DIR / "chunking_comparison.json"
    if baseline_comp_path.exists():
        with open(baseline_comp_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)
            fixed_token_baseline = b_data.get("configurations", {}).get("fixed_token_overlap_192")

    # Save mitigated_metrics.json
    mitigated_metrics_data = {
        "mitigation_method": "Reciprocal Rank Fusion (RRF)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "chunk_strategy": "sentence_aware",
            "target_chunk_tokens": 192,
            "total_corpus_chunks": len(chunks),
            "dense_retriever": {
                "model_name": engine.model_name,
                "dimension": engine.embedding_dim,
                "metric": "cosine_similarity (pure NumPy)",
                "candidate_pool_depth": CANDIDATE_POOL_DEPTH,
            },
            "lexical_retriever": {
                "algorithm": "BM25Okapi",
                "k1": 1.5,
                "b": 0.75,
                "candidate_pool_depth": CANDIDATE_POOL_DEPTH,
            },
            "rrf_parameters": {
                "c": RRF_C,
                "fused_k": 10,
                "rank_definition": "1-based",
            },
        },
        "overall_metrics": eval_rrf["overall_metrics"],
        "position_breakdown": eval_rrf["position_breakdown"],
        "total_queries_evaluated": eval_rrf["total_queries_evaluated"],
        "queries_with_eligible_evidence": eval_rrf["queries_with_eligible_evidence"],
        "queries_without_eligible_evidence": eval_rrf["queries_without_eligible_evidence"],
        "semantic_completeness_rate": eval_rrf["semantic_completeness_rate"],
        "per_query_details": eval_rrf["per_query_details"],
    }

    mitigated_metrics_path = RESULTS_DIR / "mitigated_metrics.json"
    with open(mitigated_metrics_path, "w", encoding="utf-8") as f:
        json.dump(mitigated_metrics_data, f, indent=2)
    logger.info("Saved mitigated metrics to %s", mitigated_metrics_path)

    # Compute deltas and build mitigation_comparison.json
    dense_ov = eval_dense["overall_metrics"]
    rrf_ov = eval_rrf["overall_metrics"]
    bm25_ov = eval_bm25["overall_metrics"]

    overall_deltas = {}
    for metric_key in ["hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@10", "precision@5", "recall@5", "mrr@10"]:
        d_val, d_pct = compute_relative_change(dense_ov[metric_key], rrf_ov[metric_key])
        overall_deltas[metric_key] = {
            "dense_baseline": dense_ov[metric_key],
            "bm25_baseline": bm25_ov[metric_key],
            "rrf_mitigated": rrf_ov[metric_key],
            "absolute_delta": d_val,
            "relative_change": d_pct,
        }

    position_deltas = {}
    for b in ["beginning", "middle", "end"]:
        position_deltas[b] = {}
        for m in ["hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@10", "mrr@10"]:
            b_dense = eval_dense["position_breakdown"][b][m]
            b_bm25 = eval_bm25["position_breakdown"][b][m]
            b_rrf = eval_rrf["position_breakdown"][b][m]
            d_val, d_pct = compute_relative_change(b_dense, b_rrf)
            position_deltas[b][m] = {
                "dense_baseline": b_dense,
                "bm25_baseline": b_bm25,
                "rrf_mitigated": b_rrf,
                "absolute_delta": d_val,
                "relative_change": d_pct,
            }

    mitigation_comparison_data = {
        "experiment_name": "Dense vs. BM25 vs. RRF Retrieval Fidelity Comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "apples_to_apples_corpus": {
            "chunk_strategy": "sentence_aware_192",
            "total_chunks": len(chunks),
            "total_queries": len(queries),
            "candidate_depth_each": CANDIDATE_POOL_DEPTH,
            "rrf_constant": RRF_C,
        },
        "overall_comparison": overall_deltas,
        "position_stratified_comparison": position_deltas,
        "context_fixed_192_baseline": fixed_token_baseline,
        "raw_summaries": {
            "dense_only": {
                "overall": eval_dense["overall_metrics"],
                "position_breakdown": eval_dense["position_breakdown"],
            },
            "bm25_only": {
                "overall": eval_bm25["overall_metrics"],
                "position_breakdown": eval_bm25["position_breakdown"],
            },
            "rrf_hybrid": {
                "overall": eval_rrf["overall_metrics"],
                "position_breakdown": eval_rrf["position_breakdown"],
            },
        },
    }

    mitigation_comparison_path = RESULTS_DIR / "mitigation_comparison.json"
    with open(mitigation_comparison_path, "w", encoding="utf-8") as f:
        json.dump(mitigation_comparison_data, f, indent=2)
    logger.info("Saved mitigation comparison to %s", mitigation_comparison_path)

    # Plot comparative chart
    chart_path = RESULTS_DIR / "baseline_vs_mitigation.png"
    plot_mitigation_comparison(eval_dense, eval_rrf, chart_path)

    total_duration = time.time() - t_start

    # Print summary table
    print("\n" + "=" * 90)
    print("RETRIEVAL FIDELITY MITIGATION RESULTS (DENSE VS. BM25 VS. RRF)")
    print("=" * 90)
    print(f"Total Documents: {len(corpus)} | Chunks: {len(chunks)} (Sentence-Aware) | Queries: {len(queries)}")
    print(f"Candidate Depth M: {CANDIDATE_POOL_DEPTH} | RRF Constant c: {RRF_C} | Duration: {total_duration:.1f}s")
    print("-" * 90)
    print("OVERALL CORPUS RETRIEVAL PERFORMANCE (30 QUERIES):")
    print(f"{'Metric':16s} | {'Dense Baseline':14s} | {'BM25 Baseline':14s} | {'RRF Mitigated':14s} | {'Delta':8s} | {'Rel %':8s}")
    print("-" * 90)
    for m, vals in overall_deltas.items():
        print(f"{m:16s} | {vals['dense_baseline']:14.4f} | {vals['bm25_baseline']:14.4f} | {vals['rrf_mitigated']:14.4f} | {vals['absolute_delta']:+8.4f} | {vals['relative_change']:8s}")

    print("\n" + "-" * 90)
    print("POSITION-STRATIFIED BREAKDOWN (DENSE VS. RRF):")
    for b in ["beginning", "middle", "end"]:
        print(f"\n--- Bucket: {b.upper()} (10 Queries) ---")
        print(f"{'Metric':16s} | {'Dense Baseline':14s} | {'RRF Mitigated':14s} | {'Delta':8s} | {'Rel %':8s}")
        for m, vals in position_deltas[b].items():
            print(f"{m:16s} | {vals['dense_baseline']:14.4f} | {vals['rrf_mitigated']:14.4f} | {vals['absolute_delta']:+8.4f} | {vals['relative_change']:8s}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
