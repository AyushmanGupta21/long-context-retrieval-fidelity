#!/usr/bin/env python3
"""
Baseline Experiment Runner for Long-Context Retrieval Fidelity Engine.
Executes the ground-truth benchmark over the 10 SEC Form 10-K reports without mitigation.
Evaluates:
1. Fixed-size overlapping chunking (192 tokens, overlap 38)
2. Fixed-size overlapping chunking (96 tokens, overlap 19)
3. Sentence-aware chunking (192 tokens target)
Outputs results/baseline_metrics.json, results/chunking_comparison.json, and results/baseline_position_accuracy.png.
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

from src.chunking import Chunk, fixed_token_chunking, sentence_aware_chunking
from src.embeddings import EmbeddingEngine, get_embedding_engine
from src.evaluator import BenchmarkQuery, evaluate_retrieval, load_benchmark_queries
from src.retriever import retrieve_top_k

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BaselineRunner")

MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_queries.json"
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_corpus() -> list[tuple[str, str, str]]:
    """
    Loads all documents listed in corpus manifest.
    Returns list of (doc_id, company, text).
    """
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


def build_corpus_chunks(
    corpus: list[tuple[str, str, str]],
    strategy: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[Chunk]:
    """
    Chunks all documents in corpus according to specified strategy.
    """
    all_chunks: list[Chunk] = []
    for doc_id, company, text in corpus:
        if strategy == "fixed":
            doc_chunks = fixed_token_chunking(
                text=text,
                document_id=doc_id,
                title=company,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        elif strategy == "sentence":
            doc_chunks = sentence_aware_chunking(
                text=text,
                document_id=doc_id,
                title=company,
                target_tokens=chunk_size,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        all_chunks.extend(doc_chunks)
    return all_chunks


def run_configuration(
    config_name: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    corpus: list[tuple[str, str, str]],
    queries: list[BenchmarkQuery],
    engine: EmbeddingEngine,
    query_embeddings: np.ndarray,
) -> dict[str, Any]:
    """
    Executes benchmark evaluation for a single chunking configuration.
    """
    logger.info("=== Running Configuration: %s ===", config_name)
    t0 = time.time()
    chunks = build_corpus_chunks(corpus, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
    chunking_dur = time.time() - t0
    logger.info("Generated %d chunks across 10 documents in %.2fs", len(chunks), chunking_dur)

    # Batch embed all chunks
    logger.info("Embedding %d chunks with %s...", len(chunks), engine.model_name)
    t_emb = time.time()
    chunk_texts = [c.text for c in chunks]
    corpus_embeddings = engine.encode(chunk_texts, batch_size=64, normalize_embeddings=True)
    emb_dur = time.time() - t_emb
    logger.info("Encoded in %.2fs (%.1f chunks/sec)", emb_dur, len(chunks) / emb_dur if emb_dur > 0 else 0)

    # Perform retrieval for all 30 queries
    logger.info("Performing pure NumPy top-10 retrieval for %d queries...", len(queries))
    all_retrievals = []
    for i, q in enumerate(queries):
        q_emb = query_embeddings[i : i + 1]
        results = retrieve_top_k(
            query_vector=q_emb,
            corpus_matrix=corpus_embeddings,
            chunks=chunks,
            k=10,
        )
        all_retrievals.append(results)

    # Evaluate metrics
    eval_summary = evaluate_retrieval(
        queries=queries,
        all_retrievals=all_retrievals,
        chunks=chunks,
        k_values=[1, 3, 5, 10],
    )

    eval_summary["configuration"] = {
        "config_name": config_name,
        "strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    eval_summary["execution_time_seconds"] = round(time.time() - t0, 2)

    return eval_summary


def plot_position_accuracy(
    baseline_metrics: dict[str, Any],
    output_image_path: Path,
):
    """
    Produces a comparative bar chart showing retrieval Hit Rate and MRR
    across Beginning, Middle, and End position buckets.
    """
    pos_data = baseline_metrics["position_breakdown"]
    buckets = ["beginning", "middle", "end"]
    labels = ["Beginning (<10%)", "Middle (40-60%)", "End (>=90%)"]

    hit_1 = [pos_data[b]["hit_rate@1"] for b in buckets]
    hit_3 = [pos_data[b]["hit_rate@3"] for b in buckets]
    hit_5 = [pos_data[b]["hit_rate@5"] for b in buckets]
    mrr_10 = [pos_data[b]["mrr@10"] for b in buckets]

    x = np.arange(len(buckets))
    width = 0.20

    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    plt.bar(x - 1.5 * width, hit_1, width, label="Hit Rate@1", color="#1f77b4", alpha=0.9)
    plt.bar(x - 0.5 * width, hit_3, width, label="Hit Rate@3", color="#ff7f0e", alpha=0.9)
    plt.bar(x + 0.5 * width, hit_5, width, label="Hit Rate@5", color="#2ca02c", alpha=0.9)
    plt.bar(x + 1.5 * width, mrr_10, width, label="MRR@10", color="#d62728", alpha=0.9)

    plt.title("Baseline Long-Context Retrieval Fidelity by Source Document Position", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Source Position Depth in Document", fontsize=12, labelpad=10)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate values above bars
    for i in range(len(buckets)):
        plt.text(x[i] - 1.5 * width, hit_1[i] + 0.02, f"{hit_1[i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i] - 0.5 * width, hit_3[i] + 0.02, f"{hit_3[i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i] + 0.5 * width, hit_5[i] + 0.02, f"{hit_5[i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i] + 1.5 * width, mrr_10[i] + 0.02, f"{mrr_10[i]:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()
    logger.info("Saved position fidelity chart to %s", output_image_path)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading corpus and benchmark queries...")
    corpus = load_corpus()
    queries = load_benchmark_queries(BENCHMARK_PATH)
    logger.info("Loaded %d documents and %d benchmark queries.", len(corpus), len(queries))

    engine = get_embedding_engine()

    # Pre-encode all queries
    logger.info("Pre-encoding 30 benchmark queries...")
    query_texts = [q.question for q in queries]
    query_embeddings = engine.encode(query_texts, batch_size=32, normalize_embeddings=True)

    # 1. Primary Baseline: Fixed-Size Overlapping (192 tokens, overlap 38)
    baseline_eval = run_configuration(
        config_name="fixed_token_overlap_192",
        strategy="fixed",
        chunk_size=192,
        overlap=38,
        corpus=corpus,
        queries=queries,
        engine=engine,
        query_embeddings=query_embeddings,
    )

    # 2. Chunk-Size Tradeoff: Fixed-Size Overlapping (96 tokens, overlap 19)
    fixed_96_eval = run_configuration(
        config_name="fixed_token_overlap_96",
        strategy="fixed",
        chunk_size=96,
        overlap=19,
        corpus=corpus,
        queries=queries,
        engine=engine,
        query_embeddings=query_embeddings,
    )

    # 3. Sentence-Aware Chunking (192 target tokens)
    sentence_eval = run_configuration(
        config_name="sentence_aware_192",
        strategy="sentence",
        chunk_size=192,
        overlap=0,
        corpus=corpus,
        queries=queries,
        engine=engine,
        query_embeddings=query_embeddings,
    )

    # Save primary baseline metrics
    baseline_metrics_path = RESULTS_DIR / "baseline_metrics.json"
    with open(baseline_metrics_path, "w", encoding="utf-8") as f:
        json.dump(baseline_eval, f, indent=2)
    logger.info("Saved baseline metrics to %s", baseline_metrics_path)

    # Save comparison across chunking strategies
    comparison_data = {
        "benchmark_name": "Chunking Strategy & Size Tradeoff Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configurations": {
            "fixed_token_overlap_192": {
                "overall": baseline_eval["overall_metrics"],
                "position_breakdown": baseline_eval["position_breakdown"],
                "total_chunks": baseline_eval["total_corpus_chunks"],
                "mean_chunk_tokens": baseline_eval["mean_chunk_tokens"],
                "semantic_completeness": baseline_eval["semantic_completeness_rate"],
            },
            "fixed_token_overlap_96": {
                "overall": fixed_96_eval["overall_metrics"],
                "position_breakdown": fixed_96_eval["position_breakdown"],
                "total_chunks": fixed_96_eval["total_corpus_chunks"],
                "mean_chunk_tokens": fixed_96_eval["mean_chunk_tokens"],
                "semantic_completeness": fixed_96_eval["semantic_completeness_rate"],
            },
            "sentence_aware_192": {
                "overall": sentence_eval["overall_metrics"],
                "position_breakdown": sentence_eval["position_breakdown"],
                "total_chunks": sentence_eval["total_corpus_chunks"],
                "mean_chunk_tokens": sentence_eval["mean_chunk_tokens"],
                "semantic_completeness": sentence_eval["semantic_completeness_rate"],
            },
        },
    }

    comparison_path = RESULTS_DIR / "chunking_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    logger.info("Saved chunking comparison to %s", comparison_path)

    # Generate comparative chart
    chart_path = RESULTS_DIR / "baseline_position_accuracy.png"
    plot_position_accuracy(baseline_eval, chart_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("BASELINE RETRIEVAL FIDELITY EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Documents: {len(corpus)} | Total Queries: {len(queries)} (10 Beg, 10 Mid, 10 End)")
    print("\n--- 1. Primary Baseline (Fixed Token 192, Overlap 38) ---")
    ov = baseline_eval["overall_metrics"]
    print(f"Overall Hit Rate@1:  {ov['hit_rate@1']:.4f} | Hit Rate@3: {ov['hit_rate@3']:.4f} | Hit Rate@5: {ov['hit_rate@5']:.4f} | Hit Rate@10: {ov['hit_rate@10']:.4f}")
    print(f"Overall MRR@10:      {ov['mrr@10']:.4f} | Precision@5: {ov['precision@5']:.4f} | Recall@5: {ov['recall@5']:.4f}")

    print("\nPosition-Stratified Performance (Fixed Token 192):")
    for b in ["beginning", "middle", "end"]:
        bp = baseline_eval["position_breakdown"][b]
        print(f"  {b.capitalize():10s}: Hit@1={bp['hit_rate@1']:.2f} | Hit@3={bp['hit_rate@3']:.2f} | Hit@5={bp['hit_rate@5']:.2f} | Hit@10={bp['hit_rate@10']:.2f} | MRR@10={bp['mrr@10']:.4f}")

    print("\n--- 2. Chunking Strategy & Size Comparison ---")
    for cfg_name, cfg in comparison_data["configurations"].items():
        o = cfg["overall"]
        print(f"  {cfg_name:24s} | Chunks: {cfg['total_chunks']:5d} | Mean Tok: {cfg['mean_chunk_tokens']:6.1f} | Sem. Compl.: {cfg['semantic_completeness']*100:5.1f}% | Hit@5: {o['hit_rate@5']:.2f} | MRR@10: {o['mrr@10']:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
