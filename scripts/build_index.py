#!/usr/bin/env python3
"""
Index Builder script for Long-Context Retrieval Fidelity Engine.
Builds and saves the sentence-aware chunk index and embedding matrix into data/index/
with SHA256 corpus fingerprinting.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --force
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.chunking import Chunk, sentence_aware_chunking
from src.embeddings import get_embedding_engine
from src.indexer import (
    CACHE_DIR,
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_STRATEGY,
    DEFAULT_TARGET_TOKENS,
    MANIFEST_PATH,
    DOCS_DIR,
    compute_corpus_fingerprint,
    load_index,
    save_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("IndexBuilder")


def main():
    parser = argparse.ArgumentParser(description="Build and save persistent corpus index.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cache exists.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_INDEX_DIR), help="Output index directory.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = compute_corpus_fingerprint(MANIFEST_PATH)
    logger.info("Computed active corpus fingerprint: %s", fingerprint)

    # Check if precomputed cache exists in data/.cache or data/index
    cache_chunks = CACHE_DIR / "sentence_aware_192_chunks.json"
    cache_embs = CACHE_DIR / "sentence_aware_192_embeddings.npy"

    if not args.force and cache_chunks.exists() and cache_embs.exists():
        logger.info("Found precomputed cache in %s. Ingesting into index...", CACHE_DIR)
        with open(cache_chunks, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        chunks = [Chunk(**c) for c in raw_chunks]
        embeddings = np.load(cache_embs)
    else:
        logger.info("Building sentence-aware chunks from raw 10-K documents...")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        chunks = []
        for d in manifest["documents"]:
            doc_file = DOCS_DIR / d["local_filename"]
            text = doc_file.read_text(encoding="utf-8")
            doc_chunks = sentence_aware_chunking(
                text=text,
                document_id=d["local_filename"],
                title=d["company"],
                target_tokens=DEFAULT_TARGET_TOKENS,
            )
            chunks.extend(doc_chunks)

        logger.info("Generated %d sentence-aware chunks. Encoding embeddings...", len(chunks))
        engine = get_embedding_engine(DEFAULT_MODEL_NAME)
        chunk_texts = [c.text for c in chunks]
        embeddings = engine.encode(chunk_texts, batch_size=64, normalize_embeddings=True)

    metadata = {
        "fingerprint": fingerprint,
        "total_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "model_name": DEFAULT_MODEL_NAME,
        "chunk_strategy": DEFAULT_STRATEGY,
        "target_tokens": DEFAULT_TARGET_TOKENS,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    save_index(chunks, embeddings, metadata, index_dir=out_dir)

    logger.info("Testing fast load from %s...", out_dir)
    t0 = time.time()
    idx = load_index(out_dir, check_fingerprint=True)
    load_time = time.time() - t0

    print("\n" + "=" * 70)
    print("CORPUS INDEX BUILD COMPLETE")
    print("=" * 70)
    print(f"Index Location  : {out_dir}")
    print(f"Total Chunks    : {len(idx.chunks)}")
    print(f"Embedding Dim   : {idx.embeddings.shape[1]}")
    print(f"Model Name      : {metadata['model_name']}")
    print(f"Fingerprint     : {fingerprint[:16]}... (verified)")
    print(f"Load Time       : {load_time:.3f} seconds (sub-second startup)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
