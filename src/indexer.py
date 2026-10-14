"""
Persistent indexing and cache management module for Long-Context Retrieval Fidelity Engine.
Handles serialization, versioning, SHA256 corpus fingerprinting, and rapid
sub-second startup for sentence-aware chunk embeddings and BM25 lexical structures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.chunking import Chunk, sentence_aware_chunking
from src.embeddings import EmbeddingEngine, get_embedding_engine
from src.lexical_retriever import BM25Retriever

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"
CACHE_DIR = PROJECT_ROOT / "data" / ".cache"

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_STRATEGY = "sentence_aware"
DEFAULT_TARGET_TOKENS = 192


def compute_corpus_fingerprint(
    manifest_path: Path = MANIFEST_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    chunk_strategy: str = DEFAULT_STRATEGY,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
) -> str:
    """
    Computes a deterministic SHA256 fingerprint representing the current corpus
    content, chunking configuration, and embedding model specifications.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Corpus manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Gather document identifiers and sha256 checksums in sorted order
    doc_signatures = []
    for d in sorted(manifest.get("documents", []), key=lambda x: x["local_filename"]):
        doc_signatures.append(f"{d['local_filename']}:{d.get('sha256', '')}")

    hasher = hashlib.sha256()
    hasher.update("|".join(doc_signatures).encode("utf-8"))
    hasher.update(f"|strategy={chunk_strategy}|tokens={target_tokens}|model={model_name}".encode("utf-8"))
    return hasher.hexdigest()


@dataclass
class CorpusIndex:
    """
    In-memory container for precomputed corpus chunks, dense embedding matrix,
    lexical BM25 index, and index versioning metadata.
    """
    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: BM25Retriever
    metadata: dict[str, Any]

    def __post_init__(self):
        self.chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}


def save_index(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    metadata: dict[str, Any],
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> None:
    """
    Persists chunk metadata, embedding array, and manifest metadata to disk.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = index_dir / "chunks.json"
    embeddings_path = index_dir / "embeddings.npy"
    metadata_path = index_dir / "metadata.json"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f)

    np.save(embeddings_path, embeddings)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved persistent index to %s (%d chunks)", index_dir, len(chunks))


def load_index(
    index_dir: Optional[Path] = None,
    check_fingerprint: bool = True,
    k1: float = 1.5,
    b: float = 0.75,
) -> CorpusIndex:
    """
    Loads precomputed chunks and embeddings from index directory.
    Constructs the BM25 lexical index in-memory in ~0.5s.

    Args:
        index_dir: Path to directory containing chunks.json, embeddings.npy, metadata.json.
        check_fingerprint: Whether to verify corpus SHA256 fingerprint against manifest.
        k1: BM25 saturation parameter.
        b: BM25 length normalization parameter.

    Returns:
        CorpusIndex object ready for dense, BM25, or RRF retrieval.

    Raises:
        FileNotFoundError: If index directory or index files do not exist.
        ValueError: If index fingerprint does not match active corpus manifest.
    """
    target_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR

    # Fallback to .cache if DEFAULT_INDEX_DIR doesn't have it yet
    if not (target_dir / "metadata.json").exists() and (CACHE_DIR / "sentence_aware_192_chunks.json").exists():
        target_dir = CACHE_DIR

    chunks_path = target_dir / "chunks.json"
    if not chunks_path.exists():
        chunks_path = target_dir / "sentence_aware_192_chunks.json"

    embeddings_path = target_dir / "embeddings.npy"
    if not embeddings_path.exists():
        embeddings_path = target_dir / "sentence_aware_192_embeddings.npy"

    metadata_path = target_dir / "metadata.json"

    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Persistent index files not found in {target_dir}. "
            f"Please run 'python scripts/build_index.py' to generate the index."
        )

    # Validate metadata and fingerprint if present
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    expected_fp = compute_corpus_fingerprint()
    cached_fp = metadata.get("fingerprint")

    if check_fingerprint and cached_fp and cached_fp != expected_fp:
        raise ValueError(
            f"Stale index detected! Fingerprint in index ({cached_fp[:10]}...) "
            f"does not match current corpus manifest ({expected_fp[:10]}...). "
            f"Run 'python scripts/build_index.py --force' to rebuild."
        )

    t0 = time.time()
    with open(chunks_path, "r", encoding="utf-8") as f:
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

    embeddings = np.load(embeddings_path)
    if len(chunks) != len(embeddings):
        raise ValueError(f"Index chunk count ({len(chunks)}) does not match embedding rows ({len(embeddings)}).")

    bm25 = BM25Retriever(chunks, k1=k1, b=b)
    load_dur = time.time() - t0
    logger.debug("Loaded CorpusIndex in %.3fs (%d chunks)", load_dur, len(chunks))

    if not metadata:
        metadata = {
            "fingerprint": expected_fp,
            "total_chunks": len(chunks),
            "embedding_dim": embeddings.shape[1],
            "model_name": DEFAULT_MODEL_NAME,
            "chunk_strategy": DEFAULT_STRATEGY,
            "target_tokens": DEFAULT_TARGET_TOKENS,
        }

    return CorpusIndex(chunks=chunks, embeddings=embeddings, bm25=bm25, metadata=metadata)
