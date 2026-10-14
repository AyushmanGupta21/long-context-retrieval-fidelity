"""
End-to-end smoke test verifying real embedding generation with
sentence-transformers/all-MiniLM-L6-v2 coupled with pure NumPy Top-K retrieval.
"""

import numpy as np
import pytest
from src.chunking import Chunk
from src.embeddings import EmbeddingEngine, get_embedding_engine
from src.retriever import retrieve_top_k


def test_embedding_engine_properties():
    engine = get_embedding_engine()
    assert engine.embedding_dim == 384
    assert engine.max_seq_length == 256
    assert engine.model.max_seq_length == 256


def test_real_embedding_and_retrieval_smoke():
    engine = get_embedding_engine()

    texts = [
        "Alphabet operates Google Cloud providing high-performance computing, storage, and AI infrastructure.",
        "Amazon Web Services AWS delivers comprehensive cloud and database infrastructure globally.",
        "Netflix is an entertainment subscription service streaming movies and television series worldwide.",
        "Apple manufactures consumer hardware devices including iPhone, iPad, Mac computers, and Apple Watch.",
    ]

    # Generate real embeddings (384 dimensions)
    embeddings = engine.encode(texts, batch_size=4, normalize_embeddings=True)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (4, 384)
    assert embeddings.dtype == np.float32
    assert np.all(np.isfinite(embeddings))

    # Build dummy Chunk metadata for retrieval
    chunks = [
        Chunk("chk_goog", "googl_doc", "Alphabet", texts[0], "sentence_aware", 20, 0, 100, 0, 15, 0.05, "beginning"),
        Chunk("chk_amzn", "amzn_doc", "Amazon", texts[1], "sentence_aware", 18, 101, 200, 16, 30, 0.45, "middle"),
        Chunk("chk_nflx", "nflx_doc", "Netflix", texts[2], "sentence_aware", 19, 201, 300, 31, 45, 0.70, "intermediate"),
        Chunk("chk_aapl", "aapl_doc", "Apple", texts[3], "sentence_aware", 21, 301, 400, 46, 60, 0.95, "end"),
    ]

    # Query specifically targeted at Google Cloud
    query = "Which cloud computing and AI infrastructure services are operated by Google?"
    query_emb = engine.encode(query, normalize_embeddings=True)

    assert query_emb.shape == (1, 384)

    # Execute pure NumPy retrieval
    results = retrieve_top_k(
        query_vector=query_emb,
        corpus_matrix=embeddings,
        chunks=chunks,
        k=3,
    )

    assert len(results) == 3
    # Top ranked result must be Google Cloud
    assert results[0].rank == 1
    assert results[0].chunk_id == "chk_goog"
    assert results[0].document_id == "googl_doc"
    assert "Google Cloud" in results[0].text
    assert results[0].score > results[1].score
    assert results[0].score > 0.60  # Semantically strong match in MiniLM
    print(f"\n[Smoke Test Result] Query: '{query}'")
    for r in results:
        print(f"Rank {r.rank}: Score={r.score:.4f} | ID={r.chunk_id} | {r.text[:60]}...")
