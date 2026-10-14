"""
Lexical retrieval core using Okapi BM25.
Implements pure Python and NumPy BM25 without external search engines,
vector databases, or third-party retrieval frameworks.

=============================================================================
MATHEMATICAL FOUNDATIONS & BM25 FORMULATION:
=============================================================================
Okapi BM25 (Robertson & Zaragoza, 2009) is a non-linear probabilistic lexical
ranking function that scores document relevance based on matching query terms,
incorporating term frequency saturation and document length normalization.

1. Term Frequency Saturation:
   Unlike linear TF-IDF where score grows unboundedly with term count, BM25
   uses a concave saturation function controlled by parameter k1:
       TF_sat(t, D) = (f(t, D) * (k1 + 1)) / (f(t, D) + k1 * (1 - b + b * (|D| / avgdl)))
   As f(t, D) -> inf, TF_sat -> k1 + 1.
   Default k1 = 1.5 calibrates the saturation rate for technical/financial disclosures.

2. Document Length Normalization:
   Parameter b in [0, 1] controls how aggressively long documents are penalized:
       len_norm(D) = 1 - b + b * (|D| / avgdl)
   - When b = 1.0: Full document length normalization (scores inversely proportional to length).
   - When b = 0.0: No length normalization (raw term count matters).
   Default b = 0.75 balances verbosity penalization against legitimate informative length.

3. Inverse Document Frequency (IDF):
   To guarantee non-negative IDF values even for terms appearing in > 50% of chunks,
   we implement the standard Robertson-Spärck-Jones smoothed formulation:
       IDF(t) = ln( 1.0 + (N - n(t) + 0.5) / (n(t) + 0.5) )
   where:
   - N is the total number of chunks in the corpus.
   - n(t) is the document frequency (number of chunks containing term t).
   - As n(t) -> N, IDF(t) -> ln(1 + 0.5 / (N + 0.5)) > 0 (always positive).

4. Total BM25 Score:
   For query Q = {q_1, q_2, ..., q_m} and chunk D:
       BM25(D, Q) = sum_{t in Q} IDF(t) * TF_sat(t, D)

5. Deterministic Tie-Breaking:
   Primary Sort: BM25 score descending.
   Secondary Sort: Chunk identifier (chunk_id) ascending.
   Tertiary Sort: Original chunk index ascending.
=============================================================================
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.chunking import Chunk
from src.retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Default Okapi BM25 hyperparameters
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

# Tokenization regex: extracts lowercase alphanumeric sequences
_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")


def tokenize_text(text: str) -> list[str]:
    """
    Standardized tokenizer and normalizer for both queries and chunks.
    Converts text to lowercase and extracts alphanumeric tokens.

    Args:
        text: Input text string.

    Returns:
        List of lowercase string tokens.
    """
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """
    In-memory Okapi BM25 lexical retriever operating over Chunk objects.
    Vectorized and pre-indexed for fast deterministic scoring across large corpora.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ):
        """
        Initializes and indexes the BM25 corpus.

        Args:
            chunks: List of Chunk objects to index.
            k1: Term frequency saturation parameter (must be >= 0, default 1.5).
            b: Document length normalization parameter (must be in [0.0, 1.0], default 0.75).
        """
        if k1 < 0.0:
            raise ValueError(f"BM25 parameter k1 must be non-negative (got {k1}).")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"BM25 parameter b must be between 0.0 and 1.0 (got {b}).")

        self.chunks = list(chunks)
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus_size = len(self.chunks)

        # Inverted index data structures:
        # doc_lens: array of token counts for each chunk
        self.doc_lens: np.ndarray = np.zeros(self.corpus_size, dtype=np.float32)
        # term_freqs: mapping from term -> dict[chunk_idx, count]
        self.inverted_index: dict[str, dict[int, int]] = {}
        # idf: mapping from term -> float
        self.idf: dict[str, float] = {}

        self._build_index()

    def _build_index(self) -> None:
        """
        Tokenizes all chunks, populates document lengths, inverted term frequencies,
        and pre-computes Robertson smoothed IDF for all terms in vocabulary.
        """
        if self.corpus_size == 0:
            self.avgdl = 0.0
            return

        doc_frequencies: Counter[str] = Counter()

        for idx, chunk in enumerate(self.chunks):
            tokens = tokenize_text(chunk.text)
            self.doc_lens[idx] = len(tokens)
            term_counts = Counter(tokens)

            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][idx] = count
                doc_frequencies[term] += 1

        total_tokens = float(np.sum(self.doc_lens))
        self.avgdl = (total_tokens / float(self.corpus_size)) if self.corpus_size > 0 else 0.0

        # Precompute Robertson-Spärck-Jones smoothed IDF
        N = float(self.corpus_size)
        for term, n_t in doc_frequencies.items():
            # ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
            val = (N - float(n_t) + 0.5) / (float(n_t) + 0.5)
            self.idf[term] = math.log(1.0 + val)

        logger.debug(
            "BM25 indexed %d chunks (%d unique terms, avgdl=%.2f)",
            self.corpus_size,
            len(self.idf),
            self.avgdl,
        )

    def compute_scores(self, query: str) -> np.ndarray:
        """
        Computes BM25 scores for all corpus chunks against a query.

        Args:
            query: Natural language query string.

        Returns:
            1D float32 NumPy array of shape (corpus_size,) containing BM25 scores.
        """
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if self.corpus_size == 0 or not query or not query.strip():
            return scores

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return scores

        # Calculate term frequencies within query for weighting if needed,
        # or use set for standard Okapi single-match accumulation
        unique_query_terms = set(query_tokens)

        # Precompute length normalization denominator component:
        # k1 * (1 - b + b * (doc_lens / avgdl))
        if self.avgdl > 0.0:
            len_norm = self.k1 * (1.0 - self.b + self.b * (self.doc_lens / self.avgdl))
        else:
            len_norm = np.full(self.corpus_size, self.k1, dtype=np.float32)

        for term in unique_query_terms:
            if term not in self.inverted_index:
                continue

            term_idf = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, tf in postings.items():
                tf_float = float(tf)
                # Saturation formula: (tf * (k1 + 1)) / (tf + len_norm[doc_idx])
                num = tf_float * (self.k1 + 1.0)
                denom = tf_float + len_norm[doc_idx]
                scores[doc_idx] += term_idf * (num / denom)

        return scores

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """
        Retrieves top-K chunks using deterministic BM25 ranking.

        Deterministic Tie-Breaking:
        Primary: BM25 score descending.
        Secondary: Chunk identifier ascending.
        Tertiary: Chunk index ascending.

        Args:
            query: Query string.
            k: Maximum number of results to return (must be >= 1).

        Returns:
            List of RetrievalResult objects ordered from rank 1 to k.
        """
        if k <= 0:
            raise ValueError(f"k must be a positive integer >= 1 (got {k}).")

        if self.corpus_size == 0:
            return []

        scores = self.compute_scores(query)
        effective_k = min(k, self.corpus_size)

        # Deterministic sorting
        sorted_indices = sorted(
            range(self.corpus_size),
            key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id, i),
        )[:effective_k]

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            c = self.chunks[idx]
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(scores[idx]),
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    title=c.title,
                    text=c.text,
                    strategy=c.strategy,
                    token_count=c.token_count,
                    source_char_start=c.source_char_start,
                    source_char_end=c.source_char_end,
                    source_word_start=c.source_word_start,
                    source_word_end=c.source_word_end,
                    source_position_pct=c.source_position_pct,
                    position_bucket=c.position_bucket,
                )
            )

        return results
