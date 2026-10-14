"""
Embedding engine module for the Long-Context Retrieval Fidelity Engine.
Wraps sentence-transformers/all-MiniLM-L6-v2 without any third-party RAG frameworks
or vector databases.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Model specifications for sentence-transformers/all-MiniLM-L6-v2
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256

_GLOBAL_ENGINE: Optional["EmbeddingEngine"] = None


class EmbeddingEngine:
    """
    Dedicated embedding generator using sentence-transformers/all-MiniLM-L6-v2.

    Specifications:
    - Embedding Dimension: 384 float32 dimensions
    - Maximum Sequence Length: 256 tokens
    - Normalization: L2 normalized by default (unit length for exact cosine similarity via dot product)
    """

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.embedding_dim = EMBEDDING_DIM
        self.max_seq_length = MAX_SEQ_LENGTH

        logger.info("Initializing EmbeddingEngine with model: %s", self.model_name)
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=device)
            # Ensure model max sequence length matches MiniLM-L6-v2 specification
            self.model.max_seq_length = self.max_seq_length
        except Exception as exc:
            err_msg = (
                f"Failed to load SentenceTransformer model '{self.model_name}'. "
                f"Ensure internet connectivity is available or model weights are present in the local HuggingFace cache. "
                f"Underlying error: {exc}"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from exc

    def encode(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Computes dense vector representations for one or more text inputs.

        Args:
            texts: Single string or list of text strings.
            batch_size: Batch size for batched encoding (default: 32).
            normalize_embeddings: Whether to L2-normalize vectors to unit length (default: True).
            show_progress_bar: Whether to display encoding progress.

        Returns:
            2D NumPy array of shape (N, 384) with dtype float32.

        Raises:
            ValueError: If texts input is malformed, dimensions mismatch, or non-finite outputs occur.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, list):
            raise ValueError(f"Expected str or list of str, got {type(texts).__name__}.")

        if len(texts) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Ensure all items are strings
        cleaned_inputs: list[str] = []
        for idx, item in enumerate(texts):
            if item is None:
                raise ValueError(f"Input text at index {idx} is None.")
            if not isinstance(item, str):
                raise ValueError(f"Input text at index {idx} is not a string (got {type(item).__name__}).")
            cleaned_inputs.append(item)

        try:
            embeddings = self.model.encode(
                cleaned_inputs,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
            )
        except Exception as exc:
            raise RuntimeError(f"Encoding execution failed in model forward pass: {exc}") from exc

        # Validate shape and output integrity
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Unexpected embedding dimension: expected {self.embedding_dim}, got {embeddings.shape[1]}."
            )

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embedding model produced non-finite values (NaN or Inf) in vector matrix.")

        return embeddings.astype(np.float32)


def get_embedding_engine(model_name: str = MODEL_NAME) -> EmbeddingEngine:
    """
    Returns the cached singleton instance of EmbeddingEngine.
    Avoids redundant model loading across calls.
    """
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None or _GLOBAL_ENGINE.model_name != model_name:
        _GLOBAL_ENGINE = EmbeddingEngine(model_name=model_name)
    return _GLOBAL_ENGINE
