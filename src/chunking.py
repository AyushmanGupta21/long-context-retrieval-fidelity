"""
Chunking module for the Long-Context Retrieval Fidelity Engine.
Implements fixed-size token-based chunking with configurable overlap and
structural sentence-aware chunking respecting sentence-transformers token limits.
"""

from __future__ import annotations

import bisect
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Model constants for sentence-transformers/all-MiniLM-L6-v2
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MAX_SEQ_LENGTH = 256
SPECIAL_TOKENS_COUNT = 2  # [CLS] and [SEP]
MAX_CONTENT_TOKENS = MODEL_MAX_SEQ_LENGTH - SPECIAL_TOKENS_COUNT  # 254
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 40

_CACHED_TOKENIZER = None


def get_tokenizer():
    """
    Loads and caches the HuggingFace tokenizer associated with all-MiniLM-L6-v2.
    """
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is None:
        from transformers import AutoTokenizer

        try:
            _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        except Exception:
            _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _CACHED_TOKENIZER


def classify_position_bucket(position_pct: float) -> str:
    """
    Classifies normalized document position into benchmark buckets:
    - First 10%: [0.0, 0.10] -> 'beginning'
    - Middle 40-60%: [0.40, 0.60] -> 'middle'
    - Last 10%: [0.90, 1.00] -> 'end'
    - Remaining ranges: (0.10, 0.40) and (0.60, 0.90) -> 'intermediate'
    """
    if 0.0 <= position_pct <= 0.10:
        return "beginning"
    elif 0.40 <= position_pct <= 0.60:
        return "middle"
    elif 0.90 <= position_pct <= 1.00:
        return "end"
    else:
        return "intermediate"


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    strategy: str
    token_count: int
    source_char_start: int
    source_char_end: int
    source_word_start: int
    source_word_end: int
    source_position_pct: float
    position_bucket: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_word_spans(text: str) -> list[tuple[int, int]]:
    """
    Returns list of (start_char, end_char) spans for all whitespace-delimited words.
    """
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _char_to_word_indices(
    char_start: int, char_end: int, word_spans: list[tuple[int, int]]
) -> tuple[int, int]:
    """
    Maps character start and end offsets to source word indices using binary search.
    """
    if not word_spans:
        return 0, 0

    starts = [s for s, e in word_spans]
    w_start = bisect.bisect_left(starts, char_start)
    if w_start >= len(word_spans):
        w_start = len(word_spans) - 1

    # For end word, find last word whose end is <= char_end
    ends = [e for s, e in word_spans]
    w_end = bisect.bisect_right(ends, char_end) - 1
    if w_end < w_start:
        w_end = w_start
    if w_end >= len(word_spans):
        w_end = len(word_spans) - 1

    return w_start, w_end


def fixed_token_chunking(
    text: str,
    document_id: str,
    title: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenizer: Optional[Any] = None,
) -> list[Chunk]:
    """
    Splits text into fixed-size token chunks with configurable token overlap.
    Uses the exact all-MiniLM-L6-v2 tokenizer to prevent silent truncation.

    Args:
        text: Document text.
        document_id: Unique document identifier.
        title: Document title.
        chunk_size: Number of content tokens per chunk (must be <= 254).
        overlap: Token overlap between consecutive chunks (must be < chunk_size).
        tokenizer: Optional pre-loaded tokenizer.

    Returns:
        List of Chunk objects with complete positional and offset metadata.
    """
    if not text or not text.strip():
        return []

    if chunk_size > MAX_CONTENT_TOKENS:
        raise ValueError(
            f"chunk_size {chunk_size} exceeds maximum allowable content tokens ({MAX_CONTENT_TOKENS}) "
            f"for {MODEL_NAME} (max_seq_length={MODEL_MAX_SEQ_LENGTH}, 2 special tokens reserved)."
        )
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size}).")

    if tokenizer is None:
        tokenizer = get_tokenizer()

    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]
    total_tokens = len(input_ids)

    if total_tokens == 0:
        return []

    word_spans = _compute_word_spans(text)
    total_chars = len(text)
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start_tok = 0
    chunk_idx = 0

    while start_tok < total_tokens:
        end_tok = min(start_tok + chunk_size, total_tokens)
        slice_offsets = offset_mapping[start_tok:end_tok]

        # Filter out 0-length offset tuples if any
        valid_offsets = [(s, e) for s, e in slice_offsets if s < e]
        if not valid_offsets:
            start_tok += step
            continue

        char_start = valid_offsets[0][0]
        char_end = valid_offsets[-1][1]
        chunk_text = text[char_start:char_end]
        token_count = end_tok - start_tok

        w_start, w_end = _char_to_word_indices(char_start, char_end, word_spans)
        pos_pct = (char_start + char_end) / (2.0 * total_chars) if total_chars > 0 else 0.0
        bucket = classify_position_bucket(pos_pct)

        chunk_id = f"{document_id}_fixed_{chunk_idx:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title=title,
                text=chunk_text,
                strategy="fixed_token_overlap",
                token_count=token_count,
                source_char_start=char_start,
                source_char_end=char_end,
                source_word_start=w_start,
                source_word_end=w_end,
                source_position_pct=round(pos_pct, 4),
                position_bucket=bucket,
            )
        )
        chunk_idx += 1

        if end_tok == total_tokens:
            break
        start_tok += step

    return chunks


def _extract_structural_segments(text: str) -> list[tuple[int, int, str]]:
    """
    Extracts atomic sentence and structural table row segments with their exact character offsets.
    Table rows (lines beginning and ending with '|') are kept intact.
    Narrative paragraphs are split on punctuation boundaries.
    """
    segments: list[tuple[int, int, str]] = []
    line_offset = 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            line_offset += len(line)
            continue

        # Preserve structured table rows as atomic segments
        if stripped.startswith("|") and stripped.endswith("|"):
            s_idx = text.find(stripped, line_offset)
            e_idx = s_idx + len(stripped)
            segments.append((s_idx, e_idx, stripped))
        else:
            # Narrative line: split into sentences
            start_in_line = text.find(stripped, line_offset)
            # Match sentences ending with . ! ? followed by space or line end
            sentence_matches = list(re.finditer(r"[^.!?\n]+[.!?]+(?:\s+|$)|[^.!?\n]+$", stripped))
            if sentence_matches:
                for sm in sentence_matches:
                    sent_str = sm.group().strip()
                    if sent_str:
                        s_pos = start_in_line + sm.start()
                        e_pos = s_pos + len(sent_str)
                        segments.append((s_pos, e_pos, sent_str))
            else:
                s_idx = start_in_line
                e_idx = s_idx + len(stripped)
                segments.append((s_idx, e_idx, stripped))

        line_offset += len(line)

    return segments


def sentence_aware_chunking(
    text: str,
    document_id: str,
    title: str = "",
    target_tokens: int = DEFAULT_CHUNK_SIZE,
    tokenizer: Optional[Any] = None,
) -> list[Chunk]:
    """
    Splits text into cohesive chunks aligned with structural sentence and table-row boundaries.
    Ensures chunks do not exceed MAX_CONTENT_TOKENS (254 tokens) to prevent silent model truncation.

    Args:
        text: Document text.
        document_id: Unique document identifier.
        title: Document title.
        target_tokens: Target content token limit per chunk (must be <= 254).
        tokenizer: Optional pre-loaded tokenizer.

    Returns:
        List of Chunk objects aligned to sentence boundaries.
    """
    if not text or not text.strip():
        return []

    if target_tokens > MAX_CONTENT_TOKENS:
        raise ValueError(
            f"target_tokens {target_tokens} exceeds maximum allowable content tokens ({MAX_CONTENT_TOKENS}) "
            f"for {MODEL_NAME} (max_seq_length={MODEL_MAX_SEQ_LENGTH}, 2 special tokens reserved)."
        )

    if tokenizer is None:
        tokenizer = get_tokenizer()

    raw_segments = _extract_structural_segments(text)
    if not raw_segments:
        return []

    word_spans = _compute_word_spans(text)
    total_chars = len(text)

    # Tokenize segments and sub-split any oversized individual segment
    processed_units: list[tuple[int, int, str, int]] = []
    for s_start, s_end, s_text in raw_segments:
        tokens = tokenizer.encode(s_text, add_special_tokens=False)
        t_count = len(tokens)
        if t_count <= target_tokens:
            processed_units.append((s_start, s_end, s_text, t_count))
        else:
            # Sub-split oversized single sentence by token offsets
            sub_chunks = fixed_token_chunking(
                text=s_text,
                document_id="sub",
                chunk_size=target_tokens,
                overlap=min(20, target_tokens // 4),
                tokenizer=tokenizer,
            )
            for sc in sub_chunks:
                actual_start = s_start + sc.source_char_start
                actual_end = s_start + sc.source_char_end
                processed_units.append((actual_start, actual_end, sc.text, sc.token_count))

    chunks: list[Chunk] = []
    chunk_idx = 0
    current_units: list[tuple[int, int, str, int]] = []
    current_tokens = 0

    for unit in processed_units:
        u_start, u_end, u_text, u_tokens = unit

        # Check if adding this unit exceeds target_tokens
        if current_units and (current_tokens + u_tokens > target_tokens):
            # Emit current accumulated chunk
            c_start = current_units[0][0]
            c_end = current_units[-1][1]
            c_text = text[c_start:c_end]
            c_tokens = current_tokens

            w_start, w_end = _char_to_word_indices(c_start, c_end, word_spans)
            pos_pct = (c_start + c_end) / (2.0 * total_chars) if total_chars > 0 else 0.0
            bucket = classify_position_bucket(pos_pct)

            chunk_id = f"{document_id}_sent_{chunk_idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    title=title,
                    text=c_text,
                    strategy="sentence_aware",
                    token_count=c_tokens,
                    source_char_start=c_start,
                    source_char_end=c_end,
                    source_word_start=w_start,
                    source_word_end=w_end,
                    source_position_pct=round(pos_pct, 4),
                    position_bucket=bucket,
                )
            )
            chunk_idx += 1
            current_units = []
            current_tokens = 0

        current_units.append(unit)
        current_tokens += u_tokens

    # Emit final remaining chunk
    if current_units:
        c_start = current_units[0][0]
        c_end = current_units[-1][1]
        c_text = text[c_start:c_end]
        c_tokens = current_tokens

        w_start, w_end = _char_to_word_indices(c_start, c_end, word_spans)
        pos_pct = (c_start + c_end) / (2.0 * total_chars) if total_chars > 0 else 0.0
        bucket = classify_position_bucket(pos_pct)

        chunk_id = f"{document_id}_sent_{chunk_idx:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title=title,
                text=c_text,
                strategy="sentence_aware",
                token_count=c_tokens,
                source_char_start=c_start,
                source_char_end=c_end,
                source_word_start=w_start,
                source_word_end=w_end,
                source_position_pct=round(pos_pct, 4),
                position_bucket=bucket,
            )
        )

    return chunks
