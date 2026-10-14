"""
Validation tests for the ground-truth benchmark dataset (data/benchmark_queries.json).
Verifies exact substring evidence fidelity, character/word offsets, position buckets,
and balanced distribution across Beginning, Middle, and End document sections.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_queries.json"
DOCS_DIR = PROJECT_ROOT / "data" / "documents"


@pytest.fixture(scope="module")
def benchmark_data():
    assert BENCHMARK_PATH.exists(), f"Benchmark file {BENCHMARK_PATH} does not exist."
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_benchmark_schema_and_counts(benchmark_data):
    assert benchmark_data["total_queries"] == 30
    assert len(benchmark_data["queries"]) == 30

    dist = benchmark_data["bucket_distribution"]
    assert dist["beginning"] == 10
    assert dist["middle"] == 10
    assert dist["end"] == 10

    # Ensure 3 queries per document
    doc_counts = {}
    for q in benchmark_data["queries"]:
        doc_counts[q["document_id"]] = doc_counts.get(q["document_id"], 0) + 1
    assert len(doc_counts) == 10
    for doc_id, count in doc_counts.items():
        assert count == 3, f"Document {doc_id} has {count} queries instead of 3."


def test_benchmark_exact_evidence_and_offsets(benchmark_data):
    for q in benchmark_data["queries"]:
        qid = q["question_id"]
        doc_path = DOCS_DIR / q["document_id"]
        assert doc_path.exists(), f"[{qid}] Missing source document: {doc_path}"

        text = doc_path.read_text(encoding="utf-8")
        words = text.split()
        total_words = len(words)

        assert q["total_word_count"] == total_words, f"[{qid}] Total word count mismatch"

        # 1. Exact character slice check
        c_start = q["source_char_start"]
        c_end = q["source_char_end"]
        evidence = q["evidence"]

        assert 0 <= c_start < c_end <= len(text), f"[{qid}] Invalid character offsets"
        assert text[c_start:c_end] == evidence, f"[{qid}] Substring at [{c_start}:{c_end}] does not match evidence"

        # 2. Word offset alignment
        w_start = q["source_word_start"]
        w_end = q["source_word_end"]
        assert 0 <= w_start <= w_end < total_words, f"[{qid}] Invalid word offsets"
        evidence_words = evidence.split()
        assert words[w_start : w_end + 1] == evidence_words, f"[{qid}] Word slice does not match evidence words"

        # 3. Normalized position calculation
        norm_pos = w_start / total_words
        assert abs(norm_pos - q["normalized_position"]) < 1e-3, f"[{qid}] Normalized position mismatch"

        # 4. Strict bucket boundary verification
        bucket = q["position_bucket"]
        if bucket == "beginning":
            assert norm_pos < 0.10, f"[{qid}] Beginning bucket violation: {norm_pos:.4f} >= 0.10"
        elif bucket == "middle":
            assert 0.40 <= norm_pos <= 0.60, f"[{qid}] Middle bucket violation: {norm_pos:.4f} not in [0.40, 0.60]"
        elif bucket == "end":
            assert norm_pos >= 0.90, f"[{qid}] End bucket violation: {norm_pos:.4f} < 0.90"
        else:
            pytest.fail(f"[{qid}] Unknown bucket {bucket}")


def test_benchmark_answer_and_question_completeness(benchmark_data):
    for q in benchmark_data["queries"]:
        qid = q["question_id"]
        assert len(q["question"].strip()) > 10, f"[{qid}] Question is too short"
        assert len(q["answer"].strip()) > 0, f"[{qid}] Answer cannot be empty"
        assert len(q["evidence"].strip()) > 10, f"[{qid}] Evidence passage is too short"
