"""
Unit and integration tests for src/cli.py and src/indexer.py.
Tests single-query mode, retrieval mode selection (dense, bm25, rrf),
top-K limits, debug diagnostics, extractive evidence selection,
deterministic reproducibility, and index fingerprint validation.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

from src.chunking import Chunk
from src.cli import (
    RetrievalService,
    extract_best_sentence,
    extract_financial_table_evidence,
    select_evidence,
)
from src.indexer import CorpusIndex, compute_corpus_fingerprint, load_index
from src.lexical_retriever import BM25Retriever
from src.retriever import RetrievalResult


@pytest.fixture(scope="module")
def mock_service():
    """
    Creates a fast, self-contained mock RetrievalService with 4 chunks.
    """
    c1 = Chunk(
        chunk_id="chk_001",
        document_id="alphabet_2025_10k.txt",
        title="Alphabet Inc.",
        text="Alphabet operates Google Cloud including Google Cloud Platform and Google Workspace. Revenue increased 25%.",
        strategy="sentence",
        token_count=18,
        source_char_start=100,
        source_char_end=210,
        source_word_start=20,
        source_word_end=37,
        source_position_pct=0.05,
        position_bucket="beginning",
    )
    c2 = Chunk(
        chunk_id="chk_002",
        document_id="alphabet_2025_10k.txt",
        title="Alphabet Inc.",
        text="Google Cloud operating income reached record profitability in fiscal year 2025.",
        strategy="sentence",
        token_count=13,
        source_char_start=5000,
        source_char_end=5080,
        source_word_start=1000,
        source_word_end=1012,
        source_position_pct=0.50,
        position_bucket="middle",
    )
    c3 = Chunk(
        chunk_id="chk_003",
        document_id="amazon_2025_10k.txt",
        title="Amazon.com, Inc.",
        text="Amazon Web Services provides cloud storage, compute, and artificial intelligence solutions.",
        strategy="sentence",
        token_count=14,
        source_char_start=200,
        source_char_end=295,
        source_word_start=40,
        source_word_end=53,
        source_position_pct=0.08,
        position_bucket="beginning",
    )
    c4 = Chunk(
        chunk_id="chk_004",
        document_id="nvidia_2026_10k.txt",
        title="NVIDIA Corporation",
        text="Data Center GPU revenue was driven by Blackwell AI accelerators and accelerated computing.",
        strategy="sentence",
        token_count=14,
        source_char_start=3000,
        source_char_end=3095,
        source_word_start=600,
        source_word_end=613,
        source_position_pct=0.48,
        position_bucket="middle",
    )

    chunks = [c1, c2, c3, c4]

    # Deterministic synthetic embedding vectors (4, 384)
    np.random.seed(42)
    embs = np.random.randn(4, 384).astype(np.float32)
    # Normalize rows
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norms

    bm25 = BM25Retriever(chunks, k1=1.5, b=0.75)
    index = CorpusIndex(
        chunks=chunks,
        embeddings=embs,
        bm25=bm25,
        metadata={
            "fingerprint": "mock_fingerprint_12345",
            "total_chunks": 4,
            "embedding_dim": 384,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        },
    )

    return RetrievalService(index, candidate_depth=10, rrf_c=60.0)


def test_extractive_answer_candidate_selection():
    text = (
        "Alphabet is a global technology leader. "
        "For reporting purposes Google comprises two segments: Google Services and Google Cloud. "
        "Capital expenditures expanded significantly in 2025."
    )
    query = "What segments comprise Google?"
    best_sentence = extract_best_sentence(query, text)
    assert "Google comprises two segments: Google Services and Google Cloud" in best_sentence


def test_cli_search_dense_mode(mock_service):
    results, debug = mock_service.search("Google Cloud operating income", mode="dense", top_k=2)
    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert isinstance(results[0].score, float)
    assert -1.0 <= results[0].score <= 1.0


def test_cli_search_bm25_mode(mock_service):
    results, debug = mock_service.search("Blackwell AI accelerators NVIDIA", mode="bm25", top_k=2)
    assert len(results) == 2
    # NVIDIA chunk should be rank 1 due to exact lexical matches
    assert results[0].chunk_id == "chk_004"
    assert results[0].rank == 1
    assert results[0].score > 0.0


def test_cli_search_rrf_mode(mock_service):
    results, debug = mock_service.search("Amazon Web Services cloud compute", mode="rrf", top_k=3)
    assert len(results) == 3
    # Check that fusion breakdown exists in debug
    assert "fusion_breakdown" in debug
    assert len(debug["fusion_breakdown"]) == 3
    # RRF score must be positive reciprocal fraction
    assert 0.0 < results[0].score < 1.0
    # Verify rank order
    for idx, r in enumerate(results, start=1):
        assert r.rank == idx


def test_cli_top_k_limit(mock_service):
    results_1, _ = mock_service.search("cloud services", mode="rrf", top_k=1)
    assert len(results_1) == 1

    results_3, _ = mock_service.search("cloud services", mode="rrf", top_k=3)
    assert len(results_3) == 3


def test_cli_debug_diagnostics(mock_service):
    _, debug = mock_service.search("Alphabet Google Cloud", mode="rrf", top_k=2, debug=True)
    assert "query_tokens" in debug
    assert "timestamps" in debug
    assert "dense_top5" in debug
    assert "bm25_top5" in debug
    assert "fused_candidate_union_size" in debug
    assert debug["fused_candidate_union_size"] > 0


def test_cli_empty_query(mock_service):
    results, debug = mock_service.search("", mode="rrf", top_k=5)
    assert results == []

    results_spaces, _ = mock_service.search("   \n\t  ", mode="dense", top_k=5)
    assert results_spaces == []


def test_cli_deterministic_repeatability(mock_service):
    q = "Google Cloud infrastructure profitability"
    res1, _ = mock_service.search(q, mode="rrf", top_k=3)
    res2, _ = mock_service.search(q, mode="rrf", top_k=3)

    assert [r.chunk_id for r in res1] == [r.chunk_id for r in res2]
    assert [r.score for r in res1] == [r.score for r in res2]


def test_indexer_real_load_and_fingerprint():
    # Test loading real index from data/index
    index = load_index(check_fingerprint=True)
    assert len(index.chunks) == 3957
    assert index.embeddings.shape == (3957, 384)
    assert index.metadata["embedding_dim"] == 384
    assert index.metadata["chunk_strategy"] == "sentence_aware"

    # Verify fingerprint matches
    fp = compute_corpus_fingerprint()
    assert index.metadata["fingerprint"] == fp


def test_indexer_stale_fingerprint_detection(tmp_path):
    # Create a corrupted metadata with wrong fingerprint
    fake_dir = tmp_path / "fake_index"
    fake_dir.mkdir()

    import json
    chunks_file = fake_dir / "chunks.json"
    embs_file = fake_dir / "embeddings.npy"
    meta_file = fake_dir / "metadata.json"

    # Copy 1 mock chunk
    c = Chunk("c1", "doc.txt", "T", "text", "s", 5, 0, 4, 0, 1, 0.5, "middle")
    with open(chunks_file, "w") as f:
        json.dump([c.to_dict()], f)
    np.save(embs_file, np.zeros((1, 384), dtype=np.float32))

    with open(meta_file, "w") as f:
        json.dump({"fingerprint": "corrupted_stale_hash"}, f)

    # Must raise ValueError for stale fingerprint
    with pytest.raises(ValueError, match="Stale index detected"):
        load_index(fake_dir, check_fingerprint=True)


def test_format_passage_snippet_centering():
    from src.cli import format_passage_snippet

    long_chunk_text = (
        "We are continually enhancing these efforts over time, whether by enabling users to auto-delete their data, "
        "applying privacy technologies like on-device processing, giving people tools to control their experience, "
        "or advancing security features. Google For reporting purposes Google comprises two segments: "
        "Google Services and Google Cloud. Google Services Serving Our Users. We build helpful products worldwide."
    )
    query = "What cloud services does Alphabet operate?"

    snippet = format_passage_snippet(long_chunk_text, query=query, max_chars=180)
    # The snippet must center and visibly contain the salient matching sentence
    assert "Google comprises two segments: Google Services and Google Cloud" in snippet
    assert snippet.startswith("... ")


def test_chunk_evidence_consistency_against_disk_source():
    """
    Regression test verifying:
    1. A chunk ID resolves to exactly one chunk record.
    2. Its stored text matches the relevant slice of the correct source document.
    3. The extractive answer is an exact substring of that chunk.
    4. Document name, offsets, position, and text all refer to the same source.
    """
    index = load_index(check_fingerprint=True)
    target_id = "alphabet_2025_10k.txt_sent_0022"

    matching = [c for c in index.chunks if c.chunk_id == target_id]
    assert len(matching) == 1, f"Expected exactly 1 chunk record for {target_id}"

    chunk = matching[0]
    source_file = Path("data/documents") / chunk.document_id
    assert source_file.exists(), f"Source document {source_file} must exist"

    source_text = source_file.read_text(encoding="utf-8")
    sliced_text = source_text[chunk.source_char_start:chunk.source_char_end]
    assert sliced_text == chunk.text, "Chunk text must match exact character slice in source document"

    query = "What cloud services does Alphabet operate?"
    extractive = extract_best_sentence(query, chunk.text)
    assert extractive in chunk.text, "Extractive candidate must be a substring of the chunk"
    assert "Google comprises two segments: Google Services and Google Cloud" in extractive


@pytest.fixture(scope="module")
def real_cli_service():
    """Loads the real persistent index for CLI regression tests."""
    index = load_index(check_fingerprint=True)
    return RetrievalService(index, candidate_depth=50, rrf_c=60.0)


def test_regression_query_cost_of_sales(real_cli_service):
    """
    Regression Test 2: 'What was Amazon's cost of sales in 2025?'
    - Chunk amazon_2025_10k.txt_sent_0118 is retrieved at Rank #4.
    - select_evidence selects Rank #4 instead of Rank #1's FTC lawsuit.
    - Preserves row label 'Cost of sales', fiscal-year column headers '2024 | 2025',
      values '326,288' and '356,414', and units 'in millions'.
    - Excerpt is verified traceable to source document amazon_2025_10k.txt.
    """
    query = "What was Amazon's cost of sales in 2025?"
    results, _ = real_cli_service.search(query, mode="rrf", top_k=5)
    assert len(results) == 5

    # Target chunk must be present at Rank #4
    rank_4_result = results[3]
    assert rank_4_result.rank == 4
    assert rank_4_result.chunk_id == "amazon_2025_10k.txt_sent_0118"

    ev = select_evidence(query, results)
    assert ev["status"] == "direct_table_evidence"
    assert ev["rank"] == 4
    assert ev["source_chunk"].chunk_id == "amazon_2025_10k.txt_sent_0118"
    assert ev["table_details"]["row_label"] == "Cost of sales"
    assert ev["table_details"]["values"] == ["326,288", "356,414"]
    assert "2024" in ev["table_details"]["header"]
    assert "2025" in ev["table_details"]["header"]
    assert ev["table_details"]["units"] == "in millions"

    # Traceability check against raw source document
    doc_path = Path("data/documents") / ev["source_chunk"].document_id
    assert doc_path.exists()
    doc_text = doc_path.read_text(encoding="utf-8")
    assert ev["table_details"]["raw_line"] in doc_text
    assert "FTC" not in ev["excerpt"]


def test_regression_query_operating_income_vs_gross_profit(real_cli_service):
    """
    Regression Test 3: 'Why does Amazon consider operating income more meaningful than gross profit?'
    - amazon_2025_10k.txt_sent_0128 is retrieved at Rank #1.
    - select_evidence extracts the exact explanatory sentence at Rank #1.
    - Preserves Rank #1 behavior without misattributing other companies (e.g. Adobe).
    - Excerpt is verified traceable to source document amazon_2025_10k.txt.
    """
    query = "Why does Amazon consider operating income more meaningful than gross profit?"
    results, _ = real_cli_service.search(query, mode="rrf", top_k=5)
    assert len(results) == 5

    # Target chunk must be retrieved at Rank #1
    assert results[0].rank == 1
    assert results[0].chunk_id == "amazon_2025_10k.txt_sent_0128"

    ev = select_evidence(query, results)
    assert ev["status"] == "narrative_explanation"
    assert ev["rank"] == 1
    assert ev["source_chunk"].chunk_id == "amazon_2025_10k.txt_sent_0128"
    expected_sentence = (
        "We believe that operating income is a more meaningful measure than gross profit and "
        "gross margin due to the diversity of our product categories and services."
    )
    assert expected_sentence in ev["excerpt"]

    # Traceability check against raw source document
    doc_path = Path("data/documents") / ev["source_chunk"].document_id
    assert doc_path.exists()
    doc_text = doc_path.read_text(encoding="utf-8")
    assert expected_sentence in doc_text


def test_regression_query_consolidated_net_sales(real_cli_service):
    """
    Regression Test 1: 'What were Amazon's consolidated net sales in 2025?'
    - Chunk amazon_2025_10k.txt_sent_0115 is retrieved at Rank #3.
    - Expanding table structure via documented offsets retrieves the consolidated row
      (which was split at the chunk boundary into sent_0116).
    - Preserves row label 'Consolidated', fiscal-year column headers '2024 | 2025',
      values '637,959' and '716,924', year '2025', and units 'in millions'.
    - Does not select TOC page numbers ('Consolidated Statements of Cash Flows | 36')
      even when Top-K is increased to 30.
    - Excerpt is verified traceable to source document amazon_2025_10k.txt.
    """
    query = "What were Amazon's consolidated net sales in 2025?"
    results, _ = real_cli_service.search(query, mode="rrf", top_k=5)
    assert len(results) == 5

    ev = select_evidence(query, results)
    assert ev["status"] == "direct_table_evidence"
    assert ev["rank"] == 3
    assert ev["source_chunk"].chunk_id == "amazon_2025_10k.txt_sent_0115"
    assert ev["table_details"]["row_label"] == "Consolidated"
    assert ev["table_details"]["values"] == ["637,959", "716,924"]
    assert "2024" in ev["table_details"]["header"]
    assert "2025" in ev["table_details"]["header"]
    assert ev["table_details"]["units"] == "in millions"
    assert ev["table_details"]["year"] == "2025"
    assert ev["table_details"]["year_value"] == "716,924"

    # Explicit provenance distinction: row was recovered via adjacent continuation, not inside chunk
    assert ev["table_details"]["provenance_type"] == "adjacent_following_continuation"
    assert ev["table_details"]["row_char_start"] >= ev["source_chunk"].source_char_end
    assert "adjacent" in ev["excerpt"].lower()

    # Traceability check against raw source document
    doc_path = Path("data/documents") / ev["source_chunk"].document_id
    assert doc_path.exists()
    doc_text = doc_path.read_text(encoding="utf-8")
    assert ev["table_details"]["raw_line"] in doc_text
    assert "Cash Flows" not in ev["excerpt"]

    # Test with top_k=30 (reproducing the reported failure condition)
    results_30, _ = real_cli_service.search(query, mode="rrf", top_k=30)
    assert len(results_30) == 30
    ev_30 = select_evidence(query, results_30)
    assert ev_30["status"] == "direct_table_evidence"
    assert ev_30["table_details"]["row_label"] in ("Consolidated", "Total net sales")
    assert "716,924" in ev_30["table_details"]["values"]
    assert "Cash Flows" not in ev_30["excerpt"]
    assert "Page" not in ev_30["excerpt"]


def test_cross_company_financial_table_generalization(real_cli_service):
    """
    Generalization test verifying financial table extraction across multiple companies:
    - Apple: Operating income in 2025 ($133,050 million)
    - Alphabet: Operating income (Total income from operations) in 2025 ($129,039 million)
    - Meta: Net income in 2025 ($60,458 million)
    Verifies column-to-year alignment, denomination units, and source document traceability.
    """
    # 1. Apple Operating Income 2025
    q_apple = "What was Apple's operating income in 2025?"
    res_apple, _ = real_cli_service.search(q_apple, mode="rrf", top_k=10)
    ev_apple = select_evidence(q_apple, res_apple)
    assert ev_apple["status"] == "direct_table_evidence"
    assert ev_apple["table_details"]["row_label"] == "Operating income"
    assert ev_apple["table_details"]["year_value"] == "133,050"
    assert "2025" in ev_apple["table_details"]["header"]
    assert "millions" in ev_apple["table_details"]["units"].lower()

    # 2. Alphabet Operating Income 2025
    q_alpha = "What was Alphabet's operating income in 2025?"
    res_alpha, _ = real_cli_service.search(q_alpha, mode="rrf", top_k=10)
    ev_alpha = select_evidence(q_alpha, res_alpha)
    assert ev_alpha["status"] == "direct_table_evidence"
    assert ev_alpha["table_details"]["row_label"] in ("Total income from operations", "Operating income")
    assert ev_alpha["table_details"]["year_value"] == "129,039"
    assert "2025" in ev_alpha["table_details"]["header"]

    # 3. Meta Net Income 2025
    q_meta = "What was Meta's net income in 2025?"
    res_meta, _ = real_cli_service.search(q_meta, mode="rrf", top_k=10)
    ev_meta = select_evidence(q_meta, res_meta)
    assert ev_meta["status"] == "direct_table_evidence"
    assert ev_meta["table_details"]["row_label"] == "Net income"
    assert ev_meta["table_details"]["year_value"] == "60,458"
    assert "2025" in ev_meta["table_details"]["header"]


def test_financial_table_abstention_and_failure_behavior(real_cli_service):
    """
    Failure behavior test verifying faithful abstention:
    1. Non-existent reporting year (Amazon net sales in 2019):
       The filing only covers 2023-2025. Must abstain rather than present 2025's $716,924M.
    2. Unreported metric (Amazon gross profit in 2025):
       Amazon does not report gross profit. Must abstain rather than present unrelated numbers.
    """
    # 1. Non-existent reporting year
    q_wrong_year = "What were Amazon's consolidated net sales in 2019?"
    res_wrong_year, _ = real_cli_service.search(q_wrong_year, mode="rrf", top_k=10)
    ev_wrong_year = select_evidence(q_wrong_year, res_wrong_year)
    assert ev_wrong_year["status"] == "not_established"
    assert "not established" in ev_wrong_year["excerpt"].lower()
    assert "716,924" not in ev_wrong_year["excerpt"]

    # 2. Unreported metric
    q_unreported = "What was Amazon's gross profit in 2025?"
    res_unreported, _ = real_cli_service.search(q_unreported, mode="rrf", top_k=10)
    ev_unreported = select_evidence(q_unreported, res_unreported)
    assert ev_unreported["status"] == "not_established"
    assert "gross profit" in ev_unreported["excerpt"].lower()


def test_toc_row_never_accepted_as_financial_evidence(real_cli_service):
    """
    Directly exercises chunk amazon_2025_10k.txt_sent_0148 which contains the
    Table of Contents row '| Consolidated Statements of Cash Flows | 36 |'.
    Verifies that extract_financial_table_evidence returns None and
    select_evidence never accepts page 36 as financial evidence.
    """
    target_id = "amazon_2025_10k.txt_sent_0148"
    matching = [c for c in real_cli_service.index.chunks if c.chunk_id == target_id]
    assert len(matching) == 1
    c148 = matching[0]
    assert "Consolidated Statements of Cash Flows" in c148.text
    assert "36" in c148.text

    res = RetrievalResult(
        chunk_id=c148.chunk_id,
        document_id=c148.document_id,
        title=c148.title,
        text=c148.text,
        strategy=c148.strategy,
        source_position_pct=c148.source_position_pct,
        position_bucket=c148.position_bucket,
        token_count=c148.token_count,
        source_char_start=c148.source_char_start,
        source_char_end=c148.source_char_end,
        source_word_start=c148.source_word_start,
        source_word_end=c148.source_word_end,
        score=1.0,
        rank=1,
    )

    queries = [
        "What were Amazon's consolidated net sales in 2025?",
        "What was Amazon's cash flow in 2025?",
        "What were Amazon's consolidated revenues in 2025?",
    ]
    for q in queries:
        tbl = extract_financial_table_evidence(q, res)
        assert tbl is None, f"TOC chunk 148 must not be accepted as financial evidence for '{q}'"

        ev = select_evidence(q, [res])
        assert ev["status"] == "not_established"
        assert "36" not in ev.get("excerpt", "")
        assert "Cash Flows" not in ev.get("excerpt", "")


def test_regression_query_broad_what_is_amazon(real_cli_service):
    """
    Regression Test 7: Broad query 'what is amazon':
    - Does NOT select an exhibit index row (e.g. '| 97.1 | | Amazon.com, Inc. Clawback Policy')
    - Clearly abstains with status: 'not_established'
    - Displays retrieved candidate passages rather than an erroneous extractive answer
    """
    query = "what is amazon"
    results, _ = real_cli_service.search(query, mode="rrf", top_k=5)
    assert len(results) == 5

    ev = select_evidence(query, results)
    assert ev["status"] == "not_established"
    assert "not established" in ev["excerpt"].lower()
    assert "97.1" not in ev["excerpt"]
    assert "Clawback Policy" not in ev["excerpt"]
    assert "19.1" not in ev["excerpt"]


def test_cli_color_helpers_and_no_color(monkeypatch):
    """
    Tests that ANSI terminal coloring properly respects the NO_COLOR standard
    and non-interactive TTY output.
    """
    from src.cli import supports_color, c_bold, c_green, c_yellow

    # When NO_COLOR is set, color codes must be suppressed
    monkeypatch.setenv("NO_COLOR", "1")
    assert supports_color() is False
    assert c_bold("Test Text") == "Test Text"
    assert c_green("Success") == "Success"
    assert c_yellow("Warning") == "Warning"

    # When NO_COLOR is unset
    monkeypatch.delenv("NO_COLOR", raising=False)
    # If not on TTY, supports_color remains False safely
    if not sys.stdout.isatty():
        assert supports_color() is False


def test_cli_display_results_formatting_and_debug_sections(capsys, mock_service):
    """
    Tests that display_results executes cleanly and renders all 7 diagnostic
    trace sections (A, B, C, D, E, F, G) when debug_flag=True.
    """
    from src.cli import display_results

    q = "Alphabet Google Cloud Platform"
    results, debug_info = mock_service.search(q, mode="rrf", top_k=2, debug=True)

    # 1. Test normal display
    display_results(q, results, mode="rrf", debug_info=debug_info, debug_flag=False)
    captured = capsys.readouterr()
    assert "LONG-CONTEXT RETRIEVAL FIDELITY ENGINE" in captured.out
    assert "EXTRACTIVE ANSWER EVIDENCE EXCERPT" in captured.out
    assert "RANKED RETRIEVED CHUNKS" in captured.out
    assert "Score Semantics Note" in captured.out

    # 2. Test debug display with all 7 diagnostic sections
    display_results(q, results, mode="rrf", debug_info=debug_info, debug_flag=True)
    captured_debug = capsys.readouterr()
    assert "RETRIEVAL PIPELINE DIAGNOSTIC TRACE (--debug)" in captured_debug.out
    assert "[SECTION A: INPUT & SYSTEM CONFIGURATION]" in captured_debug.out
    assert "[SECTION B: QUERY EMBEDDING STAGE]" in captured_debug.out
    assert "[SECTION C: DENSE SEMANTIC RETRIEVAL" in captured_debug.out
    assert "[SECTION D: LEXICAL BM25 RETRIEVAL]" in captured_debug.out
    assert "[SECTION E: RECIPROCAL RANK FUSION (RRF)]" in captured_debug.out
    assert "[SECTION F: EVIDENCE SELECTION" in captured_debug.out
    assert "[SECTION G: FINAL DIAGNOSTIC & PIPELINE SUMMARY]" in captured_debug.out
    assert "Pipeline Latency Breakdown" in captured_debug.out


def test_cli_interactive_configuration_commands(capsys, mock_service, monkeypatch):
    """
    Tests interactive session command handling (:mode, :k, :debug, :config, :help, :quit).
    Simulates user inputs via monkeypatched stdin.
    """
    from src.cli import run_interactive

    simulated_inputs = iter([
        ":help",
        ":config",
        ":mode dense",
        ":k 3",
        ":debug on",
        ":quit",
    ])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(simulated_inputs))

    run_interactive(mock_service, default_mode="rrf", default_k=2, debug_flag=False)
    captured = capsys.readouterr()

    assert "INTERACTIVE SESSION" in captured.out
    assert "INTERACTIVE COMMAND REFERENCE" in captured.out
    assert "CURRENT RUNTIME CONFIGURATION" in captured.out
    assert "Retrieval mode changed to: DENSE" in captured.out
    assert "Top-K candidate depth changed to: 3" in captured.out
    assert "Debug diagnostic trace: ON" in captured.out
    assert "Session closed. Goodbye." in captured.out




