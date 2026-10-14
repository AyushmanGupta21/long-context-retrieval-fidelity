"""
Unit tests for corpus downloader logic, user agent resolution,
manifest structure, and safe rerun behavior.
Uses mocks to avoid live network requests during tests.
"""

import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
from scripts.download_corpus import (
    FilingTarget,
    get_user_agent,
    download_and_process_target,
    run_downloader,
    FILING_TARGETS,
)


def test_filing_targets_count_and_uniqueness():
    """Verify that we define exactly 10 targets with distinct filenames and URLs."""
    assert len(FILING_TARGETS) == 10
    filenames = [t.local_filename for t in FILING_TARGETS]
    urls = [t.url for t in FILING_TARGETS]
    assert len(set(filenames)) == 10
    assert len(set(urls)) == 10


def test_get_user_agent_fallback(monkeypatch):
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    ua = get_user_agent(None)
    assert "FidelityEngine/1.0" in ua
    assert "researcher@example.com" in ua


def test_get_user_agent_from_env(monkeypatch):
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "custom_analyst@firm.com")
    ua = get_user_agent(None)
    assert "custom_analyst@firm.com" in ua


def test_get_user_agent_from_cli(monkeypatch):
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "env_user@firm.com")
    ua = get_user_agent("cli_override@firm.com")
    assert "cli_override@firm.com" in ua


def test_download_and_process_mock_success(tmp_path):
    target = FilingTarget(
        company="Mock Corp",
        ticker="MCK",
        fiscal_period="FY 2025",
        report_title="Form 10-K",
        url="https://www.sec.gov/mock.htm",
        local_filename="mock_2025_10k.txt",
    )

    mock_html = "<html><body>" + ("<p>Sample corporate disclosure content sentence.</p>" * 1000) + "</body></html>"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = mock_html

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp

    output_dir = tmp_path / "documents"
    output_dir.mkdir()

    record = download_and_process_target(
        target=target,
        session=mock_session,
        output_dir=output_dir,
        min_words=5000,
        force=False,
        delay=0.0,
    )

    assert record["status"] == "validated"
    assert record["actual_word_count"] >= 5000
    output_file = output_dir / "mock_2025_10k.txt"
    assert output_file.exists()
    assert len(output_file.read_text(encoding="utf-8")) > 0


def test_download_and_process_mock_http_error(tmp_path):
    target = FilingTarget(
        company="Mock Corp",
        ticker="MCK",
        fiscal_period="FY 2025",
        report_title="Form 10-K",
        url="https://www.sec.gov/mock_404.htm",
        local_filename="mock_404_10k.txt",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.reason = "Not Found"

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp

    output_dir = tmp_path / "documents"
    output_dir.mkdir()

    record = download_and_process_target(
        target=target,
        session=mock_session,
        output_dir=output_dir,
        min_words=5000,
        force=False,
        delay=0.0,
    )

    assert record["status"] == "failed_download"
    assert "404" in record["error_message"]
    output_file = output_dir / "mock_404_10k.txt"
    assert not output_file.exists()


def test_safe_rerun_skips_download_if_valid(tmp_path):
    target = FilingTarget(
        company="Mock Corp",
        ticker="MCK",
        fiscal_period="FY 2025",
        report_title="Form 10-K",
        url="https://www.sec.gov/mock.htm",
        local_filename="mock_existing_10k.txt",
    )
    output_dir = tmp_path / "documents"
    output_dir.mkdir()
    output_file = output_dir / "mock_existing_10k.txt"
    valid_text = "word " * 5200
    output_file.write_text(valid_text, encoding="utf-8")

    mock_session = MagicMock()

    record = download_and_process_target(
        target=target,
        session=mock_session,
        output_dir=output_dir,
        min_words=5000,
        force=False,
        delay=0.0,
    )

    assert record["status"] == "validated"
    assert record["actual_word_count"] == 5200
    # Crucially, session.get should NOT have been called
    mock_session.get.assert_not_called()
