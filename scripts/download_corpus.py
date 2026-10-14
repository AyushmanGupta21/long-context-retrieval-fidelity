#!/usr/bin/env python3
"""
SEC Form 10-K Corpus Downloader & Cleaner.
Downloads official Form 10-K filings from SEC EDGAR Archives,
cleans HTML/iXBRL while preserving tables and structure,
validates minimum word counts (>= 5,000 words),
and generates data/corpus_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Ensure project root is in sys.path so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import clean_sec_html, count_words, validate_document

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CorpusDownloader")


@dataclass
class FilingTarget:
    company: str
    ticker: str
    fiscal_period: str
    report_title: str
    url: str
    local_filename: str


# 10 Primary Form 10-K Filings from SEC EDGAR Archives
FILING_TARGETS: list[FilingTarget] = [
    FilingTarget(
        company="Alphabet Inc.",
        ticker="GOOGL",
        fiscal_period="FY 2025 (Period ended Dec 31, 2025)",
        report_title="Form 10-K Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/1652044/000165204426000018/goog-20251231.htm",
        local_filename="alphabet_2025_10k.txt",
    ),
    FilingTarget(
        company="Amazon.com, Inc.",
        ticker="AMZN",
        fiscal_period="FY 2025 (Period ended Dec 31, 2025)",
        report_title="Form 10-K Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm",
        local_filename="amazon_2025_10k.txt",
    ),
    FilingTarget(
        company="Meta Platforms, Inc.",
        ticker="META",
        fiscal_period="FY 2025 (Period ended Dec 31, 2025)",
        report_title="Form 10-K Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/1326801/000162828026003942/meta-20251231.htm",
        local_filename="meta_2025_10k.txt",
    ),
    FilingTarget(
        company="International Business Machines Corp.",
        ticker="IBM",
        fiscal_period="FY 2025 (Period ended Dec 31, 2025)",
        report_title="Form 10-K Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/51143/000005114326000010/ibm-20251231.htm",
        local_filename="ibm_2025_10k.txt",
    ),
    FilingTarget(
        company="Netflix, Inc.",
        ticker="NFLX",
        fiscal_period="FY 2025 (Period ended Dec 31, 2025)",
        report_title="Form 10-K Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/1065280/000106528026000034/nflx-20251231.htm",
        local_filename="netflix_2025_10k.txt",
    ),
    FilingTarget(
        company="Microsoft Corporation",
        ticker="MSFT",
        fiscal_period="FY 2026 (Period ended June 30, 2026)",
        report_title="Form 10-K Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/msft-20260630.htm",
        local_filename="microsoft_2026_10k.txt",
    ),
    FilingTarget(
        company="NVIDIA Corporation",
        ticker="NVDA",
        fiscal_period="FY 2026 (Period ended Jan 25, 2026)",
        report_title="Form 10-K Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm",
        local_filename="nvidia_2026_10k.txt",
    ),
    FilingTarget(
        company="Apple Inc.",
        ticker="AAPL",
        fiscal_period="FY 2025 (Period ended Sept 27, 2025)",
        report_title="Form 10-K Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
        local_filename="apple_2025_10k.txt",
    ),
    FilingTarget(
        company="Adobe Inc.",
        ticker="ADBE",
        fiscal_period="FY 2025 (Period ended Nov 28, 2025)",
        report_title="Form 10-K Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/796343/000079634326000003/adbe-20251128.htm",
        local_filename="adobe_2025_10k.txt",
    ),
    FilingTarget(
        company="Oracle Corporation",
        ticker="ORCL",
        fiscal_period="FY 2026 (Period ended May 31, 2026)",
        report_title="Form 10-K Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/1341439/000119312526277521/orcl-20260531.htm",
        local_filename="oracle_2026_10k.txt",
    ),
]


def get_user_agent(cli_email: Optional[str] = None) -> str:
    """
    Constructs an SEC-compliant User-Agent header.
    Adheres to SEC guidelines: Sample Company Name AdminContact@<sample company domain>.com
    """
    contact_email = cli_email or os.environ.get("SEC_CONTACT_EMAIL")
    if not contact_email:
        contact_email = "researcher@example.com"
        logger.warning(
            "Environment variable 'SEC_CONTACT_EMAIL' is not set. "
            "Falling back to default '%s'. "
            "Set SEC_CONTACT_EMAIL in your environment for custom identification.",
            contact_email,
        )
    return f"FidelityEngine/1.0 ({contact_email})"


def create_session(user_agent: str) -> requests.Session:
    """
    Creates a requests.Session configured with standard SEC headers and exponential backoff retries.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def download_and_process_target(
    target: FilingTarget,
    session: requests.Session,
    output_dir: Path,
    min_words: int,
    force: bool = False,
    delay: float = 0.5,
) -> dict[str, Any]:
    """
    Downloads, parses, validates, and stores a single 10-K filing.
    """
    output_file = output_dir / target.local_filename
    try:
        file_path_str = str(output_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        file_path_str = str(output_file).replace("\\", "/")

    manifest_entry: dict[str, Any] = {
        "company": target.company,
        "ticker": target.ticker,
        "fiscal_period": target.fiscal_period,
        "report_title": target.report_title,
        "original_url": target.url,
        "local_filename": target.local_filename,
        "file_path": file_path_str,
        "actual_word_count": 0,
        "status": "pending",
        "sha256": None,
        "error_message": None,
    }

    # Safe rerun check: if file already exists and passes validation
    if output_file.exists() and not force:
        logger.info("[%s] Local file %s exists. Verifying word count...", target.ticker, target.local_filename)
        existing_text = output_file.read_text(encoding="utf-8")
        is_valid, w_count, msg = validate_document(existing_text, min_words=min_words)
        if is_valid:
            logger.info("[%s] Verified existing file (%d words). Skipping download.", target.ticker, w_count)
            manifest_entry["actual_word_count"] = w_count
            manifest_entry["status"] = "validated"
            manifest_entry["sha256"] = compute_sha256(existing_text)
            return manifest_entry
        else:
            logger.warning("[%s] Existing file failed validation (%s). Re-downloading...", target.ticker, msg)

    logger.info("[%s] Downloading from %s ...", target.ticker, target.url)
    try:
        response = session.get(target.url, timeout=30)
        if response.status_code != 200:
            err = f"HTTP {response.status_code}: {response.reason}"
            logger.error("[%s] Download failed: %s", target.ticker, err)
            manifest_entry["status"] = "failed_download"
            manifest_entry["error_message"] = err
            return manifest_entry

        raw_html = response.text
        if not raw_html or len(raw_html.strip()) == 0:
            err = "Received empty response body from SEC server."
            logger.error("[%s] %s", target.ticker, err)
            manifest_entry["status"] = "failed_empty_response"
            manifest_entry["error_message"] = err
            return manifest_entry

        logger.info("[%s] Successfully downloaded %d bytes. Parsing HTML & inline XBRL...", target.ticker, len(raw_html))
        cleaned_text = clean_sec_html(raw_html)

        is_valid, w_count, val_msg = validate_document(cleaned_text, min_words=min_words)
        manifest_entry["actual_word_count"] = w_count
        manifest_entry["sha256"] = compute_sha256(cleaned_text)

        if not is_valid:
            logger.error("[%s] Validation failed: %s", target.ticker, val_msg)
            manifest_entry["status"] = "failed_insufficient_length"
            manifest_entry["error_message"] = val_msg
            return manifest_entry

        # Save cleaned text to target path
        output_file.write_text(cleaned_text, encoding="utf-8")
        logger.info("[%s] Saved cleaned report to %s (%d words).", target.ticker, output_file.name, w_count)

        manifest_entry["status"] = "validated"
        return manifest_entry

    except requests.RequestException as exc:
        err = f"Network or request error: {exc}"
        logger.error("[%s] %s", target.ticker, err)
        manifest_entry["status"] = "failed_network_error"
        manifest_entry["error_message"] = str(exc)
        return manifest_entry
    except Exception as exc:
        err = f"Unexpected processing error: {exc}"
        logger.error("[%s] %s", target.ticker, err)
        manifest_entry["status"] = "failed_processing_error"
        manifest_entry["error_message"] = str(exc)
        return manifest_entry
    finally:
        # Rate-limiting courtesy delay
        time.sleep(delay)


def run_downloader(
    targets: list[FilingTarget],
    output_dir: Path,
    manifest_path: Path,
    min_words: int = 5000,
    force: bool = False,
    delay: float = 0.5,
    cli_email: Optional[str] = None,
) -> dict[str, Any]:
    """
    Executes sequential downloads for all specified filing targets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    user_agent = get_user_agent(cli_email)
    logger.info("Initializing SEC Session with User-Agent: %s", user_agent)
    session = create_session(user_agent)

    manifest_records: list[dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    logger.info("Starting processing for %d target filings...", len(targets))

    for idx, target in enumerate(targets, 1):
        logger.info("--- Processing [%d/%d]: %s (%s) ---", idx, len(targets), target.company, target.ticker)
        record = download_and_process_target(
            target=target,
            session=session,
            output_dir=output_dir,
            min_words=min_words,
            force=force,
            delay=delay,
        )
        manifest_records.append(record)
        if record["status"] == "validated":
            success_count += 1
        else:
            fail_count += 1

    manifest_data = {
        "corpus_name": "Long-Context SEC 10-K Corpus",
        "description": "Cleaned, structured SEC Form 10-K annual reports preserving financial tables and narrative disclosures.",
        "word_count_method": "Whitespace-delimited token splitting (len(text.split())) on clean UTF-8 text.",
        "min_words_requirement": min_words,
        "total_documents": len(targets),
        "validated_documents": success_count,
        "failed_documents": fail_count,
        "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "documents": manifest_records,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Manifest written to %s", manifest_path)
    logger.info("Download & validation summary: %d successful, %d failed.", success_count, fail_count)

    return manifest_data


def main():
    parser = argparse.ArgumentParser(
        description="Download and clean SEC Form 10-K filings for the Long-Context Retrieval Fidelity Engine."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-cleaning even if files already exist.",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Optional: filter by company name or ticker (case-insensitive) to process a single filing.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Polite inter-request delay in seconds to respect SEC rate limits (default: 0.5s).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5000,
        help="Minimum required word count threshold (default: 5000).",
    )
    parser.add_argument(
        "--contact-email",
        type=str,
        default=None,
        help="Contact email for SEC EDGAR User-Agent header (overrides SEC_CONTACT_EMAIL env var).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "documents",
        help="Directory where cleaned document text files are stored.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus_manifest.json",
        help="Path where corpus_manifest.json is saved.",
    )

    args = parser.parse_args()

    targets = FILING_TARGETS
    if args.company:
        query = args.company.strip().lower()
        targets = [t for t in targets if query in t.company.lower() or query == t.ticker.lower()]
        if not targets:
            logger.error("No filings found matching company filter '%s'.", args.company)
            sys.exit(1)

    result = run_downloader(
        targets=targets,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        min_words=args.min_words,
        force=args.force,
        delay=args.delay,
        cli_email=args.contact_email,
    )

    if result["failed_documents"] > 0:
        logger.warning("Corpus run completed with %d failed documents.", result["failed_documents"])
        sys.exit(1)
    else:
        logger.info("All %d corpus documents successfully processed and validated!", result["validated_documents"])


if __name__ == "__main__":
    main()
