"""
Document parsing and HTML cleanup module for SEC Form 10-K filings.
Converts raw HTML/iXBRL documents into clean, structured UTF-8 plain text,
preserving document hierarchy, headings, and financial table structures.
"""

from __future__ import annotations

import logging
import re
from typing import Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)


def count_words(text: str) -> int:
    """
    Computes the canonical word count of a text document.

    Methodology:
    Splits the text on whitespace boundaries (standard Python split()),
    counting all non-whitespace tokens. This is the standard, documented
    metric for long-context text corpus length.

    Args:
        text: Plain text content.

    Returns:
        Integer count of whitespace-delimited words.
    """
    if not text:
        return 0
    return len(text.split())


def validate_document(text: str, min_words: int = 5000) -> Tuple[bool, int, str]:
    """
    Validates that extracted document text meets minimum length requirements.

    Args:
        text: Extracted plain text content.
        min_words: Minimum threshold for valid corpus documents (default: 5000).

    Returns:
        Tuple of (is_valid, word_count, message).
    """
    if not text or not text.strip():
        return False, 0, "Document is empty or contains only whitespace."

    w_count = count_words(text)
    if w_count < min_words:
        msg = f"Document contains {w_count} words, failing minimum requirement of {min_words} words."
        return False, w_count, msg

    return True, w_count, f"Document passed validation with {w_count} words (>= {min_words})."


def _format_table(table: Tag) -> str:
    """
    Converts an HTML table into structured, human-readable text.
    Preserves table rows and column alignments so financial statements
    remain distinct and unambiguous.

    Args:
        table: BeautifulSoup Tag representing <table>.

    Returns:
        Clean multiline string representing the table rows.
    """
    rows_text: list[str] = []
    trs = table.find_all("tr")

    if not trs:
        # Fallback for tables without tr elements
        plain = table.get_text(separator=" ", strip=True)
        return plain

    # Check if table is merely a 1-cell layout wrapper
    if len(trs) == 1:
        cells = trs[0].find_all(["th", "td"])
        if len(cells) == 1:
            return cells[0].get_text(separator=" ", strip=True)

    for tr in trs:
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue

        cell_strings: list[str] = []
        for cell in cells:
            # Normalize whitespace within each cell
            raw_cell = cell.get_text(separator=" ", strip=True)
            cleaned_cell = re.sub(r"\s+", " ", raw_cell)
            cell_strings.append(cleaned_cell)

        # Skip rows where all cells are empty
        if not any(cell_strings):
            continue

        # Format as pipe-delimited row
        formatted_row = "| " + " | ".join(cell_strings) + " |"
        rows_text.append(formatted_row)

    if not rows_text:
        return ""

    return "\n" + "\n".join(rows_text) + "\n"


def clean_sec_html(html_content: str) -> str:
    """
    Parses and cleans SEC EDGAR HTML and inline XBRL (iXBRL) documents into
    structured UTF-8 plain text.

    Preservation Rules:
    - Retains original document flow, section sequence, headings, and paragraphs.
    - Preserves financial table rows as distinguishable pipe-delimited lines.
    - Removes scripts, styles, metadata headers, and hidden presentation tags.
    - Strips obvious navigational noise while retaining all substantive disclosures.

    Args:
        html_content: Raw HTML/iXBRL text.

    Returns:
        Clean UTF-8 plain text.
    """
    if not html_content or not html_content.strip():
        logger.warning("Empty HTML content received.")
        return ""

    # Parse HTML using standard html.parser
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Strip non-content and metadata elements
    unwanted_tags = [
        "script",
        "style",
        "noscript",
        "meta",
        "link",
        "svg",
        "ix:header",
    ]
    for tag_name in unwanted_tags:
        for element in soup.find_all(tag_name):
            element.decompose()

    # 2. Strip elements explicitly marked hidden (style="display:none" or similar)
    hidden_pattern = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)
    for element in soup.find_all(style=hidden_pattern):
        element.decompose()

    # 3. Format and preserve table structures before general text extraction
    for table in soup.find_all("table"):
        table_text = _format_table(table)
        if table_text.strip():
            # Replace table element with formatted plain text representation
            table.replace_with(NavigableString(f"\n{table_text}\n"))
        else:
            table.decompose()

    # 4. Extract structured text preserving line breaks
    raw_text = soup.get_text(separator="\n")

    # 5. Clean up line formatting and excess whitespace
    lines = raw_text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        # Collapse multiple horizontal spaces and strip edge spaces
        norm_line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        # Remove standalone navigation artifacts (e.g. repeated "Table of Contents" jump links)
        if norm_line.lower() == "table of contents":
            continue
        cleaned_lines.append(norm_line)

    # Reassemble with clean paragraph breaks
    full_text = "\n".join(cleaned_lines)

    # Collapse 3 or more consecutive newlines into double newlines (paragraph boundary)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    return full_text.strip()
