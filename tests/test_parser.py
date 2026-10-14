"""
Unit tests for document parser, HTML cleanup, table preservation,
and word count validation.
"""

import pytest
from src.parser import clean_sec_html, count_words, validate_document


def test_count_words():
    assert count_words("") == 0
    assert count_words("   \n\t  ") == 0
    assert count_words("one two three") == 3
    assert count_words("First line.\nSecond line with five words.") == 7


def test_validate_document_success():
    sample_text = "word " * 5050
    is_valid, count, msg = validate_document(sample_text, min_words=5000)
    assert is_valid is True
    assert count == 5050
    assert "passed validation" in msg


def test_validate_document_too_short():
    sample_text = "word " * 4999
    is_valid, count, msg = validate_document(sample_text, min_words=5000)
    assert is_valid is False
    assert count == 4999
    assert "failing minimum requirement" in msg


def test_validate_document_empty():
    is_valid, count, msg = validate_document("", min_words=5000)
    assert is_valid is False
    assert count == 0
    assert "empty" in msg.lower()


def test_clean_sec_html_strips_scripts_and_styles():
    raw_html = """
    <html>
        <head>
            <style>body { color: red; }</style>
            <script>console.log("tracking code");</script>
        </head>
        <body>
            <p>Substantive disclosure content.</p>
            <noscript>JavaScript is required</noscript>
        </body>
    </html>
    """
    cleaned = clean_sec_html(raw_html)
    assert "color: red" not in cleaned
    assert "console.log" not in cleaned
    assert "JavaScript is required" not in cleaned
    assert "Substantive disclosure content." in cleaned


def test_clean_sec_html_strips_hidden_elements():
    raw_html = """
    <div>
        <p>Visible annual report disclosure.</p>
        <span style="display: none;">Internal metadata identifier</span>
        <div style="visibility:hidden">Hidden tracking element</div>
    </div>
    """
    cleaned = clean_sec_html(raw_html)
    assert "Visible annual report disclosure." in cleaned
    assert "Internal metadata identifier" not in cleaned
    assert "Hidden tracking element" not in cleaned


def test_clean_sec_html_preserves_tables():
    raw_html = """
    <html>
        <body>
            <h2>Consolidated Statements of Income</h2>
            <table border="1">
                <thead>
                    <tr><th>Segment</th><th>FY 2025</th><th>FY 2024</th></tr>
                </thead>
                <tbody>
                    <tr><td>Cloud Operations</td><td>$45,000</td><td>$33,000</td></tr>
                    <tr><td>Consumer Services</td><td>$120,000</td><td>$110,000</td></tr>
                </tbody>
            </table>
            <p>End of financial statement.</p>
        </body>
    </html>
    """
    cleaned = clean_sec_html(raw_html)
    assert "Consolidated Statements of Income" in cleaned
    # Check that table rows are structured with pipe delimiters
    assert "| Segment | FY 2025 | FY 2024 |" in cleaned
    assert "| Cloud Operations | $45,000 | $33,000 |" in cleaned
    assert "| Consumer Services | $120,000 | $110,000 |" in cleaned
    assert "End of financial statement." in cleaned


def test_clean_sec_html_handles_single_cell_layout_table():
    raw_html = """
    <table>
        <tr>
            <td>This is a single narrative paragraph wrapped in a layout table.</td>
        </tr>
    </table>
    """
    cleaned = clean_sec_html(raw_html)
    assert "This is a single narrative paragraph wrapped in a layout table." in cleaned
    # Single-cell layout table should not have unnecessary pipes
    assert "|" not in cleaned


def test_clean_sec_html_removes_standalone_table_of_contents_jump_links():
    raw_html = """
    <div>
        <a href="#toc">Table of Contents</a>
        <h1>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h1>
        <p>Revenues increased significantly across all segments.</p>
        <a href="#toc">Table of Contents</a>
    </div>
    """
    cleaned = clean_sec_html(raw_html)
    lines = [line.strip() for line in cleaned.splitlines()]
    assert "Table of Contents" not in lines
    assert "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS" in cleaned
    assert "Revenues increased significantly across all segments." in cleaned


def test_clean_sec_html_malformed_html():
    raw_html = "<p>Unclosed paragraph <b>bold text without closing tags <table><tr><td>Cell"
    cleaned = clean_sec_html(raw_html)
    assert "Unclosed paragraph" in cleaned
    assert "bold text without closing tags" in cleaned
    assert "Cell" in cleaned


def test_clean_sec_html_ixbrl_tags():
    raw_html = """
    <ix:header>
        <ix:hidden>Taxonomy Context Ref</ix:hidden>
    </ix:header>
    <ix:nonNumeric contextRef="FY2025" name="us-gaap:Revenue">
        Total Revenue was $150,000 million.
    </ix:nonNumeric>
    """
    cleaned = clean_sec_html(raw_html)
    assert "Taxonomy Context Ref" not in cleaned
    assert "Total Revenue was $150,000 million." in cleaned
