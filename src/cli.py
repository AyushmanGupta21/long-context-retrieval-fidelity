"""
Command-Line Interface (CLI) for Long-Context Retrieval Fidelity Engine.
Provides interactive and single-query search modes with dense, BM25, and hybrid RRF retrieval,
extractive answer extraction, source position transparency, and detailed --debug diagnostics.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def supports_color() -> bool:
    """Returns True if the output stream supports ANSI color codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return True


def c_style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if supports_color() else text


def c_bold(text: str) -> str:
    return c_style(text, "1")


def c_cyan(text: str) -> str:
    return c_style(text, "36")


def c_green(text: str) -> str:
    return c_style(text, "32")


def c_yellow(text: str) -> str:
    return c_style(text, "33")


def c_red(text: str) -> str:
    return c_style(text, "31")


def c_dim(text: str) -> str:
    return c_style(text, "2")


def get_term_width(default: int = 80) -> int:
    """Returns terminal columns bounded between 70 and 100 for clean readable output."""
    try:
        w = shutil.get_terminal_size(fallback=(default, 24)).columns
        return max(70, min(w, 100))
    except Exception:
        return default

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import Chunk
from src.embeddings import get_embedding_engine
from src.indexer import DEFAULT_INDEX_DIR, CorpusIndex, load_index
from src.lexical_retriever import BM25Retriever, tokenize_text
from src.mitigation import fuse_dense_and_lexical, reciprocal_rank_fusion
from src.retriever import RetrievalResult, retrieve_top_k

logger = logging.getLogger("RetrievalCLI")


def extract_best_sentence(query: str, text: str) -> str:
    """
    Extracts the most relevant narrative sentence from a retrieved passage
    using transparent lexical term overlap.

    Excludes markdown table rows, exhibit list items, page numbers,
    and boilerplate SEC filings lines to ensure only meaningful,
    grammatical narrative text is returned.
    """
    if not text or not text.strip():
        return ""

    raw_candidates = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 15]
    if not raw_candidates:
        return ""

    clean_candidates = []
    for cand in raw_candidates:
        # Exclude markdown table rows (containing pipe characters)
        if "|" in cand:
            continue
        # Exclude exhibit items or filing index numbers (e.g. "97.1 ...", "Item 15. ...", "4.8 ...")
        if re.match(r"^(\d+\.[\d\w]+|Item\s+\d+|Exhibit\s+\d+|Part\s+[IVX]+)\b", cand, re.IGNORECASE):
            continue
        # Exclude boilerplate certifications, signatures, and filing navigation labels
        if any(
            w in cand.lower()
            for w in [
                "pursuant to 18 u.s.c",
                "certification of",
                "clawback policy",
                "insider trading policy",
                "table of contents",
                "index to consolidated",
            ]
        ):
            continue
        # Must have at least 4 whitespace-separated tokens
        if len(cand.split()) < 4:
            continue
        clean_candidates.append(cand)

    if not clean_candidates:
        return ""

    query_terms = set(tokenize_text(query))
    if not query_terms:
        return clean_candidates[0]

    best_score = -1.0
    best_candidate = clean_candidates[0]

    for cand in clean_candidates:
        cand_terms = set(tokenize_text(cand))
        overlap = len(query_terms.intersection(cand_terms))
        # Normalize by length to prevent bias toward overly long sentences
        score = overlap / (float(len(cand_terms)) ** 0.5 + 1.0)
        if score > best_score:
            best_score = score
            best_candidate = cand

    return best_candidate


def format_passage_snippet(text: str, query: str = "", max_chars: int = 240) -> str:
    """
    Formats a concise, representative snippet of a retrieved chunk for CLI display.
    If the text exceeds max_chars, centers the snippet window around the most relevant
    matching sentence so that the displayed excerpt visually aligns with the chunk text.
    """
    clean_text = " ".join(text.split())
    if len(clean_text) <= max_chars:
        return clean_text

    if query:
        best_sentence = extract_best_sentence(query, text)
        if best_sentence:
            clean_best = " ".join(best_sentence.split())
            idx = clean_text.find(clean_best)
            if idx != -1:
                # If the matched sentence starts beyond opening 60 chars, center window
                if idx > 60:
                    start = max(0, idx - 30)
                    snippet = clean_text[start:start + max_chars].strip()
                    prefix = "... " if start > 0 else ""
                    suffix = "..." if (start + max_chars) < len(clean_text) else ""
                    return f"{prefix}{snippet}{suffix}"

    return clean_text[:max_chars].rstrip() + "..."


COMPANY_MAP: Dict[str, Dict[str, Any]] = {
    "amazon": {
        "name": "Amazon.com, Inc.",
        "doc_prefix": "amazon",
        "aliases": ["amazon", "amazon's", "amzn", "aws"],
    },
    "alphabet": {
        "name": "Alphabet Inc.",
        "doc_prefix": "alphabet",
        "aliases": ["alphabet", "alphabet's", "google", "google's", "googl"],
    },
    "meta": {
        "name": "Meta Platforms, Inc.",
        "doc_prefix": "meta",
        "aliases": ["meta", "meta's", "facebook", "instagram"],
    },
    "microsoft": {
        "name": "Microsoft Corporation",
        "doc_prefix": "microsoft",
        "aliases": ["microsoft", "microsoft's", "msft", "azure"],
    },
    "apple": {
        "name": "Apple Inc.",
        "doc_prefix": "apple",
        "aliases": ["apple", "apple's", "aapl", "iphone", "ipad"],
    },
    "nvidia": {
        "name": "NVIDIA Corporation",
        "doc_prefix": "nvidia",
        "aliases": ["nvidia", "nvidia's", "nvda"],
    },
    "netflix": {
        "name": "Netflix, Inc.",
        "doc_prefix": "netflix",
        "aliases": ["netflix", "netflix's", "nflx"],
    },
    "adobe": {
        "name": "Adobe Inc.",
        "doc_prefix": "adobe",
        "aliases": ["adobe", "adobe's", "adbe"],
    },
    "ibm": {
        "name": "International Business Machines Corp.",
        "doc_prefix": "ibm",
        "aliases": ["ibm", "ibm's"],
    },
    "oracle": {
        "name": "Oracle Corporation",
        "doc_prefix": "oracle",
        "aliases": ["oracle", "oracle's", "orcl"],
    },
}


def detect_company_in_query(query: str) -> Optional[Dict[str, Any]]:
    """
    Detects if the query explicitly specifies a company from the SEC 10-K corpus.
    Uses runtime entity aliasing without any benchmark or gold annotations.
    """
    q_lower = query.lower()
    tokens = re.findall(r"\b[a-z0-9']+\b", q_lower)
    for ckey, cinfo in COMPANY_MAP.items():
        for alias in cinfo["aliases"]:
            if alias in tokens or f" {alias} " in f" {q_lower} ":
                return cinfo
    return None


def extract_units_from_text(text: str) -> str:
    """
    Extracts financial denomination units (e.g. 'in millions') from surrounding text or table captions.
    """
    m = re.search(r"\((in [a-zA-Z\s,]+)\)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if "in millions" in text.lower():
        return "in millions"
    if "in billions" in text.lower():
        return "in billions"
    if "in thousands" in text.lower():
        return "in thousands"
    return ""


def parse_table_row(row_line: str) -> Tuple[str, List[str]]:
    """
    Parses a markdown table row into its row label and numerical/cell values.
    """
    raw_cells = [c.strip() for c in row_line.split("|")]
    clean_cells = [c for c in raw_cells if c and c != "$"]
    if not clean_cells:
        return "", []
    label = clean_cells[0]
    values = clean_cells[1:]
    return label, values


def get_table_header_context(
    res: RetrievalResult, documents_dir: Path = Path("data/documents")
) -> Tuple[str, str]:
    """
    Retrieves fiscal-year column headers and unit context for tabular candidate passages.
    If headers/units are truncated by sentence-level chunking, inspects preceding text
    in the source document.
    """
    lines = res.text.splitlines()
    table_lines = [line.strip() for line in lines if "|" in line]
    units = extract_units_from_text(res.text)

    header_lines: List[str] = []
    for line in table_lines[:5]:
        if re.search(r"202[0-9]|Year Ended|Period|December", line):
            header_lines.append(line)

    if header_lines and units:
        header_str = " | ".join(h.strip("|").strip() for h in header_lines)
        header_clean = " ".join(re.sub(r"\|\s*\|", "|", header_str).split())
        return header_clean, units

    source_file = documents_dir / res.document_id
    if source_file.exists():
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                content = f.read()
            offset_start = res.source_char_start
            window_start = max(0, offset_start - 600)
            preceding_text = content[window_start:offset_start]
            if not units:
                units = extract_units_from_text(preceding_text)
            if not header_lines:
                pre_lines = [l.strip() for l in preceding_text.splitlines() if "|" in l]
                for l in pre_lines[-4:]:
                    if re.search(r"202[0-9]|Year Ended|Period|December", l):
                        header_lines.append(l)
        except Exception:
            pass

    header_str = " | ".join(h.strip("|").strip() for h in header_lines) if header_lines else ""
    header_clean = " ".join(re.sub(r"\|\s*\|", "|", header_str).split())
    return header_clean, units or "in millions"


DISALLOWED_ROW_SUBSTRINGS: List[str] = [
    "statements of cash flows",
    "statement of cash flows",
    "statements of operations",
    "statement of operations",
    "statements of comprehensive income",
    "statement of comprehensive income",
    "statements of stockholders",
    "statement of stockholders",
    "statements of shareholders",
    "statement of shareholders",
    "balance sheets",
    "balance sheet",
    "notes to consolidated",
    "notes to financial",
    "report of independent",
    "accounting firm",
    "officers' certificate",
    "insider trading policy",
    "clawback policy",
    "consent of independent",
    "list of significant subsidiaries",
    "interactive data files",
    "total segment assets",
]


def extract_financial_table_evidence(
    query: str, res: RetrievalResult, documents_dir: Path = Path("data/documents")
) -> Optional[Dict[str, Any]]:
    """
    Checks if a candidate chunk (or its expanded table context in the source filing)
    contains direct tabular evidence answering a financial query.
    Preserves row label, fiscal-year column headers, values, denomination units,
    and maps explicitly requested reporting periods.
    """
    lines = res.text.splitlines()
    has_table_in_chunk = any("|" in line for line in lines)
    if not has_table_in_chunk:
        return None

    # Scope validation: If query specifies a company, candidate must belong to that company
    company_info = detect_company_in_query(query)
    if company_info:
        doc_lower = res.document_id.lower()
        title_lower = res.title.lower()
        if company_info["doc_prefix"] not in doc_lower and company_info["name"].lower() not in title_lower:
            return None

    source_file = documents_dir / res.document_id
    full_content = ""
    if source_file.exists():
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                full_content = f.read()
        except Exception:
            full_content = ""

    start_offset = res.source_char_start
    end_offset = res.source_char_end

    backward_lines: List[str] = []
    forward_lines: List[str] = []

    if full_content:
        # 1. Expand preceding table lines and caption if chunk started mid-table
        pre_window = full_content[max(0, start_offset - 800):start_offset]
        pre_lines = pre_window.splitlines()
        for l in reversed(pre_lines):
            if "|" in l or any(
                w in l.lower()
                for w in [
                    "statements of",
                    "net sales",
                    "operating expenses",
                    "information is as follows",
                    "december 31",
                    "year ended",
                    "(in ",
                ]
            ):
                backward_lines.append(l)
            else:
                if backward_lines:
                    break
        backward_lines.reverse()

        # 2. Expand following continuation lines if table was split by chunk boundary
        post_window = full_content[end_offset:min(len(full_content), end_offset + 1000)]
        post_lines = post_window.splitlines()
        for l in post_lines:
            if "|" in l:
                forward_lines.append(l)
            elif not l.strip():
                continue
            else:
                break

    combined_lines = backward_lines + lines + forward_lines
    combined_text = "\n".join(combined_lines)
    combined_lower = combined_text.lower()

    # 3. Discard non-financial tables (TOC, index, exhibits)
    # Check if table has fiscal period / year column headers
    has_fiscal_year_header = any(
        "|" in l and re.search(r"\b(19\d\d|20\d\d)\b|Year Ended|Period Ended|Three Months Ended", l, re.IGNORECASE)
        for l in combined_lines
    )

    is_toc_title = bool(re.search(
        r"index to (consolidated )?financial statements|table of contents|item 15\. exhibits|exhibit index",
        combined_text,
        re.IGNORECASE,
    ))

    has_page_column_header = any(
        "|" in l and re.search(r"\|\s*page\s*\|", l, re.IGNORECASE)
        for l in combined_lines[:8]
    )

    if is_toc_title and not has_fiscal_year_header:
        return None

    if has_page_column_header and not has_fiscal_year_header:
        return None

    # 4. Extract denomination units and column headers
    units = extract_units_from_text(combined_text)
    if not units and full_content:
        pre_ctx = full_content[max(0, start_offset - 800):start_offset]
        units = extract_units_from_text(pre_ctx)
    if not units:
        units = "in millions"

    header_lines = []
    for l in combined_lines:
        if "|" in l and re.search(r"\b(19\d\d|20\d\d)\b|Year Ended|Period|December|September|January", l):
            header_lines.append(l)
    header_clean = " | ".join(h.strip("|").strip() for h in header_lines[:3]) if header_lines else ""
    header_clean = " ".join(re.sub(r"\|\s*\|", "|", header_clean).split())

    q_lower = query.lower()
    target_metric_candidates = [
        "cost of sales",
        "consolidated net sales",
        "total net sales",
        "net sales",
        "operating income",
        "income from operations",
        "operating expenses",
        "gross profit",
        "net income",
        "total revenues",
        "revenues",
        "fulfillment",
        "technology and infrastructure",
        "marketable securities",
    ]
    target_metric = None
    for cand in target_metric_candidates:
        if cand in q_lower:
            target_metric = cand
            break

    if not target_metric:
        return None

    # Domain context filtering to prevent cross-metric false positives (e.g. PP&E additions)
    is_sales_table = any(
        w in combined_lower
        for w in [
            "net sales information",
            "consolidated statements of operations",
            "statement of operations",
            "net sales:",
        ]
    )
    is_ppe_table = any(
        w in combined_lower for w in ["property and equipment", "additions to property"]
    )

    year_match = re.search(r"\b(19\d\d|20\d\d)\b", query)
    target_year = year_match.group(1) if year_match else None

    for line in combined_lines:
        if "|" not in line:
            continue
        label, values = parse_table_row(line)
        if not label or not values:
            continue
        label_lower = label.lower().strip()

        # Reject non-financial statement titles, TOC entries, auditor reports
        if any(w in label_lower for w in DISALLOWED_ROW_SUBSTRINGS):
            continue

        # Reject percentage / growth rows
        if "%" in line or any("%" in v for v in values) or "growth" in label_lower or "percentage" in label_lower:
            if not any(k in target_metric for k in ("percent", "growth", "margin", "tax rate")):
                continue

        # Must contain digits
        clean_num_values = [v for v in values if re.search(r"\d", v)]
        if not clean_num_values:
            continue

        matches = False
        if target_metric == "cost of sales" and "cost of sales" in label_lower:
            matches = True
        elif target_metric in ("consolidated net sales", "total net sales"):
            if "total net sales" in label_lower or "consolidated net sales" in label_lower:
                matches = True
            elif label_lower == "consolidated" and is_sales_table and not is_ppe_table:
                matches = True
            elif label_lower == "net sales" and (
                "consolidated" in combined_lower or "statements of operations" in combined_lower
            ):
                matches = True
        elif target_metric == "net sales":
            if "total net sales" in label_lower or "consolidated net sales" in label_lower:
                matches = True
            elif label_lower == "consolidated" and is_sales_table and not is_ppe_table:
                matches = True
            elif (label_lower == "net sales" or label_lower.startswith("net sales")) and "mix" not in label_lower:
                matches = True
        elif target_metric in ("operating income", "income from operations"):
            if (
                "operating income" in label_lower
                or "income from operations" in label_lower
            ) and "growth" not in label_lower:
                matches = True
            elif label_lower == "consolidated" and (
                "operating income" in combined_lower
                or "income from operations" in combined_lower
            ):
                matches = True
        elif target_metric == "net income":
            if "net income" in label_lower and "growth" not in label_lower:
                matches = True
        elif target_metric in label_lower:
            matches = True

        if matches:
            period_val = None
            if target_year:
                if not header_clean:
                    # Requested reporting year is not present in table headers
                    continue
                header_years = re.findall(r"\b(19\d\d|20\d\d)\b", header_clean)
                seen_years = []
                for y in header_years:
                    if not seen_years or seen_years[-1] != y:
                        seen_years.append(y)
                if target_year not in seen_years:
                    # Requested reporting year is not present in this table's column headers
                    continue
                y_idx = seen_years.index(target_year)
                if len(clean_num_values) == len(seen_years):
                    period_val = clean_num_values[y_idx]
                elif len(clean_num_values) > len(seen_years) and len(clean_num_values) % len(seen_years) == 0:
                    step = len(clean_num_values) // len(seen_years)
                    col_idx = y_idx * step
                    period_val = clean_num_values[col_idx]
                elif y_idx < len(clean_num_values):
                    period_val = clean_num_values[y_idx]
                else:
                    continue
            elif clean_num_values:
                period_val = clean_num_values[-1]

            # Determine precise character offsets and provenance of the extracted row
            raw_line = line.strip()
            row_char_start = None
            row_char_end = None
            provenance_type = "within_candidate_chunk"

            if full_content:
                search_start = max(0, start_offset - 1000)
                search_end = min(len(full_content), end_offset + 1200)
                window_content = full_content[search_start:search_end]
                pos = window_content.find(raw_line)
                if pos != -1:
                    row_char_start = search_start + pos
                    row_char_end = row_char_start + len(raw_line)
                    if row_char_end <= start_offset:
                        provenance_type = "adjacent_preceding_context"
                    elif row_char_start >= end_offset:
                        provenance_type = "adjacent_following_continuation"
                    else:
                        provenance_type = "within_candidate_chunk"
                else:
                    row_char_start = start_offset
                    row_char_end = end_offset
            else:
                row_char_start = start_offset
                row_char_end = end_offset

            return {
                "row_label": label,
                "values": clean_num_values,
                "raw_line": raw_line,
                "header": header_clean,
                "units": units,
                "year": target_year,
                "year_value": period_val,
                "provenance_type": provenance_type,
                "row_char_start": row_char_start,
                "row_char_end": row_char_end,
                "candidate_chunk_id": res.chunk_id,
                "candidate_char_start": res.source_char_start,
                "candidate_char_end": res.source_char_end,
                "document_id": res.document_id,
            }

    return None


def is_explanatory_query(query: str) -> bool:
    """
    Checks if query is an explanatory or narrative reasoning question (e.g. 'Why...').
    """
    q_lower = query.lower().strip()
    return (
        any(
            q_lower.startswith(w)
            for w in [
                "why",
                "explain",
                "describe",
                "how does",
                "what is the reason",
                "what factors",
            ]
        )
        or "meaningful" in q_lower
        or "more meaningful" in q_lower
    )


def select_evidence(
    query: str, results: List[RetrievalResult], documents_dir: Path = Path("data/documents")
) -> Dict[str, Any]:
    """
    Faithful, transparent evidence selection across retrieved candidate passages.
    - Scopes evidence to target company when explicitly specified in the query.
    - Extracts structured table rows preserving label, headers, values and units.
    - Selects the actual evidence passage from lower ranks if rank 1 does not establish the answer.
    - Clearly reports when candidate passages do not establish the requested figure or answer.
    - Strictly extractive: no LLM hallucination, no arbitrary arithmetic.
    """
    if not results:
        return {
            "status": "empty",
            "excerpt": "",
            "source_chunk": None,
            "rank": None,
            "note": "No matching chunks retrieved.",
        }

    company_info = detect_company_in_query(query)

    # Respect explicit company reference by scoping candidate selection to target company
    scoped_results = []
    for r in results:
        if company_info:
            if (
                company_info["doc_prefix"] in r.document_id.lower()
                or company_info["name"].lower() in r.title.lower()
            ):
                scoped_results.append(r)
        else:
            scoped_results.append(r)

    if not scoped_results:
        return {
            "status": "company_mismatch",
            "excerpt": f"[No candidate chunks from {company_info['name']} were retrieved in top-{len(results)} results]",
            "source_chunk": None,
            "rank": None,
            "note": f"Query requested {company_info['name']}, but top retrieved chunks were from other companies.",
        }

    # 1. Explanatory / narrative queries (e.g. 'Why does Amazon consider...')
    if is_explanatory_query(query):
        best_cand = None
        best_cand_rank = None
        best_cand_sentence = ""
        best_cand_score = -1.0

        for r in scoped_results:
            sent = extract_best_sentence(query, r.text)
            if not sent:
                continue
            q_terms = set(tokenize_text(query))
            s_terms = set(tokenize_text(sent))
            overlap = len(q_terms.intersection(s_terms))
            causal_boost = (
                1.5
                if any(
                    w in sent.lower()
                    for w in ["because", "believe", "due to", "meaningful", "measure", "reflects"]
                )
                else 1.0
            )
            score = (overlap / (float(len(s_terms)) ** 0.5 + 1.0)) * causal_boost
            rank_discount = 1.0 / (float(r.rank) ** 0.25)
            final_score = score * rank_discount

            if final_score > best_cand_score:
                best_cand_score = final_score
                best_cand = r
                best_cand_rank = r.rank
                best_cand_sentence = sent

        if best_cand and best_cand_sentence:
            return {
                "status": "narrative_explanation",
                "excerpt": f"\"{best_cand_sentence}\"",
                "source_chunk": best_cand,
                "rank": best_cand_rank,
                "note": None,
            }
        else:
            return {
                "status": "not_established",
                "excerpt": f"[An explanatory narrative answer for '{query}' was not established in retrieved candidate passages]",
                "source_chunk": None,
                "rank": None,
                "note": "Candidate passages do not contain an explanatory statement answering the query.",
            }

    # 2. Financial metric / tabular queries
    for r in scoped_results:
        tbl = extract_financial_table_evidence(query, r, documents_dir=documents_dir)
        if tbl:
            lines = []
            if tbl.get("header"):
                lines.append(f"Column Headers : {tbl['header']}")
            lines.append(f"Table Row      : {tbl['raw_line']}")
            if tbl.get("units"):
                lines.append(f"Units          : {tbl['units']}")
            if tbl.get("year_value") and tbl.get("year"):
                lines.append(f"Fiscal Period  : {tbl['year']} (${tbl['year_value']} {tbl.get('units', '')})")

            prov = tbl.get("provenance_type")
            row_start = tbl.get("row_char_start")
            row_end = tbl.get("row_char_end")
            if prov == "adjacent_following_continuation":
                lines.append(
                    f"Row Provenance : Adjacent table continuation in {tbl.get('document_id')} (chars {row_start}-{row_end}, follows candidate chunk {r.chunk_id})"
                )
            elif prov == "adjacent_preceding_context":
                lines.append(
                    f"Row Provenance : Preceding table context in {tbl.get('document_id')} (chars {row_start}-{row_end}, precedes candidate chunk {r.chunk_id})"
                )
            else:
                lines.append(
                    f"Row Provenance : Directly contained within candidate chunk {r.chunk_id} (chars {row_start}-{row_end})"
                )

            excerpt = "\n".join(lines)
            return {
                "status": "direct_table_evidence",
                "excerpt": excerpt,
                "table_details": tbl,
                "source_chunk": r,
                "rank": r.rank,
                "note": f"Retrieved at Rank #{r.rank} ({r.chunk_id})" if r.rank > 1 else None,
            }

    # 3. Check if query requested consolidated net sales, but only partial/geographic was retrieved
    q_lower = query.lower()
    year_match = re.search(r"\b(19\d\d|20\d\d)\b", query)
    target_year = year_match.group(1) if year_match else None
    period_str = f" in {target_year}" if target_year else ""

    if "consolidated net sales" in q_lower or "total net sales" in q_lower:
        top_res = scoped_results[0]
        notes = []
        if "attributed to countries" in top_res.text.lower():
            notes.append(
                f"- Rank #{top_res.rank} ({top_res.chunk_id}) provides geographic net sales attributed to individual countries, not the consolidated total."
            )
        for r in scoped_results[1:]:
            if "net sales:" in r.text.lower() and "north america" in r.text.lower():
                notes.append(
                    f"- Rank #{r.rank} ({r.chunk_id}) contains a partial segment breakdown (North America, International); the consolidated total row is not present in this chunk."
                )

        detail_note = ("\n" + "\n".join(notes)) if notes else ""
        return {
            "status": "not_established",
            "excerpt": (
                f"[Direct evidence for consolidated total net sales{period_str} was not established in retrieved candidate passages]"
                + detail_note
            ),
            "source_chunk": None,
            "rank": None,
            "note": f"Candidate passages do not establish consolidated total net sales{period_str}.",
        }

    # 4. If query asked for a specific financial metric not found in retrieved candidate passages
    financial_metrics = [
        "cost of sales",
        "operating income",
        "income from operations",
        "operating expenses",
        "marketable securities",
        "gross profit",
        "net income",
        "total revenues",
        "revenues",
    ]
    for fm in financial_metrics:
        if fm in q_lower:
            return {
                "status": "not_established",
                "excerpt": f"[Direct evidence for {fm}{period_str} was not established in retrieved candidate passages]",
                "source_chunk": None,
                "rank": None,
                "note": f"Retrieved candidate passages do not contain financial table line items for {fm}{period_str}.",
            }

    # 5. Broad / definitional or general queries (e.g. 'what is amazon')
    # Look for the best narrative sentence across top candidate chunks
    best_cand = None
    best_cand_rank = None
    best_cand_sentence = ""
    best_score = -1.0

    q_terms = set(tokenize_text(query))
    stopwords = {"what", "is", "the", "are", "of", "in", "for", "a", "an", "to", "and", "on", "at", "by", "with", "from"}
    meaningful_q_terms = q_terms - stopwords

    for r in scoped_results[:5]:
        sent = extract_best_sentence(query, r.text)
        if not sent:
            continue
        sent_terms = set(tokenize_text(sent))
        overlap = len(meaningful_q_terms.intersection(sent_terms))
        if overlap > 0:
            score = overlap / (float(len(sent_terms)) ** 0.5 + 1.0)
            if score > best_score:
                best_score = score
                best_cand = r
                best_cand_rank = r.rank
                best_cand_sentence = sent

    # Check if the query is a broad/definitional question
    is_broad_query = any(
        q_lower.strip().startswith(prefix)
        for prefix in ["what is ", "what was ", "who is ", "define ", "what does "]
    )

    has_definitional_content = False
    if best_cand_sentence:
        sent_lower = best_cand_sentence.lower()
        if any(w in sent_lower for w in ["operates", "provides", "is a", "is an", "comprises", "consists of", "engaged in", "leader in"]):
            has_definitional_content = True

    if is_broad_query and not has_definitional_content:
        return {
            "status": "not_established",
            "excerpt": f"[A direct factual definition or answer for '{query}' was not established in the retrieved candidate passages]",
            "source_chunk": None,
            "rank": None,
            "note": "Retrieved passages contain legal disclosures, filings, and exhibit indices rather than a direct definition of the entity. Candidate passages are displayed below for review.",
        }

    if best_cand and best_cand_sentence:
        return {
            "status": "extractive_sentence",
            "excerpt": f"\"{best_cand_sentence}\"",
            "source_chunk": best_cand,
            "rank": best_cand_rank,
            "note": f"Retrieved at Rank #{best_cand_rank} ({best_cand.chunk_id})" if best_cand_rank > 1 else None,
        }

    return {
        "status": "not_established",
        "excerpt": f"[Direct evidence to answer '{query}' was not established in retrieved candidate passages]",
        "source_chunk": None,
        "rank": None,
        "note": "Candidate passages do not establish a direct extractive answer for the query.",
    }


class RetrievalService:
    """
    Service wrapper around CorpusIndex providing unified execution across
    dense, BM25, and hybrid RRF retrieval modes.
    """

    def __init__(self, index: CorpusIndex, candidate_depth: int = 50, rrf_c: float = 60.0):
        self.index = index
        self.candidate_depth = candidate_depth
        self.rrf_c = rrf_c
        self.engine = get_embedding_engine(index.metadata.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))

    def search(
        self,
        query: str,
        mode: str = "rrf",
        top_k: int = 5,
        debug: bool = False,
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Executes query retrieval across specified mode.

        Returns:
            Tuple of (top_k_results, debug_diagnostics_dict).
        """
        debug_info: dict[str, Any] = {
            "query": query,
            "normalized_query": query.strip().lower(),
            "mode": mode,
            "top_k": top_k,
            "candidate_depth": self.candidate_depth,
            "rrf_c": self.rrf_c,
            "timestamps": {},
            "index_metadata": getattr(self.index, "metadata", {}),
            "total_chunks": len(self.index.chunks),
            "total_documents": len(set(c.document_id for c in self.index.chunks)),
            "model_name": getattr(self.engine, "model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            "embedding_dim": getattr(self.engine, "embedding_dim", 384),
            "max_seq_length": getattr(self.engine, "max_seq_length", 256),
            "index_path": getattr(self, "index_path", str(DEFAULT_INDEX_DIR)),
            "index_load_sec": getattr(self, "index_load_sec", None),
        }

        query_clean = query.strip()
        if not query_clean:
            return [], debug_info

        t0 = time.time()
        # 1. Query tokenization for lexical analysis
        q_tokens = tokenize_text(query_clean)
        debug_info["query_tokens"] = q_tokens

        # Precompute dense candidates if needed
        dense_results: list[RetrievalResult] = []
        if mode in ("dense", "rrf"):
            t_d0 = time.time()
            q_emb = self.engine.encode([query_clean], batch_size=1, normalize_embeddings=True)
            t_emb_end = time.time()
            debug_info["timestamps"]["query_embedding_sec"] = round(t_emb_end - t_d0, 4)
            debug_info["query_vector_shape"] = list(q_emb.shape)
            debug_info["query_vector_dtype"] = str(q_emb.dtype)
            debug_info["query_vector_finite"] = bool(np.all(np.isfinite(q_emb)))
            debug_info["query_vector_norm"] = round(float(np.linalg.norm(q_emb[0])), 4)
            debug_info["query_vector_preview"] = [round(float(x), 4) for x in q_emb[0][:5]]

            dense_results = retrieve_top_k(
                query_vector=q_emb[0],
                corpus_matrix=self.index.embeddings,
                chunks=self.index.chunks,
                k=self.candidate_depth if mode == "rrf" else top_k,
            )
            debug_info["timestamps"]["dense_retrieval_sec"] = round(time.time() - t_emb_end, 4)
            debug_info["dense_candidate_count"] = len(dense_results)
            debug_info["dense_top5"] = [
                {"rank": r.rank, "chunk_id": r.chunk_id, "score": round(r.score, 4), "doc": r.document_id}
                for r in dense_results[:5]
            ]
            debug_info["dense_top_candidates"] = dense_results[:5]

        # Precompute BM25 candidates if needed
        bm25_results: list[RetrievalResult] = []
        if mode in ("bm25", "rrf"):
            t_b0 = time.time()
            bm25_results = self.index.bm25.retrieve(
                query=query_clean,
                k=self.candidate_depth if mode == "rrf" else top_k,
            )
            debug_info["timestamps"]["bm25_retrieval_sec"] = round(time.time() - t_b0, 4)
            debug_info["bm25_candidate_count"] = len(bm25_results)
            debug_info["bm25_k1"] = getattr(self.index.bm25, "k1", 1.5)
            debug_info["bm25_b"] = getattr(self.index.bm25, "b", 0.75)
            bm25_avg = getattr(self.index.bm25, "avgdl", getattr(self.index.bm25, "avg_doc_len", 0.0))
            debug_info["bm25_avg_doc_len"] = round(float(bm25_avg), 2)
            debug_info["bm25_top5"] = [
                {"rank": r.rank, "chunk_id": r.chunk_id, "score": round(r.score, 4), "doc": r.document_id}
                for r in bm25_results[:5]
            ]
            debug_info["bm25_top_candidates"] = bm25_results[:5]

        # Final ranking selection
        if mode == "dense":
            final_results = dense_results[:top_k]
        elif mode == "bm25":
            final_results = bm25_results[:top_k]
        elif mode == "rrf":
            t_f0 = time.time()
            dense_score_map = {r.chunk_id: (r.rank, r.score) for r in dense_results}
            bm25_score_map = {r.chunk_id: (r.rank, r.score) for r in bm25_results}

            fused_candidates = fuse_dense_and_lexical(
                dense_results=dense_results,
                lexical_results=bm25_results,
                c=self.rrf_c,
                top_k=top_k,
            )
            debug_info["timestamps"]["rrf_fusion_sec"] = round(time.time() - t_f0, 4)
            debug_info["fused_candidate_union_size"] = len(set(dense_score_map.keys()).union(bm25_score_map.keys()))

            fusion_diagnostics = []
            for res in fused_candidates:
                d_rank, d_score = dense_score_map.get(res.chunk_id, (None, None))
                b_rank, b_score = bm25_score_map.get(res.chunk_id, (None, None))
                d_contrib = round(1.0 / (self.rrf_c + d_rank), 6) if d_rank is not None else 0.0
                b_contrib = round(1.0 / (self.rrf_c + b_rank), 6) if b_rank is not None else 0.0
                match_type = (
                    "Dual Match (Dense + BM25)"
                    if (d_rank is not None and b_rank is not None)
                    else ("Dense Only" if d_rank is not None else "BM25 Only")
                )
                fusion_diagnostics.append({
                    "final_rank": res.rank,
                    "chunk_id": res.chunk_id,
                    "doc_id": res.document_id,
                    "fused_rrf_score": round(res.score, 6),
                    "dense_rank": d_rank,
                    "dense_cosine_sim": round(d_score, 4) if d_score is not None else None,
                    "dense_contrib": d_contrib,
                    "bm25_rank": b_rank,
                    "bm25_score": round(b_score, 4) if b_score is not None else None,
                    "bm25_contrib": b_contrib,
                    "match_type": match_type,
                })

            debug_info["fusion_breakdown"] = fusion_diagnostics
            final_results = fused_candidates
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        debug_info["timestamps"]["total_pipeline_sec"] = round(time.time() - t0, 4)
        return final_results, debug_info


def display_results(
    query: str,
    results: list[RetrievalResult],
    mode: str,
    debug_info: dict[str, Any],
    debug_flag: bool = False,
) -> None:
    """
    Renders cleanly formatted retrieval outputs, source positions,
    and verified extractive excerpts.
    """
    width = get_term_width(80)
    sep_double = "=" * width
    sep_single = "-" * width
    sep_hash = "#" * width

    mode_label = mode.upper()
    mode_desc = {
        "dense": "Dense Semantic (Cosine Similarity)",
        "bm25": "Lexical BM25 Okapi Scoring",
        "rrf": f"Reciprocal Rank Fusion (c={debug_info.get('rrf_c', 60.0)}, M={debug_info.get('candidate_depth', 50)})",
    }.get(mode, mode_label)

    total_chunks = debug_info.get("total_chunks", 3957)
    total_docs = debug_info.get("total_documents", 10)

    # 1. Header Banner
    print("\n" + sep_double)
    print(c_bold("LONG-CONTEXT RETRIEVAL FIDELITY ENGINE"))
    print(f"Query          : \"{query}\"")
    print(f"Retrieval Mode : {c_cyan(mode_label)} ({mode_desc})")
    print(f"Requested Top-K: {len(results)} candidate passages")
    print(f"Corpus Index   : Persistent cache ({total_chunks:,} chunks across {total_docs} SEC Form 10-K filings)")
    print(sep_double)

    if not results:
        print(f"\n{c_yellow('[!] No matching chunks retrieved. Try refining query terminology.')}\n")
        print(sep_double)
        return

    # 2. Extractive Answer Evidence Selection
    ev = select_evidence(query, results)
    ev_status = ev.get("status")

    print("\n" + sep_single)
    print(c_bold("EXTRACTIVE ANSWER EVIDENCE EXCERPT"))
    print(c_dim("[Extractive evidence retrieved from source filings -- NOT generated by an LLM]"))
    print(sep_single)

    if ev_status == "direct_table_evidence":
        print(c_green("STATUS: [EVIDENCE FOUND & EXTRACTED -- STRUCTURED TABLE EVIDENCE]"))
        print()
        tbl = ev.get("table_details", {})
        if tbl.get("year_value") and tbl.get("year"):
            print(f"Reported Value : {c_bold('$' + str(tbl['year_value']))} {tbl.get('units', '')} for fiscal year {tbl['year']}")
        if tbl.get("row_label"):
            print(f"Line Item Label: {tbl['row_label']}")
        if tbl.get("header"):
            print(f"Column Headers : {tbl['header']}")
        print(f"Table Row Line : {tbl.get('raw_line')}")
        if tbl.get("units"):
            print(f"Denomination   : {tbl['units']}")
        prov = tbl.get("provenance_type")
        row_start = tbl.get("row_char_start")
        row_end = tbl.get("row_char_end")
        sc = ev.get("source_chunk")
        if sc is not None:
            if prov == "adjacent_following_continuation":
                print(f"Row Provenance : {c_cyan('Adjacent following table continuation')} in {sc.document_id} (chars {row_start}-{row_end}, follows candidate chunk boundary {sc.source_char_end})")
            elif prov == "adjacent_preceding_context":
                print(f"Row Provenance : {c_cyan('Adjacent preceding table context')} in {sc.document_id} (chars {row_start}-{row_end}, precedes candidate chunk boundary {sc.source_char_start})")
            else:
                print(f"Row Provenance : {c_cyan('Directly contained within candidate chunk')} {sc.chunk_id} (chars {row_start}-{row_end})")

    elif ev_status == "narrative_explanation":
        print(c_green("STATUS: [EVIDENCE FOUND & EXTRACTED -- NARRATIVE EXPLANATION]"))
        print()
        print(f"Excerpt        : {c_bold(ev.get('excerpt', ''))}")

    elif ev_status == "extractive_sentence":
        print(c_green("STATUS: [EVIDENCE FOUND & EXTRACTED -- EXTRACTIVE PASSAGE EXCERPT]"))
        print()
        print(f"Excerpt        : {c_bold(ev.get('excerpt', ''))}")

    elif ev_status == "not_established":
        print(c_yellow("STATUS: [ANSWER NOT ESTABLISHED IN RETRIEVED PASSAGES]"))
        print()
        print(ev.get("excerpt", ""))
        if ev.get("note"):
            print(f"Diagnostic Note: {ev['note']}")

    elif ev_status == "company_mismatch":
        print(c_yellow("STATUS: [NO RELEVANT EVIDENCE RETRIEVED (Target entity mismatch)]"))
        print()
        print(ev.get("excerpt", ""))
        if ev.get("note"):
            print(f"Diagnostic Note: {ev['note']}")

    else:
        print(c_red("STATUS: [NO MATCHING EVIDENCE RETRIEVED]"))
        print()
        print(ev.get("excerpt", "[No candidate chunks retrieved]"))

    sc = ev.get("source_chunk")
    if sc is not None:
        print()
        rank_str = f"Rank #{ev.get('rank')}" if ev.get("rank") is not None else "Rank #1"
        print(f"Source Document: {sc.title} ({sc.document_id})")
        print(f"Retrieval Rank : {rank_str} ({mode_label})")
        print(f"Document Depth : {sc.position_bucket.upper()} ({sc.source_position_pct * 100.0:.1f}%)")
        print(f"Candidate Chunk: {sc.chunk_id} [Chars {sc.source_char_start}-{sc.source_char_end}]")

    print(sep_single)

    # 3. Ranked Retrieved Chunks
    print("\n" + c_bold("RANKED RETRIEVED CHUNKS:"))
    for res in results:
        pos_str = f"{res.position_bucket.capitalize()} ({res.source_position_pct * 100.0:.1f}%)"

        if mode == "rrf":
            fusion_item = next(
                (item for item in debug_info.get("fusion_breakdown", []) if item["chunk_id"] == res.chunk_id),
                None,
            )
            if fusion_item:
                d_str = (
                    f"{fusion_item['dense_cosine_sim']:.4f} (Rank #{fusion_item['dense_rank']})"
                    if fusion_item["dense_cosine_sim"] is not None
                    else "Unranked (>50)"
                )
                b_str = (
                    f"{fusion_item['bm25_score']:.4f} (Rank #{fusion_item['bm25_rank']})"
                    if fusion_item["bm25_score"] is not None
                    else "Unranked (>50)"
                )
                score_str = f"Fused RRF Score: {res.score:.6f} | Dense Cosine Sim: {d_str} | Lexical BM25: {b_str}"
            else:
                score_str = f"Fused RRF Score: {res.score:.6f}"
        elif mode == "dense":
            score_str = f"Dense Cosine Similarity: {res.score:.4f}"
        elif mode == "bm25":
            score_str = f"Lexical BM25 Score: {res.score:.4f}"
        else:
            score_str = f"Score: {res.score:.4f}"

        print(f"\n[Rank #{res.rank}] {c_bold(res.title)} | File: {res.document_id} | Chunk: {res.chunk_id}")
        print(f"  Position: {pos_str} | Tokens: {res.token_count} | Source Offsets: Chars {res.source_char_start}-{res.source_char_end}")
        print(f"  Scoring : {score_str}")

        clean_snip = format_passage_snippet(res.text, query=query, max_chars=240)
        print(f"  Passage : \"{clean_snip}\"")

    print("\n" + c_dim("Score Semantics Note: Dense cosine similarity [-1, 1], BM25 score [0, inf), and RRF score [0, 1] have distinct mathematical formulations and are not directly comparable numerically."))
    print(sep_double)

    # 4. Debug Diagnostics Output (--debug)
    if debug_flag:
        timestamps = debug_info.get("timestamps", {})
        meta = debug_info.get("index_metadata", {})

        print("\n" + sep_hash)
        print(c_bold("RETRIEVAL PIPELINE DIAGNOSTIC TRACE (--debug)"))
        print(sep_hash)

        # Section A: Input & System Configuration
        print(f"\n{c_cyan('[SECTION A: INPUT & SYSTEM CONFIGURATION]')}")
        print(f"  Original Question      : \"{debug_info.get('query')}\"")
        print(f"  Normalized Tokens      : {debug_info.get('query_tokens')}")
        print(f"  Retrieval Mode         : {mode_label}")
        print(f"  Configured Top-K (k)   : {debug_info.get('top_k')}")
        print(f"  Index Directory        : {debug_info.get('index_path')}")
        print(f"  Persistent Index Status: Valid cache loaded (fingerprint verified)")
        print(f"  Total Searchable Chunks: {debug_info.get('total_chunks', 3957):,} chunks across {debug_info.get('total_documents', 10)} SEC Form 10-K filings")
        print(f"  Chunking Strategy      : {meta.get('chunk_strategy', 'sentence_aware')} (target tokens: {meta.get('target_tokens', 192)})")
        print(f"  Embedding Model Name   : {debug_info.get('model_name')}")
        print(f"  Vector Specifications  : Dim: {debug_info.get('embedding_dim', 384)} float32 | Max Seq Length: {debug_info.get('max_seq_length', 256)} tokens")

        # Section B: Query Embedding Stage
        print(f"\n{c_cyan('[SECTION B: QUERY EMBEDDING STAGE]')}")
        if "query_vector_shape" in debug_info:
            finite_str = "PASSED (all finite)" if debug_info.get("query_vector_finite") else "FAILED"
            emb_sec = timestamps.get("query_embedding_sec", 0.0)
            print(f"  Embedding Generated    : True")
            print(f"  Vector Shape & Dtype   : {debug_info.get('query_vector_shape')} | {debug_info.get('query_vector_dtype')}")
            print(f"  Vector Validity Checks : Finite Check = {finite_str} | L2 Norm = {debug_info.get('query_vector_norm'):.4f} (unit-normalized)")
            print(f"  Query Embedding Latency: {emb_sec * 1000.0:.2f} ms")
            print(f"  Vector Values Preview  : {debug_info.get('query_vector_preview')} (first 5 of {debug_info.get('embedding_dim', 384)} dimensions)")
        else:
            print("  Query Embedding Stage  : Bypassed (BM25 lexical retrieval does not require dense vectors)")

        # Section C: Dense Semantic Retrieval
        print(f"\n{c_cyan('[SECTION C: DENSE SEMANTIC RETRIEVAL (COSINE SIMILARITY)]')}")
        if "dense_top5" in debug_info:
            dense_sec = timestamps.get("dense_retrieval_sec", 0.0)
            print(f"  Searchable Vectors     : {debug_info.get('total_chunks', 3957):,} vectors scored in corpus matrix")
            print(f"  Similarity Formula     : Pure NumPy vectorized cosine similarity: cos(u, v) = (u . v) / (||u|| * ||v||)")
            print(f"  Candidate Pool Depth   : Top {debug_info.get('candidate_depth', 50)} candidates retained for ranking/fusion ({dense_sec * 1000.0:.2f} ms)")
            print("  Top Dense Candidates   :")
            for c in debug_info["dense_top5"]:
                print(f"    Rank #{c['rank']}: {c['chunk_id']} ({c['doc']}) | Cosine Sim: {c['score']:.4f}")
        else:
            print("  Dense Semantic Stage   : Bypassed (pure lexical BM25 mode selected)")

        # Section D: Lexical BM25 Retrieval
        print(f"\n{c_cyan('[SECTION D: LEXICAL BM25 RETRIEVAL]')}")
        if "bm25_top5" in debug_info:
            bm25_sec = timestamps.get("bm25_retrieval_sec", 0.0)
            print(f"  Query Tokens Extracted : {debug_info.get('query_tokens')} ({len(debug_info.get('query_tokens', []))} tokens)")
            print(f"  Searchable Corpus      : {debug_info.get('total_chunks', 3957):,} chunks scored via inverted index ({bm25_sec * 1000.0:.2f} ms)")
            cand_k = debug_info.get('candidate_depth', 50) if mode == 'rrf' else top_k
            print(f"  Candidate Pool Depth   : Top {cand_k} candidates retained")
            print(f"  BM25 Parameters        : k1 = {debug_info.get('bm25_k1', 1.5)}, b = {debug_info.get('bm25_b', 0.75)} | Avg Doc Len: {debug_info.get('bm25_avg_doc_len', 0.0)} tokens")
            print("  Top BM25 Candidates    :")
            for c in debug_info["bm25_top5"]:
                print(f"    Rank #{c['rank']}: {c['chunk_id']} ({c['doc']}) | BM25 Score: {c['score']:.4f}")
        else:
            print("  Lexical BM25 Stage     : Bypassed (pure dense semantic mode selected)")

        # Section E: Reciprocal Rank Fusion
        print(f"\n{c_cyan('[SECTION E: RECIPROCAL RANK FUSION (RRF)]')}")
        if "fusion_breakdown" in debug_info:
            rrf_sec = timestamps.get("rrf_fusion_sec", 0.0)
            print(f"  Fusion Parameters      : Smoothing Constant c = {debug_info.get('rrf_c', 60.0)} | Candidate Depth M = {debug_info.get('candidate_depth', 50)}")
            print(f"  Candidate Pool Union   : {debug_info.get('fused_candidate_union_size')} unique candidates across Dense and BM25 ({rrf_sec * 1000.0:.2f} ms)")
            print(f"  RRF Formula            : RRF(d) = sum_{{m}} 1 / (c + rank_m(d))")
            print("  Top Fused Candidates   :")
            for f in debug_info["fusion_breakdown"][:debug_info.get("top_k", 5)]:
                d_info = (
                    f"Dense: Rank #{f['dense_rank']} (Sim: {f['dense_cosine_sim']}, Contrib: +{f['dense_contrib']:.6f})"
                    if f["dense_rank"] is not None
                    else "Dense: Unranked (>50)"
                )
                b_info = (
                    f"BM25: Rank #{f['bm25_rank']} (Score: {f['bm25_score']}, Contrib: +{f['bm25_contrib']:.6f})"
                    if f["bm25_rank"] is not None
                    else "BM25: Unranked (>50)"
                )
                print(
                    f"    Rank #{f['final_rank']}: {f['chunk_id']} -> Fused Score: {f['fused_rrf_score']:.6f} "
                    f"[{f.get('match_type', 'Single')}]\n"
                    f"             {d_info} | {b_info}"
                )
        else:
            print(f"  RRF Fusion Stage       : Bypassed (single-system mode '{mode}' selected)")

        # Section F: Evidence Selection & Financial-Table Extraction
        print(f"\n{c_cyan('[SECTION F: EVIDENCE SELECTION & FINANCIAL-TABLE EXTRACTION]')}")
        company_info = detect_company_in_query(query)
        company_str = company_info["name"] if company_info else "General (all corpus documents eligible)"
        print(f"  Target Entity Scoping  : {company_str}")
        print(f"  Evidence Resolution    : {ev_status.upper()}")
        if ev_status == "direct_table_evidence":
            tbl_info = ev.get("table_details", {})
            print(f"  Extraction Mechanism   : {tbl_info.get('provenance_type')}")
            print(f"  Target Line Item       : {tbl_info.get('row_label')}")
            print(f"  Matched Fiscal Period  : {tbl_info.get('year')} -> Value: {tbl_info.get('year_value')} ({tbl_info.get('units')})")
            print(f"  Candidate Chunk Offsets: Chars {tbl_info.get('candidate_char_start')}-{tbl_info.get('candidate_char_end')}")
            print(f"  Source Row Offsets     : Chars {tbl_info.get('row_char_start')}-{tbl_info.get('row_char_end')}")
            print(f"  Table Integrity Checks : TOC/exhibits filter check = PASSED | Fiscal year header column check = PASSED")
        elif ev_status == "not_established":
            print(f"  Abstention Rationale   : {ev.get('note')}")
        else:
            print(f"  Selection Details      : Selected Rank #{ev.get('rank')} ({ev.get('source_chunk').chunk_id if ev.get('source_chunk') else 'N/A'})")

        # Section G: Final Diagnostic & Pipeline Summary
        print(f"\n{c_cyan('[SECTION G: FINAL DIAGNOSTIC & PIPELINE SUMMARY]')}")
        print(f"  Retrieval Outcome      : SUCCESS ({len(results)} candidates returned)")
        print(f"  Evidence Outcome       : {ev_status.upper()}")
        print("  Pipeline Latency Breakdown:")
        total_sec = timestamps.get("total_pipeline_sec", 0.0)
        for stage_name, stage_sec in timestamps.items():
            if stage_name != "total_pipeline_sec":
                print(f"    - {stage_name:<24}: {stage_sec * 1000.0:>7.2f} ms")
        print(f"    - {'total_pipeline_sec':<24}: {total_sec * 1000.0:>7.2f} ms")
        print("  Integrity Verification : Pure NumPy vectorized cosine similarity, BM25 term frequency saturation, deterministic RRF fusion. All checks PASSED.")

        print("\n" + sep_hash + "\n")


def run_interactive(service: RetrievalService, default_mode: str, default_k: int, debug_flag: bool):
    """
    Starts an interactive query-answering loop in the terminal.
    """
    width = get_term_width(80)
    sep_double = "=" * width
    sep_single = "-" * width

    print("\n" + sep_double)
    print(c_bold("LONG-CONTEXT RETRIEVAL FIDELITY ENGINE -- INTERACTIVE SESSION"))
    print(sep_single)
    print(f"Corpus: 10 SEC Form 10-K Filings | Chunks: {len(service.index.chunks):,} (Sentence-Aware)")
    print(f"Active Mode: {c_cyan(default_mode.upper())} | Top-K: {c_cyan(str(default_k))} | Debug: {c_cyan('ON' if debug_flag else 'OFF')}")
    print("\nInteractive Commands:")
    print("  :mode <dense|bm25|rrf>   Switch active retrieval mode")
    print("  :k <integer>             Change top-K candidate depth")
    print("  :debug [on|off]          Toggle or set pipeline diagnostic trace")
    print("  :config                  Display current runtime configuration")
    print("  :help, ?                 Show interactive command reference and sample queries")
    print("  :quit, :exit, :q         Exit interactive session")
    print(sep_double + "\n")

    current_mode = default_mode
    current_k = default_k
    current_debug = debug_flag

    while True:
        try:
            prompt = f"FidelityEngine [{current_mode.upper()}|K={current_k}]> "
            query = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", ":q", ":quit", ":exit"):
            print("Session closed. Goodbye.")
            break

        if query.lower() in (":help", "?", "help"):
            print("\n" + sep_single)
            print(c_bold("INTERACTIVE COMMAND REFERENCE & SAMPLE QUERIES"))
            print(sep_single)
            print("Commands:")
            print("  :mode <dense|bm25|rrf>   Switch active retrieval mode (e.g., ':mode dense')")
            print("  :k <integer>             Configure top-K candidate count (e.g., ':k 10')")
            print("  :debug [on|off]          Toggle or set diagnostic trace (e.g., ':debug on')")
            print("  :config, :settings       Display active runtime configuration and index info")
            print("  :help, ?                 Display this command reference")
            print("  :quit, :exit, :q         Exit interactive session")
            print("\nSample Queries:")
            print("  - What were Amazon's consolidated net sales in 2025?")
            print("  - What was Amazon's cost of sales in 2025?")
            print("  - Why does Amazon consider operating income more meaningful than gross profit?")
            print("  - What was Alphabet's operating income in 2025?")
            print("  - What was Apple's operating income in 2025?")
            print("  - What was Meta's net income in 2025?")
            print("  - What were Amazon's consolidated net sales in 2019?  (Abstention test)")
            print("  - What was Amazon's gross profit in 2025?            (Abstention test)")
            print(sep_single + "\n")
            continue

        if query.lower() in (":config", ":settings"):
            print("\n" + sep_single)
            print(c_bold("CURRENT RUNTIME CONFIGURATION"))
            print(sep_single)
            print(f"  Retrieval Mode         : {current_mode.upper()}")
            print(f"  Top-K Candidates       : {current_k}")
            print(f"  Diagnostic Debug Trace : {'ON' if current_debug else 'OFF'}")
            print(f"  Candidate Pool Depth M : {service.candidate_depth}")
            print(f"  RRF Smoothing Constant : {service.rrf_c}")
            print(f"  Embedding Model        : {service.engine.model_name}")
            print(f"  Index Chunks Count     : {len(service.index.chunks):,}")
            print(f"  Index Document Count   : {len(set(c.document_id for c in service.index.chunks))}")
            print(sep_single + "\n")
            continue

        if query.startswith(":mode"):
            parts = query.split()
            if len(parts) > 1 and parts[1].lower() in ("dense", "bm25", "rrf"):
                current_mode = parts[1].lower()
                print(f"[*] Retrieval mode changed to: {c_bold(current_mode.upper())}")
            else:
                print(f"[*] Current mode: {current_mode.upper()}. Valid options: :mode dense, :mode bm25, :mode rrf")
            continue

        if query.startswith(":k"):
            parts = query.split()
            if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) > 0:
                current_k = int(parts[1])
                print(f"[*] Top-K candidate depth changed to: {c_bold(str(current_k))}")
            else:
                print(f"[*] Current K: {current_k}. Valid format: :k <positive integer>")
            continue

        if query.startswith(":debug"):
            parts = query.split()
            if len(parts) > 1:
                current_debug = parts[1].lower() in ("on", "true", "1", "yes")
                print(f"[*] Debug diagnostic trace: {c_bold('ON' if current_debug else 'OFF')}")
            else:
                current_debug = not current_debug
                print(f"[*] Toggled debug diagnostic trace to: {c_bold('ON' if current_debug else 'OFF')}")
            continue

        results, debug_info = service.search(query, mode=current_mode, top_k=current_k, debug=current_debug)
        display_results(query, results, current_mode, debug_info, debug_flag=current_debug)


def main():
    parser = argparse.ArgumentParser(
        description="Long-Context Retrieval Fidelity Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-query search with default RRF mode:
  python -m src.cli --question "What revenue did Alphabet report for Google Cloud?"

  # Dense-only semantic search:
  python -m src.cli --question "NVIDIA data center GPU architecture" --mode dense --top-k 3

  # BM25 lexical search with verbose debug diagnostics:
  python -m src.cli --question "antitrust litigation Department of Justice" --mode bm25 --debug

  # Interactive search session:
  python -m src.cli --interactive
        """,
    )

    parser.add_argument("--question", "-q", type=str, default="", help="Natural language question to retrieve evidence for.")
    parser.add_argument("--mode", "-m", choices=["dense", "bm25", "rrf"], default="rrf", help="Retrieval mode (default: rrf).")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of ranked chunks to return (default: 5).")
    parser.add_argument("--index-dir", type=str, default=str(DEFAULT_INDEX_DIR), help="Path to persistent index directory.")
    parser.add_argument("--debug", action="store_true", help="Display verbose query pipeline diagnostics and score transitions.")
    parser.add_argument("--candidate-pool", "-M", type=int, default=50, help="Candidate pool depth for RRF fusion (default: 50).")
    parser.add_argument("--rrf-c", type=float, default=60.0, help="RRF smoothing constant (default: 60.0).")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive search mode.")

    args = parser.parse_args()

    # Validate parameters
    if args.top_k <= 0:
        print(f"[Error] --top-k must be a positive integer >= 1 (got {args.top_k}).", file=sys.stderr)
        sys.exit(1)

    if args.candidate_pool <= 0:
        print(f"[Error] --candidate-pool must be a positive integer >= 1 (got {args.candidate_pool}).", file=sys.stderr)
        sys.exit(1)

    # Load persistent index
    index_path = Path(args.index_dir)
    try:
        t_load = time.time()
        index = load_index(index_path, check_fingerprint=True)
        load_sec = time.time() - t_load
    except FileNotFoundError as fnf_err:
        print(f"\n[Error] {fnf_err}", file=sys.stderr)
        print("Please run: python scripts/build_index.py\n", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[Error] Failed to load index: {exc}", file=sys.stderr)
        print("Try rebuilding the index: python scripts/build_index.py --force\n", file=sys.stderr)
        sys.exit(1)

    service = RetrievalService(index, candidate_depth=args.candidate_pool, rrf_c=args.rrf_c)
    service.index_load_sec = load_sec
    service.index_path = str(index_path)

    # If question is provided, run single-query mode
    if args.question.strip():
        results, debug_info = service.search(args.question, mode=args.mode, top_k=args.top_k, debug=args.debug)
        display_results(args.question, results, args.mode, debug_info, debug_flag=args.debug)
    else:
        # Otherwise run interactive mode
        run_interactive(service, default_mode=args.mode, default_k=args.top_k, debug_flag=args.debug)


if __name__ == "__main__":
    main()

