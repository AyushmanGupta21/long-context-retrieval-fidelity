"""
Constructs and strictly validates data/benchmark_queries.json.
Generates 30 evidence-grounded queries across all 10 SEC Form 10-K filings:
- 10 Beginning (< 10%)
- 10 Middle (40% - 60%)
- 10 End (>= 90%)
Exactly 3 questions per report (1 Beginning, 1 Middle, 1 End).
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_FILE = PROJECT_ROOT / "data" / "benchmark_queries.json"

QUERY_SPECS = [
    # --- 1. ALPHABET INC. ---
    {
        "qid": "Q01",
        "doc": "alphabet_2025_10k.txt",
        "company": "Alphabet Inc.",
        "bucket": "beginning",
        "find_phrase": "Alphabet’s investment in the portfolio of Other Bets includes businesses that are at various stages of development, ranging from those in the research and development phase, such as X, our moonshot factory focused on developing breakthrough technologies",
        "question": "What is the primary role of X within Alphabet's portfolio of Other Bets?",
        "answer": "Our moonshot factory focused on developing breakthrough technologies.",
    },
    {
        "qid": "Q02",
        "doc": "alphabet_2025_10k.txt",
        "company": "Alphabet Inc.",
        "bucket": "middle",
        "find_phrase": "Maturities and sales of marketable securities | 86,672 | | | 103,428 | | | 83,240",
        "question": "According to Alphabet's cash flow statement, what were the maturities and sales of marketable securities in 2025 (in millions)?",
        "answer": "$83,240 million",
    },
    {
        "qid": "Q03",
        "doc": "alphabet_2025_10k.txt",
        "company": "Alphabet Inc.",
        "bucket": "end",
        "find_phrase": "Our insider trading policy states, among other things, that our directors, officers, and employees are prohibited from trading in such securities while in possession of material, nonpublic information.",
        "question": "Under Alphabet's insider trading policy, what are directors, officers, and employees prohibited from doing?",
        "answer": "Prohibited from trading in securities while in possession of material, nonpublic information.",
    },

    # --- 2. AMAZON.COM, INC. ---
    {
        "qid": "Q04",
        "doc": "amazon_2025_10k.txt",
        "company": "Amazon.com, Inc.",
        "bucket": "beginning",
        "find_phrase": "Jeffrey P. Bezos | | 62 | | Executive Chair |",
        "question": "What position and age are reported for Jeffrey P. Bezos in Amazon's executive officers disclosure?",
        "answer": "Executive Chair, age 62.",
    },
    {
        "qid": "Q05",
        "doc": "amazon_2025_10k.txt",
        "company": "Amazon.com, Inc.",
        "bucket": "middle",
        "find_phrase": "Retained earnings | 172,866 | | | 250,536 | |",
        "question": "According to Amazon's consolidated balance sheets, what were retained earnings as of December 31, 2025 compared to 2024 (in millions)?",
        "answer": "$250,536 million as of December 31, 2025, compared to $172,866 million in 2024.",
    },
    {
        "qid": "Q06",
        "doc": "amazon_2025_10k.txt",
        "company": "Amazon.com, Inc.",
        "bucket": "end",
        "find_phrase": "standards of the Public Company Accounting Oversight Board (United States) (PCAOB)",
        "question": "In accordance with what professional oversight standards was Amazon's financial audit conducted?",
        "answer": "Standards of the Public Company Accounting Oversight Board (United States) (PCAOB).",
    },

    # --- 3. META PLATFORMS, INC. ---
    {
        "qid": "Q07",
        "doc": "meta_2025_10k.txt",
        "company": "Meta Platforms, Inc.",
        "bucket": "beginning",
        "find_phrase": "We had a global workforce of 78,865 employees as of December",
        "question": "How many employees comprised Meta's global workforce as of December 31, 2025?",
        "answer": "78,865 employees.",
    },
    {
        "qid": "Q08",
        "doc": "meta_2025_10k.txt",
        "company": "Meta Platforms, Inc.",
        "bucket": "middle",
        "find_phrase": "Kadrey, et al. v. Meta Platforms, Inc.",
        "question": "What is the name of the consolidated copyright lawsuit against Meta involving author copyright claims in the Northern District of California?",
        "answer": "Kadrey, et al. v. Meta Platforms, Inc.",
    },
    {
        "qid": "Q09",
        "doc": "meta_2025_10k.txt",
        "company": "Meta Platforms, Inc.",
        "bucket": "end",
        "find_phrase": "Our 2020 and subsequent tax years remain open to examination by the IRS. Our 2021 and subsequent tax years remain open to examination by the Irish Revenue Commissioners.",
        "question": "Which tax years for Meta remain open to examination by the IRS and Irish Revenue Commissioners?",
        "answer": "2020 and subsequent tax years by the IRS, and 2021 and subsequent tax years by the Irish Revenue Commissioners.",
    },

    # --- 4. IBM CORP. ---
    {
        "qid": "Q10",
        "doc": "ibm_2025_10k.txt",
        "company": "International Business Machines Corp.",
        "bucket": "beginning",
        "find_phrase": "Capital stock, par value $.20 per share | | IBM | | New York Stock Exchange |",
        "question": "Under securities registered pursuant to Section 12(b), what trading symbol and stock exchange are listed for IBM capital stock?",
        "answer": "Trading symbol IBM on the New York Stock Exchange.",
    },
    {
        "qid": "Q11",
        "doc": "ibm_2025_10k.txt",
        "company": "International Business Machines Corp.",
        "bucket": "middle",
        "find_phrase": "third party security vendors to monitor and manage cybersecurity",
        "question": "How does IBM monitor and manage its cybersecurity risks?",
        "answer": "By relying on tools licensed from third party security vendors and periodically engaging third parties.",
    },
    {
        "qid": "Q12",
        "doc": "ibm_2025_10k.txt",
        "company": "International Business Machines Corp.",
        "bucket": "end",
        "find_phrase": "Certification by CFO pursuant to Rule 13A-14(a) or 15D-14(a) of the Securities Exchange Act of 1934",
        "question": "Under Exhibit 31.2, what certification was provided by IBM's Chief Financial Officer?",
        "answer": "Certification pursuant to Rule 13A-14(a) or 15D-14(a) of the Securities Exchange Act of 1934, as Adopted Pursuant to Section 302 of the Sarbanes-Oxley Act of 2002.",
    },

    # --- 5. NETFLIX, INC. ---
    {
        "qid": "Q13",
        "doc": "netflix_2025_10k.txt",
        "company": "Netflix, Inc.",
        "bucket": "beginning",
        "find_phrase": "We conduct pay equity analyses at least annually",
        "question": "How frequently does Netflix conduct pay equity analyses for its workforce?",
        "answer": "At least annually.",
    },
    {
        "qid": "Q14",
        "doc": "netflix_2025_10k.txt",
        "company": "Netflix, Inc.",
        "bucket": "middle",
        "find_phrase": "Spencer Neumann (2) | | Chief Financial Officer",
        "question": "Under Netflix's Rule 10b5-1 trading disclosure, when did CFO Spencer Neumann adopt his plan and how many aggregate shares were scheduled for potential sale?",
        "answer": "Adopted on 10/23/2025 with 314,880 shares eligible for sale through 12/31/2026.",
    },
    {
        "qid": "Q15",
        "doc": "netflix_2025_10k.txt",
        "company": "Netflix, Inc.",
        "bucket": "end",
        "find_phrase": "co-chief executive officers, who review financial information presented on a consolidated basis",
        "question": "Who serves as Netflix's chief operating decision maker (CODM) and how do they review financial information?",
        "answer": "Its co-chief executive officers, who review financial information presented on a consolidated basis.",
    },

    # --- 6. MICROSOFT CORPORATION ---
    {
        "qid": "Q16",
        "doc": "microsoft_2026_10k.txt",
        "company": "Microsoft Corporation",
        "bucket": "beginning",
        "find_phrase": "Productivity and Business Processes, Intelligent Cloud, and More Personal Computing",
        "question": "What are the three operating segments used by Microsoft to report its business performance?",
        "answer": "Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
    },
    {
        "qid": "Q17",
        "doc": "microsoft_2026_10k.txt",
        "company": "Microsoft Corporation",
        "bucket": "middle",
        "find_phrase": "Diluted | | $ | 17.95",
        "question": "According to Microsoft's income statement, what was diluted earnings per share for fiscal year 2026 compared to 2025 and 2024?",
        "answer": "$17.95 in 2026, compared to $13.64 in 2025 and $11.80 in 2024.",
    },
    {
        "qid": "Q18",
        "doc": "microsoft_2026_10k.txt",
        "company": "Microsoft Corporation",
        "bucket": "end",
        "find_phrase": "Delinquent Section 16(a) Reports",
        "question": "Where does Microsoft state it will provide disclosure of delinquent Section 16(a) reports?",
        "answer": "In its Proxy Statement under the caption 'Delinquent Section 16(a) Reports'.",
    },

    # --- 7. NVIDIA CORPORATION ---
    {
        "qid": "Q19",
        "doc": "nvidia_2026_10k.txt",
        "company": "NVIDIA Corporation",
        "bucket": "beginning",
        "find_phrase": "Over the past 5 years, we have built full software stacks that run on top of our GPUs and CUDA to bring AI to the world",
        "question": "Over what timeframe has NVIDIA built full software stacks on top of GPUs and CUDA, and what domains are cited?",
        "answer": "Over the past 5 years, bringing AI to the world's largest industries including NVIDIA DRIVE for autonomous driving and Clara for healthcare.",
    },
    {
        "qid": "Q20",
        "doc": "nvidia_2026_10k.txt",
        "company": "NVIDIA Corporation",
        "bucket": "middle",
        "find_phrase": "Revenue for fiscal year 2026 was $215.9 billion, up 65% from a year ago.",
        "question": "What was NVIDIA's total revenue for fiscal year 2026 and its growth rate from the prior year?",
        "answer": "Total revenue was $215.9 billion, up 65% from a year ago.",
    },
    {
        "qid": "Q21",
        "doc": "nvidia_2026_10k.txt",
        "company": "NVIDIA Corporation",
        "bucket": "end",
        "find_phrase": "Total revenue | $ | 215,938 | | | $ | 130,497 | | | $ | 60,922 |",
        "question": "In NVIDIA's revenue footnote table, what was the total consolidated revenue reported for fiscal year 2026 compared to 2025 and 2024 (in millions)?",
        "answer": "$215,938 million in FY2026, compared to $130,497 million in FY2025 and $60,922 million in FY2024.",
    },

    # --- 8. APPLE INC. ---
    {
        "qid": "Q22",
        "doc": "apple_2025_10k.txt",
        "company": "Apple Inc.",
        "bucket": "beginning",
        "find_phrase": "Americas, Europe, Greater China, Japan and Rest of Asia Pacific",
        "question": "What are Apple's reportable geographic operating segments?",
        "answer": "Americas, Europe, Greater China, Japan and Rest of Asia Pacific.",
    },
    {
        "qid": "Q23",
        "doc": "apple_2025_10k.txt",
        "company": "Apple Inc.",
        "bucket": "middle",
        "find_phrase": "The Company will adopt ASU 2023-09 in its fourth quarter of 2026 using a prospective transition method.",
        "question": "When will Apple adopt ASU 2023-09 and using what transition method?",
        "answer": "In its fourth quarter of 2026 using a prospective transition method.",
    },
    {
        "qid": "Q24",
        "doc": "apple_2025_10k.txt",
        "company": "Apple Inc.",
        "bucket": "end",
        "find_phrase": "Officer’s Certificate of the Registrant, dated as of May 13, 2015, including forms of global notes",
        "question": "Under Exhibit 4.7, what officer certificate and forms of notes are incorporated by reference for Apple?",
        "answer": "Officer's Certificate of the Registrant dated May 13, 2015, including forms of global notes.",
    },

    # --- 9. ADOBE INC. ---
    {
        "qid": "Q25",
        "doc": "adobe_2025_10k.txt",
        "company": "Adobe Inc.",
        "bucket": "beginning",
        "find_phrase": "For marketing professionals, we unify creative production and marketing execution",
        "question": "How does Adobe describe its solution for marketing professionals in Adobe Experience Cloud?",
        "answer": "Unifies creative production and marketing execution with comprehensive content management and digital marketing capabilities.",
    },
    {
        "qid": "Q26",
        "doc": "adobe_2025_10k.txt",
        "company": "Adobe Inc.",
        "bucket": "middle",
        "find_phrase": "ITEM 7A.  QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
        "question": "What is the heading and title of Item 7A in Adobe's report?",
        "answer": "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
    },
    {
        "qid": "Q27",
        "doc": "adobe_2025_10k.txt",
        "company": "Adobe Inc.",
        "bucket": "end",
        "find_phrase": "The critical audit matter communicated below is a matter arising from the current period audit",
        "question": "In the auditor report for Adobe, what definition is given for the critical audit matter communicated?",
        "answer": "A matter arising from the current period audit of the consolidated financial statements communicated or required to be communicated to the audit committee.",
    },

    # --- 10. ORACLE CORPORATION ---
    {
        "qid": "Q28",
        "doc": "oracle_2026_10k.txt",
        "company": "Oracle Corporation",
        "bucket": "beginning",
        "find_phrase": "Oracle Fusion Cloud Human Capital Management (HCM), which is designed to help organizations find, develop and retain their talent, enable collaboration, provide workforce insights",
        "question": "What is the primary design purpose of Oracle Fusion Cloud Human Capital Management (HCM)?",
        "answer": "Designed to help organizations find, develop and retain talent, enable collaboration, provide workforce insights, and improve business processes.",
    },
    {
        "qid": "Q29",
        "doc": "oracle_2026_10k.txt",
        "company": "Oracle Corporation",
        "bucket": "middle",
        "find_phrase": "We expect this upward trend to continue during fiscal 2027 and in the following fiscal years as we increase our existing data center capacity",
        "question": "Why does Oracle expect capital expenditures for data centers to trend upward during fiscal 2027 and following years?",
        "answer": "To increase existing data center capacity and establish data centers in new geographic locations to meet customer demand for cloud offerings.",
    },
    {
        "qid": "Q30",
        "doc": "oracle_2026_10k.txt",
        "company": "Oracle Corporation",
        "bucket": "end",
        "find_phrase": "Germany | | | 1,993 | | | | 2,432 | | | | 1,817 | | | | 2,013 | | | | 1,794 |",
        "question": "In Oracle's geographic segment disclosures, what were the total revenues attributable to Germany for fiscal 2026 compared to fiscal 2025 (in millions)?",
        "answer": "$2,013 million in fiscal 2026 compared to $1,794 million in fiscal 2025.",
    },
]


def extract_evidence_passage(text: str, find_phrase: str, target_bucket: str) -> tuple[str, int, int]:
    """
    Locates find_phrase within document text ensuring it matches strictly within the target bucket,
    and returns (exact_passage, char_start, char_end).
    """
    total_words = len(text.split())
    pos = 0
    while True:
        pos = text.find(find_phrase, pos)
        if pos == -1:
            raise ValueError(f"Could not find phrase: '{find_phrase[:60]}' in document.")

        # Expand to line boundaries
        s = text.rfind("\n", 0, pos)
        s = 0 if s == -1 else s + 1

        e = text.find("\n", pos + len(find_phrase))
        if e == -1:
            e = len(text)

        passage = text[s:e].strip()
        char_start = text.find(passage, s)
        char_end = char_start + len(passage)

        word_start = len(text[:char_start].split())
        norm_pos = word_start / total_words

        if target_bucket == "beginning" and norm_pos < 0.10:
            return passage, char_start, char_end
        elif target_bucket == "middle" and 0.40 <= norm_pos <= 0.60:
            return passage, char_start, char_end
        elif target_bucket == "end" and norm_pos >= 0.90:
            return passage, char_start, char_end

        pos += len(find_phrase)


def generate_benchmark():
    items = []
    bucket_counts = {"beginning": 0, "middle": 0, "end": 0}

    print("Generating and strictly validating 30 benchmark queries...")

    for spec in QUERY_SPECS:
        fn = spec["doc"]
        fp = DOCUMENTS_DIR / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing source file: {fp}")

        text = fp.read_text(encoding="utf-8")
        words = text.split()
        total_words = len(words)

        passage, c_start, c_end = extract_evidence_passage(text, spec["find_phrase"], spec["bucket"])

        # 1. Exact substring check in document text
        assert text[c_start:c_end] == passage, f"[{spec['qid']}] Substring mismatch"

        # 2. Exact word alignment
        w_start = len(text[:c_start].split())
        passage_words = passage.split()
        w_end = w_start + len(passage_words) - 1
        assert words[w_start : w_end + 1] == passage_words, f"[{spec['qid']}] Word alignment mismatch"

        # 3. Exact position and bucket check
        norm_pos = w_start / total_words
        b = spec["bucket"]
        if b == "beginning":
            assert norm_pos < 0.10, f"[{spec['qid']}] norm_pos {norm_pos:.4f} >= 0.10"
        elif b == "middle":
            assert 0.40 <= norm_pos <= 0.60, f"[{spec['qid']}] norm_pos {norm_pos:.4f} not in [0.40, 0.60]"
        elif b == "end":
            assert norm_pos >= 0.90, f"[{spec['qid']}] norm_pos {norm_pos:.4f} < 0.90"

        bucket_counts[b] += 1

        entry = {
            "question_id": spec["qid"],
            "document_id": fn,
            "company": spec["company"],
            "filename": fn,
            "question": spec["question"],
            "answer": spec["answer"],
            "evidence": passage,
            "source_char_start": c_start,
            "source_char_end": c_end,
            "source_word_start": w_start,
            "source_word_end": w_end,
            "total_word_count": total_words,
            "normalized_position": round(norm_pos, 4),
            "position_bucket": b,
        }
        items.append(entry)
        print(f"[{spec['qid']}] {b:9s} | {spec['company'][:20]:20s} | word={w_start:5d}/{total_words} ({norm_pos:.4f}) | VALID")

    assert len(items) == 30
    assert bucket_counts["beginning"] == 10
    assert bucket_counts["middle"] == 10
    assert bucket_counts["end"] == 10

    benchmark_json = {
        "benchmark_name": "Long-Context SEC 10-K Retrieval Fidelity Benchmark",
        "description": "30 evidence-grounded queries stratified across Beginning (<10%), Middle (40-60%), and End (>=90%) document depths.",
        "total_queries": len(items),
        "bucket_distribution": bucket_counts,
        "queries": items,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark_json, f, indent=2)

    print(f"\nAll 30 queries successfully generated and saved to {OUTPUT_FILE}!")
    print(f"Bucket Distribution: {bucket_counts}")


if __name__ == "__main__":
    generate_benchmark()
