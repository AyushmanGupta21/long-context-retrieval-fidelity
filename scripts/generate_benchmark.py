"""
Generates and validates data/benchmark_queries.json containing exactly 30
evidence-grounded question-answer pairs across 10 SEC Form 10-K reports.
10 Beginning (< 10%), 10 Middle (40% - 60%), 10 End (>= 90%).
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_FILE = PROJECT_ROOT / "data" / "benchmark_queries.json"

# Candidate definitions with unique anchors for exact substring extraction
CANDIDATES = [
    # --- ALPHABET INC. ---
    {
        "qid": "Q01",
        "doc": "alphabet_2025_10k.txt",
        "company": "Alphabet Inc.",
        "bucket": "beginning",
        "anchor": "Alphabet’s investment in the portfolio of Other Bets includes businesses that are at various stages of development, ranging from those in the research and development phase, such as X, our moonshot factory focused on developing breakthrough technologies",
        "question": "What is the primary focus of X within Alphabet's portfolio of Other Bets?",
        "answer": "Our moonshot factory focused on developing breakthrough technologies.",
    },
    {
        "qid": "Q02",
        "doc": "alphabet_2025_10k.txt",
        "company": "Alphabet Inc.",
        "bucket": "middle",
        "anchor": "| Maturities and sales of marketable securities | 86,672 | | | 103,428 | | | 83,240 |",
        "question": "According to Alphabet's consolidated statements of cash flows, what were the maturities and sales of marketable securities in 2025 (in millions)?",
        "answer": "$83,240 million",
    },
    {
        "qid": "Q03",
        "doc": "alphabet_2025_10k.txt",
        "company": "Alphabet Inc.",
        "bucket": "end",
        "anchor": "Our insider trading policy states, among other things, that our directors, officers, and employees are prohibited from trading in such securities while in possession of material, nonpublic information.",
        "question": "Under Alphabet's insider trading policy, what are directors, officers, and employees prohibited from doing?",
        "answer": "Prohibited from trading in such securities while in possession of material, nonpublic information.",
    },

    # --- AMAZON.COM, INC. ---
    {
        "qid": "Q04",
        "doc": "amazon_2025_10k.txt",
        "company": "Amazon.com, Inc.",
        "bucket": "beginning",
        "anchor": "| Jeffrey P. Bezos | | 62 | | Executive Chair | | Andrew R. Jassy | | 58 | | President and Chief Executive Officer |",
        "question": "What executive positions and ages are reported for Jeffrey P. Bezos and Andrew R. Jassy as of January 28, 2026?",
        "answer": "Jeffrey P. Bezos (age 62) is Executive Chair, and Andrew R. Jassy (age 58) is President and Chief Executive Officer.",
    },
    {
        "qid": "Q05",
        "doc": "amazon_2025_10k.txt",
        "company": "Amazon.com, Inc.",
        "bucket": "middle",
        "anchor": "| Retained earnings | 172,866 | | | 250,536 | | | Total stockholders’ equity | 285,970 | | | 411,065 |",
        "question": "According to Amazon's consolidated balance sheets, what was total stockholders' equity as of December 31, 2025 compared to 2024 (in millions)?",
        "answer": "$411,065 million as of December 31, 2025, compared to $285,970 million in 2024.",
    },
    {
        "qid": "Q06",
        "doc": "amazon_2025_10k.txt",
        "company": "Amazon.com, Inc.",
        "bucket": "end",
        "anchor": "We conducted our audit in accordance with the standards of the PCAOB. Those standards require that we plan and perform the audit to obtain reasonable assurance about whether the financial statements are free of material misstatement",
        "question": "In the auditor report on internal control, in accordance with what standards was Amazon's audit conducted?",
        "answer": "Standards of the Public Company Accounting Oversight Board (PCAOB).",
    },

    # --- META PLATFORMS, INC. ---
    {
        "qid": "Q07",
        "doc": "meta_2025_10k.txt",
        "company": "Meta Platforms, Inc.",
        "bucket": "beginning",
        "anchor": "We had a global workforce of 78,865 employees as of December 31, 2025, and we have offices in more than 90 cities around the world.",
        "question": "How many employees comprised Meta's global workforce as of December 31, 2025?",
        "answer": "78,865 employees across more than 90 cities.",
    },
    {
        "qid": "Q08",
        "doc": "meta_2025_10k.txt",
        "company": "Meta Platforms, Inc.",
        "bucket": "middle",
        "anchor": "transferred to the U.S. District Court for the Northern District of California), have been consolidated into Kadrey, et al. v. Meta Platforms, Inc . Motions for summary judgment were heard in this case on",
        "question": "What is the name of the consolidated copyright lawsuit against Meta transferred to the Northern District of California where summary judgment motions were heard?",
        "answer": "Kadrey, et al. v. Meta Platforms, Inc.",
    },
    {
        "qid": "Q09",
        "doc": "meta_2025_10k.txt",
        "company": "Meta Platforms, Inc.",
        "bucket": "end",
        "anchor": "In September 2025, we received a Statutory Notice of Deficiency (\"2017-2019 Notice\") from the IRS, asserting an additional $ 15.89 billion in tax, plus interest and penalties for our 2017 through 2019 tax years.",
        "question": "What amount of additional tax did the IRS assert against Meta in the September 2025 Statutory Notice of Deficiency?",
        "answer": "$15.89 billion in tax, plus interest and penalties for the 2017 through 2019 tax years.",
    },

    # --- IBM CORP. ---
    {
        "qid": "Q10",
        "doc": "ibm_2025_10k.txt",
        "company": "International Business Machines Corp.",
        "bucket": "beginning",
        "anchor": "common stock, $0.20 par value, outstanding as of February 6, 2026 was 932,159,101.",
        "question": "How many shares of IBM common stock were outstanding as of February 6, 2026?",
        "answer": "932,159,101 shares.",
    },
    {
        "qid": "Q11",
        "doc": "ibm_2025_10k.txt",
        "company": "International Business Machines Corp.",
        "bucket": "middle",
        "anchor": "We also rely on tools licensed from third party security vendors to monitor and manage cybersecurity risks. We periodically engage third parties to supplement our cybersecurity risk management program",
        "question": "How does IBM monitor, manage, and supplement its cybersecurity risk management program?",
        "answer": "By relying on tools licensed from third party security vendors and periodically engaging third parties.",
    },
    {
        "qid": "Q12",
        "doc": "ibm_2025_10k.txt",
        "company": "International Business Machines Corp.",
        "bucket": "end",
        "anchor": "Certification by CFO pursuant to Rule 13A-14(a) or 15D-14(a) of the Securities Exchange Act of 1934, as Adopted Pursuant to Section 302 of the Sarbanes-Oxley Act of 2002.",
        "question": "Under Exhibit 31.2, what certification was provided by IBM's Chief Financial Officer pursuant to Section 302?",
        "answer": "Certification pursuant to Rule 13A-14(a) or 15D-14(a) of the Securities Exchange Act of 1934, as Adopted Pursuant to Section 302 of the Sarbanes-Oxley Act of 2002.",
    },

    # --- NETFLIX, INC. ---
    {
        "qid": "Q13",
        "doc": "netflix_2025_10k.txt",
        "company": "Netflix, Inc.",
        "bucket": "beginning",
        "anchor": "Our paid memberships increased from 260.28 million at the end of 2023 to 301.64 million at the end of 2024, and our revenue increased from $ 33.7 billion in 2023 to $ 39.0 billion in 2024.",
        "question": "What were Netflix's global paid memberships and total revenue at year-end 2024 compared to 2023?",
        "answer": "Paid memberships grew to 301.64 million (from 260.28 million) and revenue increased to $39.0 billion (from $33.7 billion).",
    },
    {
        "qid": "Q14",
        "doc": "netflix_2025_10k.txt",
        "company": "Netflix, Inc.",
        "bucket": "middle",
        "anchor": "| Spencer Neumann (2) | | Chief Financial Officer | | Adoption | | 10/23/2025 | | 12/31/2026 | | 314,880 |",
        "question": "Under Netflix's Rule 10b5-1 trading arrangement disclosure, when did CFO Spencer Neumann adopt his plan and how many aggregate shares were scheduled for potential sale?",
        "answer": "Adopted on 10/23/2025 with 314,880 shares eligible for sale through 12/31/2026.",
    },
    {
        "qid": "Q15",
        "doc": "netflix_2025_10k.txt",
        "company": "Netflix, Inc.",
        "bucket": "end",
        "anchor": "summarizes the Company's contributions to multiemployer pension and health plans for the years ended December 31, 2025, 2024 and 2023 (in thousands): | | 2025 | | 2024 | | 2023 | | Multiemployer pension and health plan contributions | $ | 83,088 | | $ | 71,852 |",
        "question": "What were Netflix's multiemployer pension and health plan contributions for the year ended December 31, 2025 (in thousands)?",
        "answer": "$83,088 thousand.",
    },

    # --- MICROSOFT CORPORATION ---
    {
        "qid": "Q16",
        "doc": "microsoft_2026_10k.txt",
        "company": "Microsoft Corporation",
        "bucket": "beginning",
        "anchor": "OPERATING SEGMENTS We operate our business and report our financial performance using three segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
        "question": "What are the three operating segments used by Microsoft to report its business performance?",
        "answer": "Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
    },
    {
        "qid": "Q17",
        "doc": "microsoft_2026_10k.txt",
        "company": "Microsoft Corporation",
        "bucket": "middle",
        "anchor": "| Diluted | | $ | 17.95 | | | $ | 13.64 | | | $ | 11.80 |",
        "question": "According to Microsoft's income statement, what was diluted earnings per share for fiscal year 2026 compared to 2025 and 2024?",
        "answer": "$17.95 in 2026, compared to $13.64 in 2025 and $11.80 in 2024.",
    },
    {
        "qid": "Q18",
        "doc": "microsoft_2026_10k.txt",
        "company": "Microsoft Corporation",
        "bucket": "end",
        "anchor": "We will provide disclosure of delinquent Section 16(a) reports, if any, in our Proxy Statement under the caption “Delinquent Section 16(a) Reports,” and that disclosure is incorporated herein by reference.",
        "question": "Where does Microsoft state it will provide disclosure of delinquent Section 16(a) reports?",
        "answer": "In its Proxy Statement under the caption 'Delinquent Section 16(a) Reports'.",
    },

    # --- NVIDIA CORPORATION ---
    {
        "qid": "Q19",
        "doc": "nvidia_2026_10k.txt",
        "company": "NVIDIA Corporation",
        "bucket": "beginning",
        "anchor": "Over the past 25 years, we have built full software stacks that run on top of our GPUs and CUDA to bring AI to the world’s largest industries, from healthcare and financial services to manufacturing and transportation.",
        "question": "Over what timeframe has NVIDIA built full software stacks on top of GPUs and CUDA, and for what industries?",
        "answer": "Over the past 25 years, bringing AI to industries such as healthcare, financial services, manufacturing, and transportation.",
    },
    {
        "qid": "Q20",
        "doc": "nvidia_2026_10k.txt",
        "company": "NVIDIA Corporation",
        "bucket": "middle",
        "anchor": "Revenue for fiscal year 2026 was $215.9 billion, up 65% from a year ago. Data Center revenue for fiscal year 2026 was up 68% from a year ago.",
        "question": "What was NVIDIA's total revenue and growth rate for fiscal year 2026, and how much did Data Center revenue increase?",
        "answer": "Total revenue was $215.9 billion (up 65% from a year ago), with Data Center revenue up 68%.",
    },
    {
        "qid": "Q21",
        "doc": "nvidia_2026_10k.txt",
        "company": "NVIDIA Corporation",
        "bucket": "end",
        "anchor": "| Total revenue | $ | 215,938 | | | $ | 130,497 | | | $ | 60,922 |",
        "question": "In NVIDIA's revenue footnote table, what was the total consolidated revenue reported for fiscal year 2026 compared to 2025 and 2024 (in millions)?",
        "answer": "$215,938 million in FY2026, compared to $130,497 million in FY2025 and $60,922 million in FY2024.",
    },

    # --- APPLE INC. ---
    {
        "qid": "Q22",
        "doc": "apple_2025_10k.txt",
        "company": "Apple Inc.",
        "bucket": "beginning",
        "anchor": "The Company manages its business primarily on a geographic basis. The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific.",
        "question": "What are Apple's reportable geographic operating segments?",
        "answer": "Americas, Europe, Greater China, Japan and Rest of Asia Pacific.",
    },
    {
        "qid": "Q23",
        "doc": "apple_2025_10k.txt",
        "company": "Apple Inc.",
        "bucket": "middle",
        "anchor": "The Company will adopt ASU 2023-09 in its fourth quarter of 2026 using a prospective transition method.",
        "question": "When will Apple adopt ASU 2023-09 and using what transition method?",
        "answer": "In its fourth quarter of 2026 using a prospective transition method.",
    },
    {
        "qid": "Q24",
        "doc": "apple_2025_10k.txt",
        "company": "Apple Inc.",
        "bucket": "end",
        "anchor": "Floating Rate Notes due 2017, Floating Rate Notes due 2019, 1.05% Notes due 2017, 2.10% Notes due 2019, 2.85% Notes due 2021, 3.45% Notes due 2024, 4.375% Notes due 2045",
        "question": "What coupon and maturity are listed for the senior notes due 2045 in Apple's Exhibit 4.10 description?",
        "answer": "4.375% Notes due 2045.",
    },

    # --- ADOBE INC. ---
    {
        "qid": "Q25",
        "doc": "adobe_2025_10k.txt",
        "company": "Adobe Inc.",
        "bucket": "beginning",
        "anchor": "embedding AI into our portfolio of solutions, such as Adobe Experience Platform AI Assistant. Adobe Experience Platform is a foundational platform that enables organizations to deliver connected customer experiences across any channel.",
        "question": "How does Adobe describe the role and function of the Adobe Experience Platform?",
        "answer": "A foundational platform that enables organizations to deliver connected customer experiences across any channel.",
    },
    {
        "qid": "Q26",
        "doc": "adobe_2025_10k.txt",
        "company": "Adobe Inc.",
        "bucket": "middle",
        "anchor": "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK Although our business is primarily U.S. Dollar denominated, we face foreign currency risk as our international revenues and expenses are generated in foreign currencies",
        "question": "According to Adobe's Item 7A disclosure, what primary market risk does Adobe face despite being primarily U.S. Dollar denominated?",
        "answer": "Foreign currency risk from international revenues and expenses generated in foreign currencies.",
    },
    {
        "qid": "Q27",
        "doc": "adobe_2025_10k.txt",
        "company": "Adobe Inc.",
        "bucket": "end",
        "anchor": "A critical audit matter is a matter arising from the current period audit of the consolidated financial statements that was communicated or required to be communicated to the audit committee and that: (1) relates to accounts or disclosures that are material to the consolidated financial statements",
        "question": "In the auditor report for Adobe, what definition is given for a critical audit matter?",
        "answer": "A matter arising from the current period audit communicated to the audit committee relating to accounts or disclosures material to the financial statements and involving especially challenging, subjective, or complex auditor judgment.",
    },

    # --- ORACLE CORPORATION ---
    {
        "qid": "Q28",
        "doc": "oracle_2026_10k.txt",
        "company": "Oracle Corporation",
        "bucket": "beginning",
        "anchor": "Oracle Fusion Cloud Human Capital Management (HCM), which is designed to help organizations find, develop and retain their talent, enable collaboration, provide workforce insights, improve business process",
        "question": "What is the primary design purpose of Oracle Fusion Cloud Human Capital Management (HCM)?",
        "answer": "Designed to help organizations find, develop and retain talent, enable collaboration, provide workforce insights, and improve operational processes.",
    },
    {
        "qid": "Q29",
        "doc": "oracle_2026_10k.txt",
        "company": "Oracle Corporation",
        "bucket": "middle",
        "anchor": "tax contingencies was approximately $ 1,327 million as of May 31, 2026, which is recorded in other non-current liabilities in our consolidated balance sheet, the nature of which is uncertain with respect to settlement or release with the relevant tax authorities.",
        "question": "What was Oracle's accrued liability for tax contingencies as of May 31, 2026 (in millions)?",
        "answer": "Approximately $1,327 million recorded in other non-current liabilities.",
    },
    {
        "qid": "Q30",
        "doc": "oracle_2026_10k.txt",
        "company": "Oracle Corporation",
        "bucket": "end",
        "anchor": "| Germany | | | 1,993 | | | | 2,432 | | | | 1,817 | | | | 2,013 | | | | 1,794 |",
        "question": "In Oracle's geographic segment disclosures, what were the total revenues attributable to Germany for fiscal 2026 compared to fiscal 2025 (in millions)?",
        "answer": "$2,013 million in fiscal 2026 compared to $1,794 million in fiscal 2025.",
    },
]


def validate_and_build():
    results = []
    bucket_counts = {"beginning": 0, "middle": 0, "end": 0}

    for c in CANDIDATES:
        fp = DOCUMENTS_DIR / c["doc"]
        if not fp.exists():
            raise FileNotFoundError(f"Missing source file: {fp}")
        text = fp.read_text(encoding="utf-8")
        words = text.split()
        total_words = len(words)

        # 1. Exact substring check
        anchor = c["anchor"]
        char_start = text.find(anchor)
        if char_start == -1:
            raise ValueError(f"[{c['qid']}] Anchor text not found as substring in {c['doc']}:\n'{anchor[:80]}...'")
        char_end = char_start + len(anchor)

        # 2. Verify substring matches exactly
        extracted_passage = text[char_start:char_end]
        assert extracted_passage == anchor, f"[{c['qid']}] Substring mismatch"

        # 3. Compute word start and word end
        # Words strictly before char_start
        prefix_words = text[:char_start].split()
        word_start = len(prefix_words)
        passage_words = extracted_passage.split()
        word_end = word_start + len(passage_words) - 1

        # Verify word alignment: text.split()[word_start:word_end+1] contains passage
        assert words[word_start : word_end + 1] == passage_words, f"[{c['qid']}] Word alignment mismatch"

        # 4. Compute normalized position: evidence starting word / total word count
        norm_pos = word_start / total_words

        # 5. Bucket validation: beginning < 0.10, middle [0.40, 0.60], end >= 0.90
        target_bucket = c["bucket"].lower()
        if target_bucket == "beginning":
            if norm_pos >= 0.10:
                raise ValueError(f"[{c['qid']}] Beginning question has norm_pos={norm_pos:.4f} >= 0.10")
        elif target_bucket == "middle":
            if not (0.40 <= norm_pos <= 0.60):
                raise ValueError(f"[{c['qid']}] Middle question has norm_pos={norm_pos:.4f} outside [0.40, 0.60]")
        elif target_bucket == "end":
            if norm_pos < 0.90:
                raise ValueError(f"[{c['qid']}] End question has norm_pos={norm_pos:.4f} < 0.90")
        else:
            raise ValueError(f"Unknown bucket {target_bucket}")

        bucket_counts[target_bucket] += 1

        entry = {
            "question_id": c["qid"],
            "document_id": c["doc"],
            "company": c["company"],
            "filename": c["doc"],
            "question": c["question"],
            "answer": c["answer"],
            "evidence": extracted_passage,
            "source_char_start": char_start,
            "source_char_end": char_end,
            "source_word_start": word_start,
            "source_word_end": word_end,
            "total_word_count": total_words,
            "normalized_position": round(norm_pos, 4),
            "position_bucket": target_bucket,
        }
        results.append(entry)

    # Verify counts
    assert len(results) == 30, f"Expected 30 questions, got {len(results)}"
    assert bucket_counts["beginning"] == 10, f"Expected 10 beginning, got {bucket_counts['beginning']}"
    assert bucket_counts["middle"] == 10, f"Expected 10 middle, got {bucket_counts['middle']}"
    assert bucket_counts["end"] == 10, f"Expected 10 end, got {bucket_counts['end']}"

    benchmark_data = {
        "benchmark_name": "Long-Context SEC 10-K Retrieval Fidelity Benchmark",
        "description": "30 evidence-grounded queries stratified equally across Beginning (<10%), Middle (40-60%), and End (>=90%) document depths.",
        "total_queries": len(results),
        "bucket_distribution": bucket_counts,
        "queries": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"Successfully generated and validated {len(results)} benchmark queries into {OUTPUT_FILE}!")
    print(f"Distribution: {bucket_counts}")


if __name__ == "__main__":
    validate_and_build()
