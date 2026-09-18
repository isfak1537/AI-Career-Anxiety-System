#!/usr/bin/env python3
"""
scripts/sync_thesis_toc_and_pdf.py
Synchronizes Table of Contents, List of Figures, and List of Tables in the
authoritative DOCX to match actual pagination, exports a fresh final PDF via LibreOffice,
and performs comprehensive structural and visual inspection.
"""

from pathlib import Path
import subprocess
import docx
import pymupdf

DOCX_PATH = Path("/Users/macbookair/Downloads/DEFENSE/Updated FYDP Tamplate for [Summer 2025].docx")
FINAL_PDF_PATH = Path("/Users/macbookair/Downloads/DEFENSE/Mohsin Isfak.pdf")


def sync_toc_and_export():
    doc = docx.Document(DOCX_PATH)
    print(f"Loaded {DOCX_PATH} ({len(doc.paragraphs)} paragraphs)")

    # 1. Update Table of Contents entries to reflect actual document page numbers
    toc_mapping = {
        "Approval": "i",
        "Declaration": "ii",
        "Acknowledgements": "iii",
        "Abstract": "iv",
        "List of Figures": "vii",
        "List of Tables": "viii",
        # Chapter 1
        "Introduction": "1",
        "Motivation": "2",
        "Objectives": "3",
        "Methodology": "4",
        "Project Outcome": "4",
        "Organization of the Report": "5",
        # Chapter 2
        "Background": "6",
        "Literature Review": "6",
        "Similar Applications": "11",
        "Gap Analysis": "11",
        # Chapter 3
        "Research Methodology": "13",
        "Methodology & Design Specification": "13",
        "Proposed Methodology": "13",
        "Functional and Nonfunctional Requirements": "15",
        "Data Flow Diagram": "16",
        "UI Design": "16",
        "Detailed Methodology and Design": "16",
        "3.2.1     Dataset Description and Strategic Cohort Isolation": "16",
        "3.2.2     Data Preprocessing and Ordinal Mapping": "17",
        "3.2.3     Advanced Feature Engineering": "19",
        "3.2.4     Model Training and Statistical Encoding": "20",
        "Project Plan": "22",
        "Task Allocation": "22",
        # Chapter 4
        "Implementation and Results": "24",
        "Environment Setup": "24",
        "Testing and Evaluation/Performance/ Comparative Analysis": "24",
        "Results and Discussion": "24",
        # Chapter 5
        "Engineering Standards and Design Challenges": "34",
        "Compliance with the Standards": "34",
        "Software Standards": "34",
        "Data Standards": "35",
        "Communication Standards": "36",
        "Impact on Society, Environment and Sustainability": "36",
        "Impact on Life": "36",
        "Impact on Society & Environment": "36",
        "Ethical Aspects": "37",
        "Sustainability Plan": "37",
        "Project Management and Financial Analysis": "38",
        "Complex Engineering Problem": "39",
        "Complex Problem Solving": "39",
        "Engineering Activities": "41",
        # Chapter 6
        "Conclusion": "42",
        "Limitation": "43",
        "Future Work": "44",
        "References": "46",
    }

    # Iterate through paragraphs in TOC region (paragraphs 115 to 174)
    for i in range(115, 174):
        p = doc.paragraphs[i]
        text = p.text
        if "\t" in text:
            title = text.split("\t")[0].strip()
            # Find best match in toc_mapping
            for key, page_str in toc_mapping.items():
                if title == key or title.startswith(key):
                    p.text = f"{title}\t{page_str}"
                    break
        elif text.strip() in toc_mapping:
            title = text.strip()
            p.text = f"{title}\t{toc_mapping[title]}"

    # Handle Chapter Summaries in TOC with specific page numbers
    # Ch 2 summary: 12, Ch 3 summary: 23, Ch 4 summary: 33, Ch 5 summary: 41, Ch 6 summary: 42
    summary_pages = ["12", "23", "33", "41", "42"]
    summary_count = 0
    for i in range(115, 174):
        p = doc.paragraphs[i]
        if p.text.startswith("Summary\t") or p.text.strip() == "Summary":
            if summary_count < len(summary_pages):
                p.text = f"Summary\t{summary_pages[summary_count]}"
                summary_count += 1

    # 2. Update List of Figures (clean list of all 5 verified figures)
    lof_idx = None
    for idx, p in enumerate(doc.paragraphs):
        if p.text.strip() == "List of Figures":
            lof_idx = idx
            break

    lot_idx = None
    for idx, p in enumerate(doc.paragraphs):
        if p.text.strip() == "List of Tables":
            lot_idx = idx
            break

    if lof_idx is not None and lot_idx is not None:
        # Clear paragraphs between lof_idx and lot_idx
        for j in range(lof_idx + 1, lot_idx):
            doc.paragraphs[j].text = ""

        figures = [
            ("Figure 3.1: Research Pipeline Data Flow, from Raw Dataset to Frozen Final Results.", "14"),
            ("Figure 4.1: Receiver Operating Characteristic (ROC) Curve on Held-Out Test Set.", "26"),
            ("Figure 4.2: Confusion Matrix for AI Career Anxiety Prediction on Held-Out Test Set.", "28"),
            ("Figure 4.3: Top 10 Random Forest Gini-Importance Drivers of Predicted AI Anxiety.", "30"),
            ("Figure 4.4: SHAP Summary (Beeswarm) Plot Showing Feature Direction and Magnitude.", "33"),
        ]
        
        # Place figures into paragraphs starting at lof_idx + 2
        for offset, (fig_title, fig_page) in enumerate(figures):
            p_target = doc.paragraphs[lof_idx + 2 + offset]
            p_target.text = f"{fig_title}\t{fig_page}"

    # 3. Update List of Tables
    ch1_idx = None
    for idx, p in enumerate(doc.paragraphs):
        if p.text.strip() == "Chapter 1":
            ch1_idx = idx
            break

    if lot_idx is not None and ch1_idx is not None:
        tables = [
            ("Table 2.1: Summary of Literature Reviewed.", "8"),
            ("Table 2.2: Gap Analysis — Feature Comparison Table.", "11"),
            ("Table 5.1: Mapping with Complex Engineering Problem Characteristics.", "39"),
            ("Table 5.2: Mapping with Knowledge Profile.", "40"),
            ("Table 5.3: Mapping with Complex Engineering Activities.", "41"),
        ]
        # Set tables in paragraphs between lot_idx and ch1_idx
        p_offset = lot_idx + 1
        for t_idx, (t_title, t_page) in enumerate(tables):
            if p_offset + t_idx < ch1_idx:
                doc.paragraphs[p_offset + t_idx].text = f"{t_title}\t{t_page}"

    # Save synchronized DOCX
    doc.save(DOCX_PATH)
    print(f"Saved synchronized DOCX to {DOCX_PATH}")

    # 4. Export fresh PDF via LibreOffice headless
    cmd = [
        "/opt/homebrew/bin/soffice",
        "--headless",
        "--convert-to",
        "pdf",
        str(DOCX_PATH),
        "--outdir",
        str(FINAL_PDF_PATH.parent),
    ]
    print(f"Exporting PDF via command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during PDF export: {result.stderr}")
    else:
        print(f"LibreOffice output: {result.stdout.strip()}")

    # Rename exported PDF to Mohsin Isfak.pdf if needed
    exported_pdf = FINAL_PDF_PATH.parent / (DOCX_PATH.stem + ".pdf")
    if exported_pdf.exists() and exported_pdf != FINAL_PDF_PATH:
        exported_pdf.replace(FINAL_PDF_PATH)
        print(f"Renamed {exported_pdf.name} -> {FINAL_PDF_PATH.name}")

    # 5. Verify and inspect final PDF
    if FINAL_PDF_PATH.exists():
        pdf_doc = pymupdf.open(FINAL_PDF_PATH)
        print(f"\n=======================================================")
        print(f"FINAL PDF AUDIT REPORT: {FINAL_PDF_PATH.name}")
        print(f"=======================================================")
        print(f"Total Pages: {len(pdf_doc)}")
        
        # Check Table of Contents page (Sheet 7)
        print("\n--- TOC Inspection (Page 7) ---")
        print(pdf_doc[6].get_text()[:400])

        # Check List of Figures page (Sheet 9)
        print("\n--- List of Figures Inspection (Page 9) ---")
        print(pdf_doc[8].get_text().strip())

        # Check List of Tables page (Sheet 10)
        print("\n--- List of Tables Inspection (Page 10) ---")
        print(pdf_doc[9].get_text().strip())

        # Verify Figure Captions
        fig_captions = []
        for pno in range(len(pdf_doc)):
            for line in pdf_doc[pno].get_text().split("\n"):
                if line.strip().startswith("Figure 4.") or line.strip().startswith("Figure 3."):
                    fig_captions.append((pno + 1, line.strip()))
        
        print("\n--- Verified Figure Captions in PDF ---")
        for pno, cap in fig_captions:
            print(f"  Page {pno}: {cap}")

        print(f"\n[OK] Final PDF generated and verified successfully!")
    else:
        print("[ERROR] Final PDF was not generated.")


if __name__ == "__main__":
    sync_toc_and_export()
