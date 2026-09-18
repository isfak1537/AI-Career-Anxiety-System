#!/usr/bin/env python3
"""
Script to audit and update DEFENSE/Mohsin Isfak.pdf
Corrects front matter typos, TOC page numbers, List of Figures, List of Tables,
grammar/academic style, and eliminates unsupported overclaim phrases.
"""

import sys
import pymupdf

def update_pdf():
    pdf_in = "/Users/macbookair/Downloads/DEFENSE/Mohsin Isfak.ORIGINAL_BACKUP.pdf"
    pdf_out = "/Users/macbookair/Downloads/DEFENSE/Mohsin Isfak.pdf"

    doc = pymupdf.open(pdf_in)
    print(f"Loaded {pdf_in} ({len(doc)} pages)")

    def replace_line_near_y(page, y_target, new_text, fontname="tiro", fontsize=11.0, x_start=None, y_tolerance=6.0):
        target_line = None
        for b in page.get_text("dict")["blocks"]:
            if "lines" in b:
                for l in b["lines"]:
                    y_mid = (l["bbox"][1] + l["bbox"][3]) / 2.0
                    if abs(y_mid - y_target) < y_tolerance:
                        target_line = l
                        break
            if target_line:
                break
        if not target_line:
            print(f"  [ERROR] No line found near y={y_target} on page {page.number + 1}")
            return False

        bbox = target_line["bbox"]
        x0 = x_start if x_start is not None else bbox[0]
        # Redact line with white rectangle
        redact_rect = pymupdf.Rect(bbox[0] - 1.0, bbox[1] - 1.5, bbox[2] + 1.5, bbox[3] + 1.5)
        page.add_redact_annot(redact_rect, fill=(1, 1, 1))
        page.apply_redactions()

        # Insert new text at line baseline
        baseline_y = bbox[3] - 2.8
        page.insert_text((x0, baseline_y), new_text, fontname=fontname, fontsize=fontsize)
        return True

    def redact_rect_and_insert(page, rect, new_text, fontname="tiro", fontsize=11.0, x_pos=None, y_pos=None):
        page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()
        x = x_pos if x_pos is not None else rect.x0
        y = y_pos if y_pos is not None else (rect.y1 - 2.8)
        page.insert_text((x, y), new_text, fontname=fontname, fontsize=fontsize)

    # 1. Page 3: Lecturer , -> Lecturer,; Daffodil University -> Daffodil International University
    p3 = doc[2]
    replace_line_near_y(
        p3, 155.1,
        "Shahriar Shakil, Lecturer, Department of Computer Science and Engineering, Daffodil",
        fontsize=11.0, x_start=85.25
    )
    replace_line_near_y(
        p3, 707.4,
        "Engineering Daffodil International University",
        fontsize=11.0, x_start=85.25
    )

    # 2. Page 4: Acknowledgements corrections
    p4 = doc[3]
    replace_line_near_y(
        p4, 201.0,
        "First, we express our heartfelt thanks to the almighty for His divine",
        fontsize=11.0, x_start=85.25
    )
    replace_line_near_y(
        p4, 217.4,
        "blessing making it possible for us to complete the Final Year Design Project (FYDP)",
        fontsize=11.0, x_start=85.25
    )
    replace_line_near_y(
        p4, 267.4,
        "We are grateful and express our profound gratitude to  Md. Shahriar Shakil, Lecturer,",
        fontsize=11.0, x_start=85.25
    )

    # 3. Page 6: TOC Chapter 2 & 3 page numbers
    p6 = doc[5]
    # Ch 2: page number '2' -> '5'
    r_ch2 = pymupdf.Rect(502.0, 451.0, 515.0, 465.0)
    redact_rect_and_insert(p6, r_ch2, "5", fontsize=11.0, x_pos=505.0, y_pos=461.5)
    # Ch 3: page number '4' -> '16'
    r_ch3 = pymupdf.Rect(500.0, 564.0, 515.0, 578.0)
    redact_rect_and_insert(p6, r_ch3, "16", fontsize=11.0, x_pos=500.0, y_pos=574.4)

    # 4. Page 7: TOC Chapter 4, 5, 6, References
    p7 = doc[6]
    # Ch 4: page number '6' -> '28'
    r_ch4 = pymupdf.Rect(500.0, 47.0, 515.0, 61.0)
    redact_rect_and_insert(p7, r_ch4, "28", fontsize=11.0, x_pos=500.0, y_pos=56.6)
    # Ch 5: page number '7' -> '49'
    r_ch5 = pymupdf.Rect(500.0, 142.0, 515.0, 156.0)
    redact_rect_and_insert(p7, r_ch5, "49", fontsize=11.0, x_pos=500.0, y_pos=152.4)
    # Ch 6: page number '10' -> '59'
    r_ch6 = pymupdf.Rect(495.0, 409.0, 515.0, 423.0)
    redact_rect_and_insert(p7, r_ch6, "59", fontsize=11.0, x_pos=497.75, y_pos=419.4)
    # Ref: page number '11' -> '64'
    r_ref = pymupdf.Rect(495.0, 488.0, 515.0, 502.0)
    redact_rect_and_insert(p7, r_ref, "64", fontsize=11.0, x_pos=497.75, y_pos=498.0)

    # 5. Page 8: List of Figures (Fig 4.3 -> 32, Fig 4.4 -> 39)
    p8 = doc[7]
    r_fig43 = pymupdf.Rect(494.0, 291.0, 515.0, 305.0)
    redact_rect_and_insert(p8, r_fig43, " 32 ", fontsize=11.0, x_pos=494.0, y_pos=300.55)
    r_fig44 = pymupdf.Rect(494.0, 323.0, 515.0, 337.0)
    redact_rect_and_insert(p8, r_fig44, " 39 ", fontsize=11.0, x_pos=494.0, y_pos=332.45)

    # 6. Page 9: List of Tables (2.2:, Table 4.15 -> 45, Table 4.16 -> 45, Table 4.17 -> 46, Table 5.0 -> 5.0:)
    p9 = doc[8]
    # 2.2  : -> 2.2:
    r_tab22 = pymupdf.Rect(72.0, 131.0, 100.0, 145.0)
    redact_rect_and_insert(p9, r_tab22, "2.2:", fontsize=11.0, x_pos=72.0, y_pos=140.6)
    # Table 4.15 page 46 -> 45
    r_tab415 = pymupdf.Rect(495.0, 627.0, 522.0, 641.0)
    redact_rect_and_insert(p9, r_tab415, "45", fontsize=11.0, x_pos=505.0, y_pos=636.1)
    # Table 4.16 page 16 -> 45
    r_tab416 = pymupdf.Rect(500.0, 654.0, 520.0, 668.0)
    redact_rect_and_insert(p9, r_tab416, "45", fontsize=11.0, x_pos=505.0, y_pos=663.9)
    # Table 4.17 page 47 -> 46
    r_tab417 = pymupdf.Rect(500.0, 682.0, 520.0, 696.0)
    redact_rect_and_insert(p9, r_tab417, "46", fontsize=11.0, x_pos=505.0, y_pos=691.7)
    # Table 5.0. -> 5.0:
    r_tab50 = pymupdf.Rect(72.0, 710.0, 95.0, 724.0)
    redact_rect_and_insert(p9, r_tab50, "5.0:", fontsize=11.0, x_pos=72.0, y_pos=719.5)

    # 7. Page 16: Literature Review phrasing
    p16 = doc[15]
    replace_line_near_y(
        p16, 121.5,
        "This literature review is divided into four thematic groups: psychological",
        fontsize=11.0, x_start=72.0
    )
    replace_line_near_y(
        p16, 182.3,
        "studies investigating the use of AI tools and digital behavior. The categories are not",
        fontsize=11.0, x_start=72.0
    )
    replace_line_near_y(
        p16, 197.4,
        "mutually exclusive, since several papers have been classified under more than one",
        fontsize=11.0, x_start=72.0
    )

    # 8. Page 40: "motivates using of the" -> "motivates the use of the"
    p40 = doc[39]
    replace_line_near_y(
        p40, 410.0,
        "around one-third reporting low or no anxiety. Consistent ≈67/33 split motivates the use of the",
        fontsize=11.0, x_start=72.0
    )

    # 9. Page 43: "the chance-level" -> "chance-level"
    p43 = doc[42]
    replace_line_near_y(
        p43, 111.9,
        "chance-level discrimination (ROC-AUC 0.513, MCC -0.053). This means that the seventeen",
        fontsize=11.0, x_start=72.0
    )

    # 10. Page 55: "essentially the chance-level" -> "essentially chance level"
    p55 = doc[54]
    replace_line_near_y(
        p55, 673.7,
        "essentially chance level), as seen at every experiment in this chapter.",
        fontsize=11.0, x_start=72.0
    )

    # 11. Page 57: "eleven-step pipeline" -> "eleven analytical steps"
    p57 = doc[56]
    replace_line_near_y(
        p57, 520.2,
        "Combining the eleven analytical steps, four consolidated conclusions could be drawn. First, the",
        fontsize=11.0, x_start=72.0
    )

    # 12. Page 58: "Chapter 6.re." -> "Chapter 6."
    p58 = doc[57]
    replace_line_near_y(
        p58, 292.2,
        "standards in Chapter 5, informed the conclusions and limitations in Chapter 6.",
        fontsize=11.0, x_start=72.0
    )

    # 13. Page 60: "bit-level reproducibility" -> reproducible execution
    p60 = doc[59]
    replace_line_near_y(
        p60, 458.4,
        "and tree ensemble models to assure reproducible execution under the specified environment.",
        fontsize=11.0, x_start=72.0
    )

    # 14. Page 61: "disruptions in higher education student careers"
    p61 = doc[60]
    replace_line_near_y(
        p61, 710.0,
        "concerns about changes in student career outcomes [1, 2]. Among the 2,036 analytical records",
        fontsize=11.0, x_start=72.0
    )

    # 15. Page 62: "suffers from the career despair and dropouts"
    p62 = doc[61]
    replace_line_near_y(
        p62, 144.1,
        "specific interventions and career resiliency workshops before the student encounters severe",
        fontsize=11.0, x_start=72.0
    )
    replace_line_near_y(
        p62, 159.9,
        "career distress [1, 4].",
        fontsize=11.0, x_start=72.0
    )

    # 16. Page 63: "strictly pinned"
    p63 = doc[62]
    replace_line_near_y(
        p63, 162.7,
        "minimum compatible versions are specified to ensure forward compatibility without costs.",
        fontsize=11.0, x_start=108.0
    )

    # 17. Page 65: "empirically established prevalence rate"
    p65 = doc[64]
    replace_line_near_y(
        p65, 593.9,
        "Grant Writing & Research Commercialization: The observed Class 1 proportion",
        fontsize=11.0, x_start=72.0
    )

    # 18. Page 68: "without causing any environmental pollution"
    p68 = doc[67]
    replace_line_near_y(
        p68, 269.2,
        "students with modest computational requirements without exposing participants' identities.",
        fontsize=11.0, x_start=72.0
    )

    # 19. Page 69: Conclusion summary updates
    p69 = doc[68]
    # experimental stage exclusion
    replace_line_near_y(
        p69, 474.1,
        "according to the predefined analytical population used in the research pipeline.",
        fontsize=11.0, x_start=72.0
    )
    # suffer from career anxiety -> classified as Class 1
    replace_line_near_y(
        p69, 565.2,
        "where 67.58% of analytical respondents were classified as Class 1 (Medium/High) under the",
        fontsize=11.0, x_start=72.0
    )
    replace_line_near_y(
        p69, 580.4,
        "study's binary target definition. A strict 17-variable analytical framework was designed for capturing",
        fontsize=11.0, x_start=72.0
    )
    # lexical regex parsers
    replace_line_near_y(
        p69, 656.3,
        "[8], [12], timeline of GenAI takeover [23], and threat perspective [13]. Regular-expression-",
        fontsize=11.0, x_start=72.0
    )
    replace_line_near_y(
        p69, 671.5,
        "based pattern matching helped to separate comma-separated lists of software tools for measuring",
        fontsize=11.0, x_start=72.0
    )
    # latent cognitive friction
    replace_line_near_y(
        p69, 717.0,
        "Canva, Photomath). In addition, three composite indicators were developed to measure",
        fontsize=11.0, x_start=72.0
    )
    replace_line_near_y(
        p69, 732.2,
        "structural interactions: Perceived Urgency (ai_replace_jobs * ai_takeover_time), Risk-Knowledge",
        fontsize=11.0, x_start=72.0
    )

    # 20. Page 71: Institutional comparison phrasing
    p71 = doc[70]
    replace_line_near_y(
        p71, 336.5,
        "uniformity of the observed Class 1 proportions in each university cohort. Public Universities",
        fontsize=11.0, x_start=72.0
    )
    replace_line_near_y(
        p71, 366.9,
        "had 69.64% classified as Class 1. These results provide empirical evidence that AI-induced",
        fontsize=11.0, x_start=72.0
    )
    replace_line_near_y(
        p71, 382.0,
        "occupational anxiety is not limited to some particular academic funding models or",
        fontsize=11.0, x_start=72.0
    )

    # 21. Page 73: Interventions
    p73 = doc[72]
    replace_line_near_y(
        p73, 421.1,
        "showing higher observed values of the engineered feature when the threat-knowledge gap is greatest, the",
        fontsize=11.0, x_start=108.0
    )

    doc.save(pdf_out)
    print(f"Saved updated PDF successfully to {pdf_out}")

if __name__ == "__main__":
    update_pdf()
