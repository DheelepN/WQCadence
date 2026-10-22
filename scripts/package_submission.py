"""Collect the files for the journal upload into paper/submission/, named as the
Guide for Authors asks (Figure_1, Figure_2, ...; "highlights" in the file name).

    python scripts/package_submission.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "submission"

FIGURES = ["fig1_owenmore_data", "fig2_policy_example", "fig3_recall_cost", "fig4_threshold_sweep",
           "fig5_fault_classifier", "fig6_localization"]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    copies = {
        ROOT / "paper" / "WQCadence_manuscript.docx": "Manuscript.docx",
        ROOT / "paper" / "highlights.docx": "Highlights.docx",
        ROOT / "paper" / "cover_letter.md": "Cover_letter.txt",
        ROOT / "figures" / "graphical_abstract.pdf": "Graphical_abstract.pdf",
        ROOT / "figures" / "graphical_abstract.tiff": "Graphical_abstract.tiff",
    }
    for i, stem in enumerate(FIGURES, 1):
        copies[ROOT / "figures" / f"{stem}.pdf"] = f"Figure_{i}.pdf"
    for src, name in copies.items():
        shutil.copy2(src, OUT / name)
        print(f"{name:26s} <- {src.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
