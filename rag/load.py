"""STEP 1 - LOAD: turn a PDF into plain text, one record per page.

RAG can only be as good as the text you feed it. If the PDF text comes out
garbled here (columns mixed together, missing words), every later step suffers.
Always eyeball this output first.

We use pypdf's "layout" mode, which reads text in visual order (top to bottom,
as it looks on the page). The default mode follows the PDF's internal drawing
order, which for many resume templates puts headings and their content in the
wrong places (e.g. the summary paragraph ended up under "EDUCATION").
"""
import re
from pathlib import Path

from pypdf import PdfReader

from rag import DATA_DIR


def load_pdf(path: Path) -> list[dict]:
    """Return [{"page": 1, "text": "..."}, ...] for every page with text."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text(extraction_mode="layout") or ""
        # Layout mode pads with lots of spaces to mimic positions; tidy that up.
        lines = [re.sub(r" {2,}", " ", line).strip() for line in raw.splitlines()]
        text = "\n".join(line for line in lines if line)
        if text:
            pages.append({"page": i, "text": text})
    return pages


def find_pdfs() -> list[Path]:
    return sorted(DATA_DIR.glob("*.pdf"))


if __name__ == "__main__":
    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit(f"No PDFs found. Put your resume in {DATA_DIR}")
    for pdf in pdfs:
        print(f"===== {pdf.name} =====")
        for p in load_pdf(pdf):
            print(f"--- page {p['page']} ({len(p['text'])} chars) ---")
            print(p["text"])
