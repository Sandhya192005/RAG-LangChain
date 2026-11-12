"""STEP 2 - CHUNK: split text into small, self-contained pieces.

Why chunk at all?
  * Embeddings summarise a whole text into ONE vector. A vector for a whole
    resume is a blurry average; a vector for just the "Skills" section is sharp.
  * We only want to send the *relevant* parts to the LLM, not everything.

Trade-offs to experiment with:
  * CHUNK_SIZE too big  -> retrieval is fuzzy, prompts get long.
  * CHUNK_SIZE too small -> chunks lose context ("Python" without "5 years").
  * OVERLAP repeats a little text between neighbouring chunks so a sentence
    cut at a boundary still appears whole in at least one chunk.

Resumes have clear sections, so we split on headings first, then by size.
"""
import re

CHUNK_SIZE = 500   # characters
OVERLAP = 80       # characters

SECTION_WORDS = {
    "summary", "profile", "objective", "about", "about me", "profile summary",
    "professional summary", "career objective", "career summary",
    "experience", "work experience", "professional experience", "employment",
    "education", "skills", "technical skills", "projects", "certifications",
    "achievements", "awards", "publications", "languages", "interests",
    "internships", "internship", "activities", "volunteering", "contact",
}


def is_known_section(line: str) -> bool:
    return line.strip().rstrip(":").strip().lower() in SECTION_WORDS


def is_heading(line: str) -> bool:
    clean = line.strip().rstrip(":").strip()
    if not clean or len(clean) > 40:
        return False
    if is_known_section(clean):
        return True
    # Short ALL-CAPS lines like "TECHNICAL SKILLS" are usually headings.
    letters = re.sub(r"[^A-Za-z]", "", clean)
    return len(letters) >= 4 and letters.isupper() and len(clean.split()) <= 4


def split_sections(pages: list[dict]) -> list[dict]:
    """Group lines under the most recent heading -> [{section, page, text}].

    Everything before the first real section (name, title, contact details) goes
    into a "Header" section. Resumes often put the NAME and JOB TITLE in all caps,
    so we don't let the ALL-CAPS rule fire until a known section has been seen.
    """
    sections = []
    current = {"section": "Header", "page": pages[0]["page"] if pages else 1, "lines": []}
    seen_section = False
    for p in pages:
        for line in p["text"].splitlines():
            if is_known_section(line):
                seen_section = True
            if seen_section and is_heading(line):
                if current["lines"]:
                    sections.append(current)
                current = {"section": line.strip().rstrip(":").title(), "page": p["page"], "lines": []}
            elif line.strip():
                current["lines"].append(line.strip())
    if current["lines"]:
        sections.append(current)
    return [{"section": s["section"], "page": s["page"], "text": "\n".join(s["lines"])} for s in sections]


def split_by_size(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Sliding window over the text, preferring to cut at a newline or space."""
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Back up to a natural boundary so we don't cut words in half.
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
        # Move the overlap start forward to the next word boundary.
        nxt = min((i for i in (text.find(" ", start, end), text.find("\n", start, end)) if i != -1), default=-1)
        if nxt != -1:
            start = nxt + 1
    return [p for p in pieces if p]


def chunk_pages(pages: list[dict], source: str = "") -> list[dict]:
    chunks = []
    for sec in split_sections(pages):
        for piece in split_by_size(sec["text"]):
            chunks.append({
                "id": len(chunks),
                "source": source,
                "section": sec["section"],
                "page": sec["page"],
                # Prefix the section name: it gives the embedding useful context.
                "text": f"[{sec['section']}]\n{piece}",
            })
    return chunks


if __name__ == "__main__":
    from rag.load import find_pdfs, load_pdf

    for pdf in find_pdfs():
        for c in chunk_pages(load_pdf(pdf), pdf.name):
            print(f"--- chunk {c['id']} | section={c['section']} | page={c['page']} | {len(c['text'])} chars ---")
            print(c["text"])
