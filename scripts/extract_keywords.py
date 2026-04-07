"""Extract domain keywords from an NCERT PDF and save to the keyword store.

The script sends the full extracted text to the LLM in batches and asks it
to identify precise NCERT terminology — subject-specific nouns, process names,
organism names, mathematical terms, historical terms, etc. Results are saved
via the keyword store and can then be reviewed/edited via POST /keywords.

Usage:
    python scripts/extract_keywords.py \\
        --pdf  data/ncert_grade10_science.pdf \\
        --grade 10 \\
        --subject Science \\
        [--max-pages 50]        # optional page limit (default: all pages)

The extracted keyword list is printed and also saved to the in-process
keyword store. To persist across restarts wire the keyword store to Redis
(replace the in-memory dict in src/storage/keyword_store.py).
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.ingestion.pdf_parser import PDFParser
from src.llm.factory import get_llm_client
from src.storage.keyword_store import KeywordStore

# How many pages to send per LLM call — keeps prompts within context limits.
PAGES_PER_BATCH = 15

EXTRACTION_SYSTEM_PROMPT = """\
You are a curriculum expert specialising in CBSE/NCERT textbooks.
Your task is to extract a precise list of domain keywords from the provided textbook excerpt.

Extract ONLY terms that are:
- Subject-specific vocabulary (e.g. "photosynthesis", "osmosis", "polynomial", "nationalism")
- Named processes, phenomena, laws, or theorems (e.g. "Ohm's Law", "Krebs cycle")
- Important proper nouns: organisms, chemicals, historical figures, places relevant to the chapter
- NCERT-defined terms that students must know for CBSE exams

Do NOT include:
- Common English words ("the", "process", "important")
- Vague or overly broad terms ("biology", "science")
- Page numbers, captions, or figure labels

Return ONLY a comma-separated list of keywords, nothing else.
Example: chlorophyll, stomata, transpiration, xylem, phloem
"""


async def extract_from_batch(llm, batch_text: str, grade: str, subject: str) -> list[str]:
    prompt = (
        f"Grade: {grade}  |  Subject: {subject}\n\n"
        f"Textbook excerpt:\n{batch_text[:8000]}\n\n"
        "Extract keywords:"
    )
    raw = await llm.generate(system_prompt=EXTRACTION_SYSTEM_PROMPT, user_message=prompt)
    return [kw.strip() for kw in raw.split(",") if kw.strip()]


async def main():
    parser = argparse.ArgumentParser(description="Extract NCERT keywords from a PDF.")
    parser.add_argument("--pdf", required=True, help="Path to the NCERT PDF file")
    parser.add_argument("--grade", required=True, help='Grade number, e.g. "5" or "10"')
    parser.add_argument("--subject", required=True, help='Subject name, e.g. "Science"')
    parser.add_argument("--max-pages", type=int, default=0, help="Max pages to process (0 = all)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: file not found: {args.pdf}")
        sys.exit(1)

    print(f"\nParsing {args.pdf} ...")
    with open(args.pdf, "rb") as f:
        content = f.read()

    pdf_parser = PDFParser()
    pages = pdf_parser.parse(content)

    if args.max_pages:
        pages = pages[: args.max_pages]

    print(f"  {len(pages)} pages parsed.")

    llm = get_llm_client(settings.context_provider, settings.context_model)
    all_keywords: list[str] = []

    # Process in batches
    batches = [pages[i : i + PAGES_PER_BATCH] for i in range(0, len(pages), PAGES_PER_BATCH)]
    for idx, batch in enumerate(batches, 1):
        batch_text = "\n\n".join(p.text for p in batch if p.text.strip())
        print(f"  Batch {idx}/{len(batches)}: extracting keywords ...")
        kws = await extract_from_batch(llm, batch_text, args.grade, args.subject)
        all_keywords.extend(kws)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique_keywords: list[str] = []
    for kw in all_keywords:
        lower = kw.lower()
        if lower not in seen:
            seen.add(lower)
            unique_keywords.append(kw)

    print(f"\n  {len(unique_keywords)} unique keywords extracted.\n")
    print("Keywords:")
    print(", ".join(unique_keywords))

    # Save to keyword store
    store = KeywordStore()
    await store.save(args.grade, args.subject, unique_keywords)
    print(f"\nSaved to keyword store: key = '{args.grade}_{args.subject}'")
    print("To persist across restarts, POST to /keywords or wire the store to Redis.")


if __name__ == "__main__":
    asyncio.run(main())
