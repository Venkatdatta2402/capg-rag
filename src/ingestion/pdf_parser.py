"""PDF parsing for NCERT/CBSE knowledge base documents.

Section and subsection detection:
  NCERT books use decimal numbering for sections (e.g. "5.1 The Root")
  and subsections (e.g. "5.1.1 Root Hair"). This parser detects these
  patterns line-by-line and attaches them as page metadata so the
  hierarchical retriever can narrow scope precisely.
"""

import io
import re

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# Matches NCERT-style headings at the start of a line:
#   "5.1 The Root"           → section
#   "5.1.1 Root Hair Cells"  → subsection
# Allows optional leading whitespace and requires the heading to have at
# least one word of title text after the number.
_SUBSECTION_RE = re.compile(
    r"^\s*(\d{1,2}\.\d{1,2}\.\d{1,2})\s+([A-Z][^\n]{2,60})", re.MULTILINE
)
_SECTION_RE = re.compile(
    r"^\s*(\d{1,2}\.\d{1,2})\s+([A-Z][^\n]{2,60})", re.MULTILINE
)


class ParsedPage(BaseModel):
    """A single page extracted from a PDF."""

    page_number: int
    text: str
    metadata: dict = Field(default_factory=dict)


class PDFParser:
    """Extracts text and NCERT structural metadata from PDF documents."""

    def parse(self, content: bytes) -> list[ParsedPage]:
        """Parse PDF bytes into a list of pages with section metadata.

        Uses pypdf for text extraction. Section and subsection headings are
        detected via regex on the extracted text and carried forward across
        pages until a new heading overwrites them (NCERT sections span pages).
        """
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(content))
        pages: list[ParsedPage] = []

        # Carry the last seen section/subsection forward across page boundaries
        current_section_number = ""
        current_section_title = ""
        current_subsection_number = ""
        current_subsection_title = ""

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            # Detect subsection headings first (more specific, e.g. "5.1.1 ...")
            for match in _SUBSECTION_RE.finditer(text):
                current_subsection_number = match.group(1)
                current_subsection_title = match.group(2).strip()
                # A new subsection is inside a section — derive section from prefix
                parts = current_subsection_number.split(".")
                current_section_number = f"{parts[0]}.{parts[1]}"

            # Detect section headings ("5.1 ...") — only update if it doesn't
            # conflict with a more specific subsection already found on this page
            for match in _SECTION_RE.finditer(text):
                num = match.group(1)
                title = match.group(2).strip()
                # Accept if this section is a parent of (or equal to) current subsection
                if not current_subsection_number.startswith(num):
                    current_section_number = num
                    current_section_title = title
                    # A new section resets subsection (we've moved to a new section)
                    if not _SUBSECTION_RE.search(text):
                        current_subsection_number = ""
                        current_subsection_title = ""

            pages.append(ParsedPage(
                page_number=i + 1,
                text=text,
                metadata={
                    "section_number": current_section_number,
                    "section_title": current_section_title,
                    "subsection_number": current_subsection_number,
                    "subsection_title": current_subsection_title,
                },
            ))

        logger.info(
            "pdf_parser.parsed",
            pages=len(pages),
            sections_detected=len({
                p.metadata["section_number"] for p in pages
                if p.metadata["section_number"]
            }),
        )
        return pages
