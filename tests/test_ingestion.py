"""
Tests for app.ingestion.pdf_loader and app.ingestion.text_loader

Verifies:
- PDF text extraction with PyMuPDF
- Scanned / image-only PDF detection (honest error indicating OCR is required)
- Text file extraction with encoding fallback
- Empty text file handling
"""

import pymupdf
import pytest
from app.ingestion.pdf_loader import load_pdf_content, PDFExtractionError
from app.ingestion.text_loader import load_text_content, TextExtractionError


def _create_test_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate a minimal valid PDF in memory."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_pdf_loader_valid():
    """Verify text is extracted page by page from valid PDF."""
    pdf_bytes = _create_test_pdf([
        "Page one content with important facts.",
        "Page two content with secondary details.",
    ])

    pages = load_pdf_content(pdf_bytes)
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Page one content" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "Page two content" in pages[1]["text"]


def test_pdf_loader_scanned_detection():
    """PDF with no digital text (e.g. blank page or scanned image only) must raise clear OCR error."""
    # Blank PDF page without text
    doc = pymupdf.open()
    doc.new_page()
    blank_pdf = doc.write()
    doc.close()

    with pytest.raises(PDFExtractionError, match="Optical Character Recognition"):
        load_pdf_content(blank_pdf)


def test_text_loader_valid():
    """Verify plain text extraction from UTF-8 bytes."""
    raw_bytes = "Line 1: CloudDocs AI.\nLine 2: RAG architecture.".encode("utf-8")
    result = load_text_content(raw_bytes)

    assert len(result) == 1
    assert result[0]["page_number"] == 1
    assert "CloudDocs AI" in result[0]["text"]


def test_text_loader_empty():
    """Whitespace-only text file must raise TextExtractionError."""
    empty_bytes = b"   \n\t  "
    with pytest.raises(TextExtractionError, match="only whitespace or no readable characters"):
        load_text_content(empty_bytes)
