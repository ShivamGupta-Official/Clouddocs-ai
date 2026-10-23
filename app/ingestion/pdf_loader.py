"""
CloudDocs AI — PDF Loader

Extracts text from PDF documents page-by-page using PyMuPDF (fitz).
Preserves page-level metadata for citations and handles edge cases:
- Scanned / image-only PDFs (raises clear OCR recommendation)
- Password-protected / encrypted PDFs
- Corrupted documents
"""

from typing import Any

import pymupdf


class PDFExtractionError(RuntimeError):
    """Raised when PDF extraction fails or document contains no extractable text."""

    pass


def load_pdf_content(content_bytes: bytes) -> list[dict[str, Any]]:
    """
    Extract text page-by-page from raw PDF bytes.

    Args:
        content_bytes: Raw bytes of the PDF file.

    Returns:
        List of dicts per page:
        [
            {"page_number": 1, "text": "Page 1 content..."},
            {"page_number": 2, "text": "Page 2 content..."},
        ]

    Raises:
        PDFExtractionError: If document is corrupted, password-protected, or scanned/empty.
    """
    try:
        doc = pymupdf.open(stream=content_bytes, filetype="pdf")
    except Exception as e:
        raise PDFExtractionError(f"Corrupted or invalid PDF document: {e}") from e

    try:
        if doc.is_encrypted:
            raise PDFExtractionError(
                "PDF is password-protected or encrypted. Encrypted documents cannot be indexed."
            )

        if doc.page_count == 0:
            raise PDFExtractionError("PDF contains 0 pages.")

        pages_data: list[dict[str, Any]] = []
        total_text_chars = 0

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            page_text = page.get_text("text").strip()
            page_num = page_idx + 1

            if page_text:
                total_text_chars += len(page_text)
                pages_data.append(
                    {
                        "page_number": page_num,
                        "text": page_text,
                    }
                )

        # Check for scanned / image-only PDF
        if total_text_chars == 0 or len(pages_data) == 0:
            raise PDFExtractionError(
                "Scanned or image-only PDF detected: no extractable digital text found. "
                "Optical Character Recognition (OCR) is required to index this document."
            )

        return pages_data

    finally:
        doc.close()
