"""
CloudDocs AI — Plain Text Loader

Reads and extracts text from plain text files (.txt) with robust encoding detection.
Returns page-level records for consistent downstream chunking and citation tracking.
"""

from typing import List, Dict, Any


class TextExtractionError(RuntimeError):
    """Raised when text cannot be extracted from a document."""
    pass


def load_text_content(content_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text from raw text file bytes.

    Tries UTF-8 decoding first (with BOM handling), falling back to Latin-1.
    Ensures that extracted text is non-empty.

    Args:
        content_bytes: Raw bytes of the text file.

    Returns:
        List of dicts representing pages/sections:
        [
            {"page_number": 1, "text": "Extracted text content..."}
        ]

    Raises:
        TextExtractionError: If text is empty or cannot be read.
    """
    text = ""
    # Try common encodings
    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            text = content_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if not text:
        raise TextExtractionError("Unable to decode text file with supported encodings.")

    cleaned_text = text.strip()
    if not cleaned_text:
        raise TextExtractionError(
            "Text file contains only whitespace or no readable characters."
        )

    return [{"page_number": 1, "text": cleaned_text}]
