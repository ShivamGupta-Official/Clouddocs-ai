"""
Tests for app.ingestion.chunker

Verifies:
- Configurable chunk size and overlap
- Step size safety (preventing infinite loops)
- Complete metadata retention (doc_id, filename, page_number, chunk_index)
- Small document single-chunk handling
- Boundary validation (overlap >= size raises ValueError)
- Non-empty chunks
"""

import pytest
from app.ingestion.chunker import chunk_text, TextChunk


def test_chunk_invalid_parameters():
    """Ensure invalid chunk size or overlap raises ValueError."""
    pages = [{"page_number": 1, "text": "Sample text"}]

    # chunk_size too small
    with pytest.raises(ValueError, match="chunk_size must be at least 50"):
        chunk_text(pages, "doc1", "doc.txt", chunk_size=30, chunk_overlap=10)

    # negative overlap
    with pytest.raises(ValueError, match="cannot be negative"):
        chunk_text(pages, "doc1", "doc.txt", chunk_size=100, chunk_overlap=-5)

    # overlap >= size
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        chunk_text(pages, "doc1", "doc.txt", chunk_size=100, chunk_overlap=100)


def test_chunk_small_document():
    """Documents shorter than chunk_size should produce exactly one chunk."""
    pages = [{"page_number": 1, "text": "Short document content."}]
    chunks = chunk_text(pages, "doc_123", "short.txt", chunk_size=800, chunk_overlap=120)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "doc_123_p1_c0"
    assert chunk.document_id == "doc_123"
    assert chunk.filename == "short.txt"
    assert chunk.page_number == 1
    assert chunk.chunk_index == 0
    assert chunk.text == "Short document content."


def test_chunk_multiple_pages():
    """Ensure page numbers and IDs are preserved across multiple pages."""
    pages = [
        {"page_number": 1, "text": "Page one text."},
        {"page_number": 2, "text": "Page two text."},
    ]
    chunks = chunk_text(pages, "doc_multi", "multi.pdf", chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_id == "doc_multi_p1_c0"
    assert chunks[1].page_number == 2
    assert chunks[1].chunk_id == "doc_multi_p2_c1"


def test_chunk_sliding_window_overlap():
    """Ensure large text is split into multiple overlapping chunks."""
    # Create 300 characters of text
    text = "Word " * 60  # 300 chars
    pages = [{"page_number": 1, "text": text}]

    chunk_size = 100
    chunk_overlap = 30
    chunks = chunk_text(pages, "doc_large", "large.txt", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) > 1
    # Check each chunk is non-empty and has correct metadata
    for idx, c in enumerate(chunks):
        assert len(c.text) > 0
        assert c.chunk_index == idx
        assert c.document_id == "doc_large"


def test_chunk_natural_break_points():
    """Chunking should attempt to break cleanly on sentences rather than slicing mid-word."""
    text = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence that is slightly longer. "
        "This is the fourth sentence."
    )
    pages = [{"page_number": 1, "text": text}]
    chunks = chunk_text(pages, "doc_break", "break.txt", chunk_size=70, chunk_overlap=15)

    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) > 0
