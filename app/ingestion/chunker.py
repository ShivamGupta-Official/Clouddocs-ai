"""
CloudDocs AI — Text Chunking Module

Splits extracted document pages into configurable chunks while preserving:
- Document ID
- Filename
- Page number
- Chunk index
- Character offsets

Guarantees:
- Step size (chunk_size - chunk_overlap) > 0, preventing infinite loops.
- No empty chunks generated.
- Small pages/documents produce single valid chunks without error.
- Deterministic chunk IDs: {doc_id}_p{page}_c{index}.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TextChunk:
    """Represents an extracted and segmented piece of text with citation metadata."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


def chunk_text(
    pages: list[dict[str, Any]],
    document_id: str,
    filename: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[TextChunk]:
    """
    Split pages of text into overlapping chunks.

    Operates per-page to preserve strict, truthful page-level citations.
    Attempts to break chunks on paragraph and sentence boundaries when possible.

    Args:
        pages: List of dicts with {"page_number": int, "text": str}.
        document_id: Unique UUID of the document.
        filename: Name of the original source document.
        chunk_size: Maximum characters per chunk (default: 800).
        chunk_overlap: Number of characters to overlap between adjacent chunks (default: 120).

    Returns:
        List of TextChunk objects with deterministic IDs and complete metadata.

    Raises:
        ValueError: If chunk_size <= chunk_overlap or chunk_size < 50.
    """
    if chunk_size < 50:
        raise ValueError(f"chunk_size must be at least 50 characters, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    step = chunk_size - chunk_overlap
    chunks: list[TextChunk] = []
    global_chunk_idx = 0

    for page_info in pages:
        page_num = page_info["page_number"]
        page_text = page_info["text"].strip()

        if not page_text:
            continue

        text_len = len(page_text)

        # If page is shorter than chunk size, emit as a single chunk
        if text_len <= chunk_size:
            chunk_id = f"{document_id}_p{page_num}_c{global_chunk_idx}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    page_number=page_num,
                    chunk_index=global_chunk_idx,
                    text=page_text,
                    start_char=0,
                    end_char=text_len,
                )
            )
            global_chunk_idx += 1
            continue

        # Page exceeds chunk_size: iterate with sliding window
        start = 0
        while start < text_len:
            end = min(start + chunk_size, text_len)
            candidate = page_text[start:end]

            # If not at the end of text, attempt to find a natural break point
            if end < text_len:
                # Look for paragraph break \n\n in the last 25% of candidate
                search_start = int(len(candidate) * 0.75)
                para_break = candidate.rfind("\n\n", search_start)
                if para_break != -1:
                    end = start + para_break + 2
                else:
                    # Look for sentence break (. ) or newline (\n)
                    sentence_break = max(
                        candidate.rfind(". ", search_start),
                        candidate.rfind(".\n", search_start),
                        candidate.rfind("\n", search_start),
                    )
                    if sentence_break != -1:
                        end = start + sentence_break + 1

            chunk_content = page_text[start:end].strip()

            if chunk_content:
                chunk_id = f"{document_id}_p{page_num}_c{global_chunk_idx}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        page_number=page_num,
                        chunk_index=global_chunk_idx,
                        text=chunk_content,
                        start_char=start,
                        end_char=end,
                    )
                )
                global_chunk_idx += 1

            # Advance window by step size or break point
            if end == text_len:
                break
            start += step

            # Defensive safety guard against infinite loops
            if start >= end and start < text_len:
                start = end

    return chunks
