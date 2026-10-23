"""
Tests for app.ingestion.file_validation

Verifies:
- Filename sanitization against directory traversal
- Valid PDF magic bytes validation
- Valid TXT decoding and binary detection
- Size limits (empty files and files exceeding threshold)
- SHA-256 deduplication hashing
"""

import pytest

from app.ingestion.file_validation import (
    FileValidationError,
    sanitize_filename,
    validate_file,
)


def test_sanitize_filename_traversal():
    """Ensure path traversal attempts are stripped."""
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("normal_file.pdf") == "normal_file.pdf"


def test_sanitize_filename_empty():
    """Ensure empty or whitespace filename raises FileValidationError."""
    with pytest.raises(FileValidationError, match="cannot be empty"):
        sanitize_filename("   ")


def test_validate_unsupported_extension():
    """Ensure unsupported extensions (e.g. .exe, .py, .csv) are rejected."""
    with pytest.raises(FileValidationError, match="Unsupported file type"):
        validate_file("script.py", b"print('hello')", max_size_bytes=1000)


def test_validate_empty_file():
    """Ensure 0-byte files are rejected."""
    with pytest.raises(FileValidationError, match="File is empty"):
        validate_file("empty.txt", b"", max_size_bytes=1000)


def test_validate_file_exceeds_max_size():
    """Ensure files larger than max_size_bytes are rejected."""
    large_content = b"a" * 200
    with pytest.raises(FileValidationError, match="exceeds maximum allowed limit"):
        validate_file("big.txt", large_content, max_size_bytes=100)


def test_validate_valid_pdf_magic_bytes():
    """Valid PDF starting with %PDF- should pass."""
    pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    name, ext, sha = validate_file("test.pdf", pdf_content, max_size_bytes=5000)
    assert name == "test.pdf"
    assert ext == ".pdf"
    assert len(sha) == 64  # SHA-256 length


def test_validate_invalid_pdf_missing_header():
    """File named .pdf without %PDF- magic bytes must fail."""
    fake_pdf = b"This is not really a PDF file."
    with pytest.raises(FileValidationError, match="Missing standard '%PDF-' header"):
        validate_file("fake.pdf", fake_pdf, max_size_bytes=5000)


def test_validate_valid_text():
    """Valid UTF-8 text should pass."""
    txt_content = b"Hello, world! This is a test document."
    name, ext, sha = validate_file("notes.txt", txt_content, max_size_bytes=5000)
    assert name == "notes.txt"
    assert ext == ".txt"
    assert len(sha) == 64


def test_validate_binary_disguised_as_text():
    """Text file with embedded null bytes (binary) must fail."""
    binary_content = b"Some text\x00with null\x00bytes"
    with pytest.raises(FileValidationError, match="binary content detected"):
        validate_file("corrupt.txt", binary_content, max_size_bytes=5000)
