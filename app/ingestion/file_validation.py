"""
CloudDocs AI — File Validation Module

Validates uploaded files for:
- Supported extensions (.pdf, .txt)
- Magic bytes and content signatures (not just extensions)
- File size boundaries (non-empty, within max upload limit)
- Filename sanitation against path traversal attacks
- SHA-256 hash computation for duplicate detection
"""

import hashlib
import os
import re
from pathlib import Path


class FileValidationError(ValueError):
    """Raised when an uploaded file fails validation checks."""

    pass


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
PDF_MAGIC_BYTES = b"%PDF-"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and security issues.

    Removes path separators, null bytes, leading dots, and unsafe characters.
    Ensures safe storage on the local filesystem.
    """
    if not filename or not filename.strip():
        raise FileValidationError("Filename cannot be empty.")

    # Normalize backslashes (Windows paths) to forward slashes so basename works on all OS
    normalized = filename.strip().replace("\\", "/")
    clean_name = os.path.basename(normalized)

    # Replace null bytes and dangerous chars
    clean_name = clean_name.replace("\x00", "")

    # Remove characters that are problematic across operating systems
    # Allow alphanumeric, spaces, hyphens, underscores, dots
    clean_name = re.sub(r"[^a-zA-Z0-9_\-. ]", "_", clean_name)
    clean_name = clean_name.strip(". ")

    if not clean_name:
        raise FileValidationError("Invalid filename after sanitization.")

    return clean_name


def validate_file(
    filename: str,
    content: bytes,
    max_size_bytes: int,
) -> tuple[str, str, str]:
    """
    Validate file content, size, and type.

    Args:
        filename: Name of the uploaded file.
        content: Raw bytes of the uploaded file.
        max_size_bytes: Maximum allowed size in bytes.

    Returns:
        Tuple of (clean_filename, extension, sha256_hash)

    Raises:
        FileValidationError: If any validation rule fails.
    """
    # 1. Sanitize filename
    clean_name = sanitize_filename(filename)
    ext = Path(clean_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # 2. Check size bounds
    content_len = len(content)
    if content_len == 0:
        raise FileValidationError("File is empty (0 bytes). Cannot process empty documents.")

    if content_len > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = content_len / (1024 * 1024)
        raise FileValidationError(
            f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit ({max_mb:.0f} MB)."
        )

    # 3. Magic byte and content inspection
    if ext == ".pdf":
        # Valid PDFs must begin with %PDF- header (typically within first 1024 bytes)
        header = content[:1024]
        if PDF_MAGIC_BYTES not in header:
            raise FileValidationError(
                "Invalid PDF file: Missing standard '%PDF-' header signature."
            )
    elif ext == ".txt":
        # Check that it's readable text and not binary disguised as .txt
        try:
            # Check if decodable with utf-8 (or utf-8-sig)
            content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Fallback check for latin-1
                content.decode("latin-1")
            except Exception as e:
                raise FileValidationError(
                    f"Invalid text file: unable to decode text encoding ({e})."
                ) from e

        # Disallow null bytes in text files (strong indicator of binary file)
        if b"\x00" in content:
            raise FileValidationError("Text file contains null bytes (binary content detected).")

    # 4. Compute SHA-256 hash for deduplication
    content_hash = hashlib.sha256(content).hexdigest()

    return clean_name, ext, content_hash
