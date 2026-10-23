"""
CloudDocs AI — SQLite Document Metadata Storage

Provides persistent relational storage for tracking document metadata,
processing status, chunk counts, and SHA-256 hashes for deduplication.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.schemas.documents import DocumentResponse, DocumentStatus


class MetadataDB:
    """Manages SQLite database connections and document CRUD operations."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with dict-like row access."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialize database tables and indices."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    content_hash TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_created ON documents(created_at)")
            conn.commit()

    def create_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        content_hash: str,
        status: DocumentStatus = DocumentStatus.PENDING,
    ) -> DocumentResponse:
        """Insert a new document record into SQLite."""
        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, filename, file_type, file_size_bytes, status,
                    chunk_count, error_message, content_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    filename,
                    file_type,
                    file_size_bytes,
                    status.value,
                    0,
                    None,
                    content_hash,
                    now,
                    now,
                ),
            )
            conn.commit()

        return self.get_document(doc_id)  # type: ignore

    def get_document(self, doc_id: str) -> DocumentResponse | None:
        """Fetch a document by its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_response(row)

    def get_document_by_hash(self, content_hash: str) -> DocumentResponse | None:
        """Check for existing duplicate document by SHA-256 hash."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? AND status != ?",
                (content_hash, DocumentStatus.FAILED.value),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_response(row)

    def list_documents(self) -> list[DocumentResponse]:
        """List all documents ordered by creation time descending."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [self._row_to_response(row) for row in rows]

    def update_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> DocumentResponse | None:
        """Update processing status, chunk count, or error message of a document."""
        now = datetime.now(UTC).isoformat()
        updates = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status.value, now]

        if chunk_count is not None:
            updates.append("chunk_count = ?")
            params.append(chunk_count)

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(doc_id)

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )
            conn.commit()

        return self.get_document(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document record by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics across indexed documents."""
        with self._get_connection() as conn:
            total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            completed_docs = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE status = ?",
                (DocumentStatus.COMPLETED.value,),
            ).fetchone()[0]
            failed_docs = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE status = ?",
                (DocumentStatus.FAILED.value,),
            ).fetchone()[0]
            total_chunks = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM documents WHERE status = ?",
                (DocumentStatus.COMPLETED.value,),
            ).fetchone()[0]

            return {
                "total_documents": total_docs,
                "completed_documents": completed_docs,
                "failed_documents": failed_docs,
                "total_chunks": total_chunks,
            }

    @staticmethod
    def _row_to_response(row: sqlite3.Row) -> DocumentResponse:
        return DocumentResponse(
            id=row["id"],
            filename=row["filename"],
            file_type=row["file_type"],
            file_size_bytes=row["file_size_bytes"],
            status=DocumentStatus(row["status"]),
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
            content_hash=row["content_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
