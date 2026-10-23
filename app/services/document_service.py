"""
CloudDocs AI — Document Service

Orchestrates the complete document ingestion lifecycle:
1. Validation (magic bytes, size, content hash)
2. Deduplication check
3. Raw file storage
4. Text extraction (PyMuPDF or TextLoader)
5. Chunking (deterministic IDs and page metadata)
6. Embedding generation (Sentence Transformers)
7. Vector storage (ChromaDB)
8. Metadata tracking (SQLite)
9. Rollback and cleanup on failure
"""

import logging
import uuid

import aiofiles

from app.config import Settings
from app.database.metadata import MetadataDB
from app.ingestion.chunker import chunk_text
from app.ingestion.file_validation import FileValidationError, validate_file
from app.ingestion.pdf_loader import PDFExtractionError, load_pdf_content
from app.ingestion.text_loader import TextExtractionError, load_text_content
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import VectorStore
from app.schemas.documents import (
    DeleteDocumentResponse,
    DocumentResponse,
    DocumentStatus,
)

logger = logging.getLogger("clouddocs.document_service")


class DocumentService:
    """Coordinates document ingestion, listing, and deletion across storage layers."""

    def __init__(
        self,
        settings: Settings,
        metadata_db: MetadataDB,
        vector_store: VectorStore,
        embedding_engine: EmbeddingEngine,
    ):
        self.settings = settings
        self.metadata_db = metadata_db
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine

    async def process_document_upload(
        self,
        filename: str,
        content: bytes,
    ) -> DocumentResponse:
        """
        Ingest an uploaded document file end-to-end.

        Args:
            filename: Raw name of uploaded file.
            content: Raw byte payload.

        Returns:
            DocumentResponse with status and chunk count.

        Raises:
            FileValidationError: On invalid file type, size, or magic bytes.
            PDFExtractionError: On unreadable or scanned PDFs.
            TextExtractionError: On unreadable text files.
        """
        # Step 1: Validate file content & compute SHA-256
        clean_filename, ext, content_hash = validate_file(
            filename=filename,
            content=content,
            max_size_bytes=self.settings.max_upload_size_bytes,
        )

        # Step 2: Duplicate check
        existing = self.metadata_db.get_document_by_hash(content_hash)
        if existing and existing.status == DocumentStatus.COMPLETED:
            logger.info(
                "Duplicate file upload detected for '%s' (matches doc '%s')",
                clean_filename,
                existing.id,
            )
            return existing

        # Step 3: Register document in SQLite as PROCESSING
        doc_id = str(uuid.uuid4())
        doc_record = self.metadata_db.create_document(
            doc_id=doc_id,
            filename=clean_filename,
            file_type=ext,
            file_size_bytes=len(content),
            content_hash=content_hash,
            status=DocumentStatus.PROCESSING,
        )

        # Step 4: Persist raw file to upload directory
        self.settings.upload_dir_path.mkdir(parents=True, exist_ok=True)
        saved_file_path = self.settings.upload_dir_path / f"{doc_id}_{clean_filename}"
        try:
            async with aiofiles.open(saved_file_path, "wb") as f:
                await f.write(content)
        except Exception as e:
            error_msg = f"Failed to save file to disk: {e}"
            logger.error(error_msg)
            self.metadata_db.update_status(doc_id, DocumentStatus.FAILED, error_message=error_msg)
            raise RuntimeError(error_msg) from e

        # Step 5: Extract text per page
        try:
            if ext == ".pdf":
                pages = load_pdf_content(content)
            elif ext == ".txt":
                pages = load_text_content(content)
            else:
                raise FileValidationError(f"Unsupported format {ext}")
        except (PDFExtractionError, TextExtractionError, FileValidationError) as e:
            logger.warning("Extraction failed for '%s': %s", clean_filename, e)
            self.metadata_db.update_status(doc_id, DocumentStatus.FAILED, error_message=str(e))
            raise
        except Exception as e:
            error_msg = f"Unexpected text extraction failure: {e}"
            logger.error(error_msg, exc_info=True)
            self.metadata_db.update_status(doc_id, DocumentStatus.FAILED, error_message=error_msg)
            raise RuntimeError(error_msg) from e

        # Step 6: Chunk extracted pages
        try:
            chunks = chunk_text(
                pages=pages,
                document_id=doc_id,
                filename=clean_filename,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            if not chunks:
                raise ValueError("No valid text chunks could be generated from document.")
        except Exception as e:
            error_msg = f"Chunking failed: {e}"
            logger.error(error_msg)
            self.metadata_db.update_status(doc_id, DocumentStatus.FAILED, error_message=error_msg)
            raise

        # Step 7 & 8: Generate embeddings and index into ChromaDB
        try:
            chunk_texts = [c.text for c in chunks]
            embeddings = self.embedding_engine.embed_texts(chunk_texts)

            self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)
        except Exception as e:
            error_msg = f"Embedding/Indexing failed: {e}"
            logger.error(error_msg, exc_info=True)
            # Cleanup any partially added vectors
            self.vector_store.delete_document_chunks(doc_id)
            self.metadata_db.update_status(doc_id, DocumentStatus.FAILED, error_message=error_msg)
            raise

        # Step 9: Mark COMPLETED in SQLite
        updated_doc = self.metadata_db.update_status(
            doc_id=doc_id,
            status=DocumentStatus.COMPLETED,
            chunk_count=len(chunks),
        )

        logger.info(
            "Document '%s' (ID: %s) successfully indexed with %d chunks.",
            clean_filename,
            doc_id,
            len(chunks),
        )
        return updated_doc or doc_record

    def list_documents(self) -> list[DocumentResponse]:
        """List all tracked documents."""
        return self.metadata_db.list_documents()

    def get_document(self, document_id: str) -> DocumentResponse | None:
        """Get document details by ID."""
        return self.metadata_db.get_document(document_id)

    def delete_document(self, document_id: str) -> DeleteDocumentResponse | None:
        """
        Delete a document and clean up all associated vectors and disk files.
        """
        doc = self.metadata_db.get_document(document_id)
        if not doc:
            return None

        # 1. Purge from ChromaDB
        chunks_deleted = self.vector_store.delete_document_chunks(document_id)

        # 2. Delete raw uploaded file if present
        saved_file = self.settings.upload_dir_path / f"{document_id}_{doc.filename}"
        if saved_file.exists():
            try:
                saved_file.unlink()
                logger.info("Deleted raw file %s", saved_file)
            except OSError as e:
                logger.warning("Could not delete file %s: %e", saved_file, e)

        # 3. Delete metadata record from SQLite
        self.metadata_db.delete_document(document_id)

        return DeleteDocumentResponse(
            document_id=document_id,
            filename=doc.filename,
            chunks_deleted=chunks_deleted,
            message=f"Document '{doc.filename}' and {chunks_deleted} vector records were deleted.",
        )
