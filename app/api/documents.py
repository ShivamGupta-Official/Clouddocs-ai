"""
CloudDocs AI — Document Management API Endpoints

Provides endpoints for:
- POST /documents: Upload and index PDF/TXT files
- GET /documents: List all tracked documents and processing status
- GET /documents/{document_id}: Retrieve document metadata
- DELETE /documents/{document_id}: Delete document and clean up vectors
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.database.metadata import MetadataDB
from app.ingestion.file_validation import FileValidationError
from app.ingestion.pdf_loader import PDFExtractionError
from app.ingestion.text_loader import TextExtractionError
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import VectorStore
from app.schemas.documents import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.document_service import DocumentService

logger = logging.getLogger("clouddocs.api.documents")

router = APIRouter(prefix="/documents", tags=["Documents"])

# Global singletons for services, initialized once
_metadata_db: MetadataDB | None = None
_vector_store: VectorStore | None = None
_embedding_engine: EmbeddingEngine | None = None


def get_metadata_db(settings: Settings = Depends(get_settings)) -> MetadataDB:
    global _metadata_db
    if _metadata_db is None:
        _metadata_db = MetadataDB(settings.sqlite_db_path)
    return _metadata_db


def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(settings.chroma_dir_path)
    return _vector_store


def get_embedding_engine(settings: Settings = Depends(get_settings)) -> EmbeddingEngine:
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine(settings.embedding_model)
    return _embedding_engine


def get_document_service(
    settings: Settings = Depends(get_settings),
    metadata_db: MetadataDB = Depends(get_metadata_db),
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
) -> DocumentService:
    return DocumentService(
        settings=settings,
        metadata_db=metadata_db,
        vector_store=vector_store,
        embedding_engine=embedding_engine,
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or TXT document to upload and index"),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """
    Upload a document (PDF or TXT), validate its contents, extract text,
    generate chunks, embed them, and index into ChromaDB.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    try:
        content = await file.read()
        document = await service.process_document_upload(
            filename=file.filename,
            content=content,
        )
        return document
    except (FileValidationError, PDFExtractionError, TextExtractionError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Failed to process document upload: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the document: {e}",
        ) from e


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
async def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List all documents in the system with their status and chunk counts."""
    docs = service.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Fetch status and metadata for a specific document by its ID."""
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return doc


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document and its vector embeddings",
)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DeleteDocumentResponse:
    """
    Delete a document from metadata storage, remove its associated chunks
    from ChromaDB, and remove the raw file from disk.
    """
    result = service.delete_document(document_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return result
