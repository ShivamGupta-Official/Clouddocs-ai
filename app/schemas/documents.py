"""
CloudDocs AI — Document Schemas

Pydantic models for document ingestion, management, and API responses.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Processing lifecycle status of a document."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DocumentResponse(BaseModel):
    """Schema for document metadata returned by API."""
    id: str = Field(..., description="Unique UUID of the document")
    filename: str = Field(..., description="Original name of the uploaded file")
    file_type: str = Field(..., description="Extension or MIME type (.pdf, .txt)")
    file_size_bytes: int = Field(..., description="Size of file in bytes")
    status: DocumentStatus = Field(..., description="Current processing status")
    chunk_count: int = Field(default=0, description="Number of extracted text chunks")
    error_message: Optional[str] = Field(default=None, description="Error details if processing failed")
    content_hash: Optional[str] = Field(default=None, description="SHA-256 hash of file content")
    created_at: datetime = Field(..., description="Timestamp when file was uploaded")
    updated_at: datetime = Field(..., description="Timestamp of last status update")


class DocumentListResponse(BaseModel):
    """Response schema for listing multiple documents."""
    documents: list[DocumentResponse]
    total: int


class DocumentDetailResponse(DocumentResponse):
    """Detailed document response including chunk information."""
    page_count: Optional[int] = Field(default=None, description="Total pages if PDF")


class DeleteDocumentResponse(BaseModel):
    """Response returned after deleting a document."""
    document_id: str
    filename: str
    chunks_deleted: int
    message: str
