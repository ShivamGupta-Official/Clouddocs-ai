"""
Integration tests for document ingestion and management API endpoints:
- POST /documents (PDF and TXT upload, validation, duplicate handling)
- GET /documents (list all documents)
- GET /documents/{id} (document details)
- DELETE /documents/{id} (deletion and vector cleanup)
"""

import io
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _generate_test_pdf_bytes(title: str, body: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), f"{title}\n\n{body}")
    data = doc.write()
    doc.close()
    return data


def test_upload_valid_txt_document():
    """Uploading a valid text file should extract text, chunk, embed, and store."""
    content = "CloudDocs AI provides truthful, verifiable document question-answering."
    files = {"file": ("test_doc.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}

    response = client.post("/documents", files=files)
    assert response.status_code == 201
    data = response.json()

    assert data["filename"] == "test_doc.txt"
    assert data["file_type"] == ".txt"
    assert data["status"] == "COMPLETED"
    assert data["chunk_count"] >= 1
    assert "id" in data

    doc_id = data["id"]

    # Verify document appears in GET /documents
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    docs = list_resp.json()["documents"]
    assert any(d["id"] == doc_id for d in docs)

    # Clean up
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["chunks_deleted"] >= 1


def test_upload_duplicate_detection():
    """Uploading identical content should return existing document without re-indexing."""
    content = "Identical content for duplicate verification."
    files1 = {"file": ("dup1.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}
    resp1 = client.post("/documents", files=files1)
    assert resp1.status_code == 201
    doc1 = resp1.json()

    # Second upload of exact same content
    files2 = {"file": ("dup2.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}
    resp2 = client.post("/documents", files=files2)
    assert resp2.status_code == 201 or resp2.status_code == 200
    doc2 = resp2.json()

    # Must match original document ID
    assert doc1["id"] == doc2["id"]

    # Clean up
    client.delete(f"/documents/{doc1['id']}")


def test_upload_invalid_file_extension():
    """Unsupported file extension must return 400 Bad Request."""
    files = {"file": ("code.py", io.BytesIO(b"print(1)"), "text/x-python")}
    response = client.post("/documents", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_empty_file():
    """Empty 0-byte file must return 400 Bad Request."""
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    response = client.post("/documents", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_valid_pdf_and_delete():
    """Upload valid PDF, verify chunking and ChromaDB storage, then delete."""
    pdf_bytes = _generate_test_pdf_bytes(
        title="Chapter 1: Cloud Architecture",
        body="Amazon Web Services EC2 instances provide scalable compute capacity in the cloud.",
    )
    files = {"file": ("aws_guide.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

    response = client.post("/documents", files=files)
    assert response.status_code == 201
    data = response.json()

    assert data["filename"] == "aws_guide.pdf"
    assert data["file_type"] == ".pdf"
    assert data["status"] == "COMPLETED"
    assert data["chunk_count"] >= 1
    doc_id = data["id"]

    # GET document details
    get_resp = client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == doc_id

    # DELETE document
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["chunks_deleted"] >= 1

    # Verify 404 after deletion
    get_deleted = client.get(f"/documents/{doc_id}")
    assert get_deleted.status_code == 404
