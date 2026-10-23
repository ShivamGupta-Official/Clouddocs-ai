"""
CloudDocs AI — Frontend API Client

Handles HTTP communication between the Streamlit UI and the FastAPI backend.
Translates network errors into user-friendly diagnostic messages.
"""

from typing import Dict, Any, Optional, List
import requests


class CloudDocsAPIClient:
    """Client for invoking CloudDocs AI FastAPI backend endpoints."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def get_health(self) -> Dict[str, Any]:
        """Check backend health and component readiness."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {
                "success": False,
                "error": f"Backend returned HTTP {resp.status_code}: {resp.text}",
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Cannot connect to backend server at {self.base_url}. Ensure the FastAPI server is running.",
            }
        except Exception as e:
            return {"success": False, "error": f"Health check failed: {e}"}

    def list_documents(self) -> Dict[str, Any]:
        """Fetch all documents and processing statuses."""
        try:
            resp = requests.get(f"{self.base_url}/documents", timeout=10)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json().get("documents", [])}
            return {
                "success": False,
                "error": f"Error fetching documents: HTTP {resp.status_code}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def upload_document(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Upload a PDF or TXT document for extraction, chunking, and indexing."""
        try:
            files = {"file": (filename, content)}
            resp = requests.post(f"{self.base_url}/documents", files=files, timeout=60)
            if resp.status_code in (200, 201):
                return {"success": True, "data": resp.json()}
            error_detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            return {"success": False, "error": error_detail}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Upload timed out while processing embeddings."}
        except Exception as e:
            return {"success": False, "error": f"Upload failed: {e}"}

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document and purge its vectors."""
        try:
            resp = requests.delete(f"{self.base_url}/documents/{document_id}", timeout=10)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {"success": False, "error": resp.json().get("detail", resp.text)}
        except Exception as e:
            return {"success": False, "error": f"Delete failed: {e}"}

    def ask_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit question to the RAG pipeline."""
        try:
            payload: Dict[str, Any] = {"question": question}
            if top_k is not None:
                payload["top_k"] = top_k
            if threshold is not None:
                payload["threshold"] = threshold
            if filter_doc_id:
                payload["filter_doc_id"] = filter_doc_id

            resp = requests.post(f"{self.base_url}/chat", json=payload, timeout=60)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {"success": False, "error": resp.json().get("detail", resp.text)}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Question answering timed out."}
        except Exception as e:
            return {"success": False, "error": f"Chat request failed: {e}"}
