"""
Tests for Phase 4: Grounded Answer Generation, Prompts, Citations, and Chat API.

Verifies:
- Grounded prompt construction (anti-hallucination, anti-injection, source labeling)
- RAGService honest refusal when evidence is missing or insufficient
- RAGService genuine citation attribution (filename, page number, score)
- LLM Provider fallback (NoLLMProvider)
- POST /chat endpoint integration
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.generation.llm import BaseLLMProvider, NoLLMProvider
from app.generation.prompt import build_grounded_prompt, SYSTEM_INSTRUCTION
from app.main import app
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.retriever import DocumentRetriever, RetrievedChunk
from app.retrieval.vector_store import VectorStore
from app.services.rag_service import RAGService, INSUFFICIENT_EVIDENCE_MESSAGE

client = TestClient(app)


class MockLLMProvider(BaseLLMProvider):
    """Test LLM provider that echoes evidence and answers reliably."""

    def __init__(self, answer_text: str = "Mock grounded answer based on evidence."):
        self.answer_text = answer_text
        self.last_prompt = ""

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model-v1"

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.answer_text


def test_build_grounded_prompt():
    """Verify prompt contains system instructions, source passages, and anti-injection text."""
    passages = [
        RetrievedChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            filename="user_manual.pdf",
            page_number=3,
            chunk_index=0,
            text="The reset button is located on the back panel.",
            score=0.88,
        )
    ]
    prompt = build_grounded_prompt("Where is the reset button?", passages)

    assert "CloudDocs AI" in prompt
    assert "Prompt Injection Defense" in prompt
    assert "--- [Source 1 | Document: user_manual.pdf | Page: 3] ---" in prompt
    assert "The reset button is located on the back panel." in prompt
    assert "Where is the reset button?" in prompt


def test_rag_service_insufficient_evidence_when_empty(tmp_path):
    """RAGService must refuse to answer when vector store is empty."""
    settings = get_settings()
    engine = EmbeddingEngine(settings.embedding_model)
    vs = VectorStore(tmp_path / "empty_chroma")
    retriever = DocumentRetriever(engine, vs)
    mock_llm = MockLLMProvider()

    service = RAGService(retriever=retriever, llm_provider=mock_llm, settings=settings)
    response = service.answer_question("What is the capital of Mars?")

    assert response.has_sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert len(response.citations) == 0
    # Crucial: LLM generate must NOT have been called
    assert mock_llm.last_prompt == ""


def test_rag_service_insufficient_evidence_below_threshold(tmp_path):
    """RAGService must refuse to answer when no passages pass the relevance threshold."""
    settings = get_settings()
    engine = EmbeddingEngine(settings.embedding_model)
    vs = VectorStore(tmp_path / "threshold_chroma")

    from app.ingestion.chunker import TextChunk
    chunk = TextChunk(
        chunk_id="c1",
        document_id="d1",
        filename="cooking.txt",
        page_number=1,
        chunk_index=0,
        text="Bake the sourdough bread at 450 degrees Fahrenheit for 25 minutes.",
        start_char=0,
        end_char=71,
    )
    vs.add_chunks([chunk], engine.embed_texts([chunk.text]))

    retriever = DocumentRetriever(engine, vs, default_threshold=0.85)
    mock_llm = MockLLMProvider()
    service = RAGService(retriever=retriever, llm_provider=mock_llm, settings=settings)

    # An unrelated query about nuclear physics should not pass 0.85 similarity
    response = service.answer_question("How does nuclear fission work?", threshold=0.85)
    assert response.has_sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert len(response.citations) == 0


def test_rag_service_grounded_answer_and_citations(tmp_path):
    """When relevant passages exist, RAGService calls LLM and returns genuine citations."""
    settings = get_settings()
    engine = EmbeddingEngine(settings.embedding_model)
    vs = VectorStore(tmp_path / "grounded_chroma")

    from app.ingestion.chunker import TextChunk
    chunk = TextChunk(
        chunk_id="aws_c1",
        document_id="aws_doc",
        filename="aws_ec2.pdf",
        page_number=4,
        chunk_index=2,
        text="Amazon EC2 offers On-Demand, Reserved, and Spot instance pricing options.",
        start_char=0,
        end_char=75,
    )
    vs.add_chunks([chunk], engine.embed_texts([chunk.text]))

    retriever = DocumentRetriever(engine, vs, default_threshold=0.3)
    mock_llm = MockLLMProvider("Amazon EC2 offers three pricing options: On-Demand, Reserved, and Spot [Source 1].")
    service = RAGService(retriever=retriever, llm_provider=mock_llm, settings=settings)

    response = service.answer_question("What pricing options does EC2 have?")

    assert response.has_sufficient_evidence is True
    assert "Spot" in response.answer
    assert len(response.citations) == 1

    citation = response.citations[0]
    assert citation.citation_index == 1
    assert citation.filename == "aws_ec2.pdf"
    assert citation.page_number == 4
    assert citation.snippet == chunk.text
    assert citation.relevance_score > 0.4


def test_chat_endpoint_empty_question():
    """POST /chat with empty question must return HTTP 400 Bad Request."""
    response = client.post("/chat", json={"question": "   "})
    assert response.status_code == 400


def test_chat_endpoint_no_evidence_query():
    """POST /chat when no matching documents exist should return 200 with honest refusal."""
    response = client.post("/chat", json={"question": "What is the airspeed velocity of an unladen swallow?"})
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "answer" in data
    assert data["has_sufficient_evidence"] is False
    assert data["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert data["citations"] == []


def test_chat_endpoint_end_to_end_with_uploaded_doc():
    """Full workflow: upload document, query it via POST /chat, verify citations, and delete."""
    content = "Kubernetes uses Pods as the smallest deployable units of computing that can be created and managed."
    files = {"file": ("k8s_guide.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}

    # 1. Upload
    up_resp = client.post("/documents", files=files)
    assert up_resp.status_code == 201
    doc_id = up_resp.json()["id"]

    try:
        # 2. Ask question
        chat_resp = client.post("/chat", json={"question": "What is the smallest deployable unit in Kubernetes?"})
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()

        assert chat_data["has_sufficient_evidence"] is True
        assert len(chat_data["citations"]) >= 1

        top_citation = chat_data["citations"][0]
        assert top_citation["filename"] == "k8s_guide.txt"
        assert top_citation["page_number"] == 1
        assert "Pods" in top_citation["snippet"]

    finally:
        # 3. Clean up
        client.delete(f"/documents/{doc_id}")
