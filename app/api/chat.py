"""
CloudDocs AI — Chat & Question-Answering API Endpoint

Provides:
- POST /chat: Submit questions and receive answers grounded in indexed documents
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.documents import get_embedding_engine, get_vector_store
from app.config import Settings, get_settings
from app.generation.llm import BaseLLMProvider, get_llm_provider
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.retriever import DocumentRetriever
from app.retrieval.vector_store import VectorStore
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService

logger = logging.getLogger("clouddocs.api.chat")

router = APIRouter(prefix="/chat", tags=["Chat"])

# Singletons for retriever and rag_service
_retriever: DocumentRetriever | None = None
_rag_service: RAGService | None = None


def get_retriever(
    settings: Settings = Depends(get_settings),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentRetriever:
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever(
            embedding_engine=embedding_engine,
            vector_store=vector_store,
            default_top_k=settings.retrieval_top_k,
            default_threshold=settings.relevance_threshold,
        )
    return _retriever


def get_rag_service(
    settings: Settings = Depends(get_settings),
    retriever: DocumentRetriever = Depends(get_retriever),
) -> RAGService:
    global _rag_service
    llm_provider: BaseLLMProvider = get_llm_provider(settings)
    # Always rebuild RAG service so settings changes take immediate effect
    return RAGService(
        retriever=retriever,
        llm_provider=llm_provider,
        settings=settings,
    )


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about indexed documents",
)
async def chat_endpoint(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """
    Submit a question to the RAG pipeline.

    1. Searches indexed document chunks using dense semantic embeddings
    2. Filters results using relevance threshold
    3. Evaluates evidence sufficiency (returns honest refusal if evidence is missing)
    4. Generates an answer strictly grounded in the retrieved passages
    5. Returns genuine source citations with filename, page numbers, and verbatim snippets
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace.",
        )

    try:
        response = rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
            threshold=request.threshold,
            filter_doc_id=request.filter_doc_id,
        )
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Chat question answering failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering the question: {e}",
        ) from e
