"""
CloudDocs AI — RAG Orchestration Service

Orchestrates the complete Retrieval-Augmented Generation workflow:
1. Validates user question
2. Retrieves relevant candidate passages via DocumentRetriever
3. Checks evidence sufficiency
4. Constructs grounded prompt with strict anti-hallucination & anti-injection guardrails
5. Generates answer using configured LLMProvider
6. Associates genuine source citations with verbatim snippets and page numbers
7. Computes pipeline execution latency
"""

import logging
import time

from app.config import Settings
from app.generation.llm import BaseLLMProvider
from app.generation.prompt import build_grounded_prompt
from app.retrieval.retriever import DocumentRetriever
from app.schemas.chat import ChatResponse, SourceCitation

logger = logging.getLogger("clouddocs.rag_service")

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The provided documents do not contain sufficient evidence to answer this question."
)


class RAGService:
    """Coordinates retrieval and grounded answer generation."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_provider: BaseLLMProvider,
        settings: Settings,
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.settings = settings

    def answer_question(
        self,
        question: str,
        top_k: int | None = None,
        threshold: float | None = None,
        filter_doc_id: str | None = None,
    ) -> ChatResponse:
        """
        Process user question and return grounded answer with source citations.

        Args:
            question: User's question text.
            top_k: Maximum candidate passages to retrieve.
            threshold: Minimum relevance similarity score.
            filter_doc_id: Optional UUID to restrict search scope.

        Returns:
            ChatResponse containing answer, citations, evidence flag, and latency.

        Raises:
            ValueError: If question is empty or invalid.
        """
        start_time = time.perf_counter()
        cleaned_question = question.strip() if question else ""
        if not cleaned_question:
            raise ValueError("Question cannot be empty or blank.")

        # Step 1: Retrieve candidate passages
        passages = self.retriever.retrieve(
            query=cleaned_question,
            top_k=top_k,
            threshold=threshold,
            filter_doc_id=filter_doc_id,
        )

        # Step 2: Evidence sufficiency check
        if not passages:
            latency = time.perf_counter() - start_time
            logger.info("No sufficient evidence retrieved for query: '%s'", cleaned_question)
            return ChatResponse(
                question=cleaned_question,
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                has_sufficient_evidence=False,
                llm_provider=self.llm_provider.provider_name,
                llm_model=self.llm_provider.model_name,
                latency_seconds=round(latency, 3),
            )

        # Step 3: Map genuine source citations
        citations = [
            SourceCitation(
                citation_index=idx,
                chunk_id=p.chunk_id,
                document_id=p.document_id,
                filename=p.filename,
                page_number=p.page_number,
                chunk_index=p.chunk_index,
                snippet=p.text,
                relevance_score=p.score,
            )
            for idx, p in enumerate(passages, start=1)
        ]

        # Step 4: Construct grounded prompt
        prompt = build_grounded_prompt(
            question=cleaned_question,
            passages=passages,
        )

        # Step 5: Generate answer via configured LLM
        try:
            raw_answer = self.llm_provider.generate(prompt)
        except Exception as e:
            logger.error("LLM answer generation failed: %s", e)
            raw_answer = f"Error generating answer with {self.llm_provider.provider_name}: {e}"

        latency = time.perf_counter() - start_time

        return ChatResponse(
            question=cleaned_question,
            answer=raw_answer,
            citations=citations,
            has_sufficient_evidence=True,
            llm_provider=self.llm_provider.provider_name,
            llm_model=self.llm_provider.model_name,
            latency_seconds=round(latency, 3),
        )
