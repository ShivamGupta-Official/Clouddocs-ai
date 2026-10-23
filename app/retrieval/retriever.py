"""
CloudDocs AI — Document Retriever Module

Performs semantic search across indexed document chunks:
1. Validates user search query
2. Computes 384-dimensional dense query vector using Sentence Transformers
3. Executes cosine similarity search in ChromaDB
4. Converts cosine distance to similarity scores
5. Applies configurable relevance filtering threshold
6. Returns structured passages with source metadata (filename, page number, chunk ID)
"""

import logging
from dataclasses import dataclass
from typing import Any

from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger("clouddocs.retriever")


@dataclass
class RetrievedChunk:
    """Represents a passage retrieved from the vector store with attribution metadata."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float  # Cosine similarity score [0.0, 1.0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "score": round(self.score, 4),
        }


class DocumentRetriever:
    """Orchestrates query embedding, vector search, and relevance filtering."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        default_top_k: int = 5,
        default_threshold: float = 0.3,
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        filter_doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant document passages for a user query.

        Args:
            query: User's question or search query string.
            top_k: Maximum candidate passages to retrieve (defaults to config value).
            threshold: Minimum relevance similarity score (defaults to config value).
            filter_doc_id: Optional UUID to restrict search to a single document.

        Returns:
            List of RetrievedChunk objects sorted by relevance score descending.

        Raises:
            ValueError: If query is empty or contains only whitespace.
        """
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            raise ValueError("Search query cannot be empty or blank.")

        effective_top_k = top_k if top_k is not None and top_k > 0 else self.default_top_k
        effective_threshold = (
            threshold if threshold is not None and threshold >= 0.0 else self.default_threshold
        )

        # Handle empty vector store
        if self.vector_store.count() == 0:
            logger.info("Vector database is empty. No passages to retrieve.")
            return []

        # 1. Generate query embedding vector
        query_embedding = self.embedding_engine.embed_query(cleaned_query)

        # 2. Search ChromaDB
        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            filter_doc_id=filter_doc_id,
        )

        # 3. Transform and filter by relevance score
        results: list[RetrievedChunk] = []
        for cand in candidates:
            # ChromaDB cosine distance: distance in [0, 2]
            # Convert to similarity: 1 - distance, clamped to [0.0, 1.0]
            raw_dist = cand.get("distance", 1.0)
            similarity = max(0.0, min(1.0, 1.0 - raw_dist))

            if similarity < effective_threshold:
                logger.debug(
                    "Candidate chunk '%s' rejected (similarity %.4f < threshold %.4f)",
                    cand.get("chunk_id"),
                    similarity,
                    effective_threshold,
                )
                continue

            metadata = cand.get("metadata", {})
            results.append(
                RetrievedChunk(
                    chunk_id=cand.get("chunk_id", ""),
                    document_id=metadata.get("document_id", ""),
                    filename=metadata.get("filename", "unknown"),
                    page_number=int(metadata.get("page_number", 1)),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    text=cand.get("text", ""),
                    score=similarity,
                )
            )

        # Sort results by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "Query '%s': retrieved %d relevant chunks (top_k=%d, threshold=%.2f)",
            cleaned_query,
            len(results),
            effective_top_k,
            effective_threshold,
        )
        return results
