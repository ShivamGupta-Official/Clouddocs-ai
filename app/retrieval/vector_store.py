"""
CloudDocs AI — ChromaDB Vector Store Manager

Provides persistent vector storage and similarity retrieval using ChromaDB:
- Persistent disk storage in data/chromadb/
- Metadata indexing (document_id, filename, page_number, chunk_index)
- Document chunk deletion with count tracking
- Collection dimension validation
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.ingestion.chunker import TextChunk

logger = logging.getLogger("clouddocs.vector_store")


class VectorStore:
    """Manages persistent ChromaDB vector collections."""

    COLLECTION_NAME = "clouddocs_knowledge_base"

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create the main collection using cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialized at %s. Current item count: %d",
            self.persist_dir,
            self.collection.count(),
        )

    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store text chunks and their embeddings into ChromaDB.

        Args:
            chunks: List of TextChunk objects with metadata.
            embeddings: Corresponding vector embeddings.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but {len(embeddings)} embeddings."
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "filename": c.filename,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Indexed %d chunks into ChromaDB", len(chunks))
        return len(chunks)

    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all vector records associated with a specific document ID.

        Args:
            document_id: UUID of the document whose chunks should be purged.

        Returns:
            Number of deleted chunks.
        """
        # Find existing chunks for this document
        existing = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        count_to_delete = len(existing["ids"]) if existing and "ids" in existing else 0

        if count_to_delete > 0:
            self.collection.delete(
                where={"document_id": document_id}
            )
            logger.info(
                "Deleted %d chunks for document_id '%s' from ChromaDB",
                count_to_delete,
                document_id,
            )

        return count_to_delete

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search for candidate chunks.

        Args:
            query_embedding: Vector representation of the user query.
            top_k: Maximum number of candidate chunks to retrieve.
            filter_doc_id: Optional document ID to filter search scope.

        Returns:
            List of dicts containing chunk text, metadata, and distance.
        """
        if self.collection.count() == 0:
            return []

        where_clause = {"document_id": filter_doc_id} if filter_doc_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i, chunk_id in enumerate(ids):
            retrieved.append({
                "chunk_id": chunk_id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                # Cosine distance: 0 is identical, 2 is opposite.
                # Cosine similarity = 1 - (distance / 2) or 1 - distance
                "distance": distances[i] if i < len(distances) else 1.0,
            })

        return retrieved

    def count(self) -> int:
        """Total chunks currently stored in the vector collection."""
        return self.collection.count()
