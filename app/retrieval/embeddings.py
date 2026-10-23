"""
CloudDocs AI — Embeddings Engine

Wraps Sentence Transformers for local CPU embedding generation.
Verifies dimension consistency and provides batched embedding for chunks
and single embedding for search queries.
"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("clouddocs.embeddings")


class EmbeddingEngine:
    """Generates dense vector embeddings using Sentence Transformers on CPU."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info("Loading embedding model '%s' on CPU...", self.model_name)
        # Explicitly set device to 'cpu' as verified in Phase 0 environment check
        self.model = SentenceTransformer(self.model_name, device="cpu")
        
        # Verify dimension dynamically using dummy inference
        dummy_vector = self.model.encode("test", convert_to_numpy=True)
        self.dimension = int(dummy_vector.shape[0])
        logger.info(
            "Embedding model '%s' loaded successfully. Dimension: %d",
            self.model_name,
            self.dimension,
        )

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed a list of text strings in batches.

        Args:
            texts: List of strings (e.g. document chunks).
            batch_size: Number of texts per batch.

        Returns:
            List of float vectors, each of length self.dimension.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Cosine similarity normalization
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single search query.

        Args:
            query: The user's query text.

        Returns:
            Float vector of length self.dimension.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query cannot be empty for embedding generation.")

        embedding = self.model.encode(
            cleaned_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()
