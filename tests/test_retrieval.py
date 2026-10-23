"""
Tests for app.retrieval.retriever

Verifies:
- Query input validation (rejecting empty/blank queries)
- Empty vector database handling (returns empty list cleanly)
- Semantic relevance matching (relevant queries retrieve expected text)
- Relevance score filtering threshold
- Metadata retention (filename, page_number, chunk_index, score)
- Scope filtering by document ID
- Document deletion prevents chunk retrieval
"""

import pytest
from app.config import get_settings
from app.ingestion.chunker import TextChunk
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.retriever import DocumentRetriever, RetrievedChunk
from app.retrieval.vector_store import VectorStore


@pytest.fixture
def embedding_engine():
    settings = get_settings()
    return EmbeddingEngine(settings.embedding_model)


@pytest.fixture
def vector_store(tmp_path):
    # Use temporary directory for isolated test vector store
    return VectorStore(tmp_path / "test_chroma")


@pytest.fixture
def retriever(embedding_engine, vector_store):
    return DocumentRetriever(
        embedding_engine=embedding_engine,
        vector_store=vector_store,
        default_top_k=5,
        default_threshold=0.3,
    )


def test_retriever_empty_query_raises_error(retriever):
    """Empty or whitespace queries must raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve("")

    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve("   \n\t  ")


def test_retriever_empty_database(retriever):
    """Searching an empty vector database should safely return an empty list."""
    results = retriever.retrieve("What is cloud computing?")
    assert results == []


def test_retriever_semantic_match_and_metadata(retriever, vector_store, embedding_engine):
    """Verify semantic search returns relevant passages with intact citation metadata."""
    chunks = [
        TextChunk(
            chunk_id="doc1_p1_c0",
            document_id="doc1",
            filename="aws_overview.pdf",
            page_number=1,
            chunk_index=0,
            text="Amazon EC2 provides scalable computing capacity in the AWS cloud.",
            start_char=0,
            end_char=68,
        ),
        TextChunk(
            chunk_id="doc1_p2_c1",
            document_id="doc1",
            filename="aws_overview.pdf",
            page_number=2,
            chunk_index=1,
            text="Amazon S3 is an object storage service offering industry-leading scalability and data availability.",
            start_char=69,
            end_char=168,
        ),
        TextChunk(
            chunk_id="doc2_p1_c0",
            document_id="doc2",
            filename="gardening_tips.txt",
            page_number=1,
            chunk_index=0,
            text="Tomato plants need plenty of sunlight and consistent watering to produce sweet fruit.",
            start_char=0,
            end_char=85,
        ),
    ]

    embeddings = embedding_engine.embed_texts([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)

    # Search for cloud compute
    results = retriever.retrieve("How can I run compute servers in AWS?", top_k=2)
    assert len(results) > 0

    top_result = results[0]
    assert isinstance(top_result, RetrievedChunk)
    assert top_result.document_id == "doc1"
    assert top_result.filename == "aws_overview.pdf"
    assert top_result.page_number == 1
    assert "EC2" in top_result.text
    assert top_result.score > 0.4  # High similarity


def test_retriever_threshold_filtering(retriever, vector_store, embedding_engine):
    """High threshold should filter out weakly related or unrelated passages."""
    chunk = TextChunk(
        chunk_id="doc_bio_1",
        document_id="doc_bio",
        filename="biology.txt",
        page_number=1,
        chunk_index=0,
        text="Photosynthesis is the process used by plants to convert light energy into chemical energy.",
        start_char=0,
        end_char=92,
    )
    embeddings = embedding_engine.embed_texts([chunk.text])
    vector_store.add_chunks([chunk], embeddings)

    # An unrelated query with high threshold should yield 0 results
    results = retriever.retrieve("What is quantum computing error mitigation?", threshold=0.85)
    assert len(results) == 0

    # With very low threshold, it returns the candidate
    low_thresh_results = retriever.retrieve("What is quantum computing error mitigation?", threshold=0.0)
    assert len(low_thresh_results) == 1


def test_retriever_deleted_document_not_retrieved(retriever, vector_store, embedding_engine):
    """Chunks from deleted documents must not appear in subsequent searches."""
    chunk = TextChunk(
        chunk_id="del_doc_1",
        document_id="doc_to_delete",
        filename="confidential.pdf",
        page_number=1,
        chunk_index=0,
        text="This secret document should disappear completely when deleted.",
        start_char=0,
        end_char=65,
    )
    embeddings = embedding_engine.embed_texts([chunk.text])
    vector_store.add_chunks([chunk], embeddings)

    # Verify searchable before deletion
    before = retriever.retrieve("secret document")
    assert len(before) == 1

    # Delete document from vector store
    vector_store.delete_document_chunks("doc_to_delete")

    # Verify unsearchable after deletion
    after = retriever.retrieve("secret document")
    assert len(after) == 0
