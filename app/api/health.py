"""
CloudDocs AI — Health Check Endpoint

Provides a simple health endpoint for verifying the API is running
and checking the status of key components (database, vector store, LLM).
"""

from pathlib import Path

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns the application status and component availability.
    This is useful for:
    - Verifying the API is running
    - Docker health checks
    - Monitoring dashboards
    - The frontend settings page

    No sensitive information (keys, paths, etc.) is exposed.
    """
    settings = get_settings()

    # Check if the SQLite database file exists
    sqlite_exists = Path(settings.sqlite_db_path).exists()

    # Check if the ChromaDB directory exists and has data
    chroma_path = Path(settings.chroma_dir_path)
    chroma_exists = (
        chroma_path.exists() and any(chroma_path.iterdir()) if chroma_path.exists() else False
    )

    return {
        "status": "healthy",
        "application": "CloudDocs AI",
        "version": "0.1.0",
        "components": {
            "api": "operational",
            "metadata_db": "connected" if sqlite_exists else "not initialized",
            "vector_store": "connected" if chroma_exists else "not initialized",
            "llm": _get_llm_status(settings),
        },
        "configuration": {
            "embedding_model": settings.embedding_model,
            "llm_provider": settings.llm_provider,
            "llm_model": _get_llm_model_name(settings),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_top_k": settings.retrieval_top_k,
            "max_upload_size_mb": settings.max_upload_size_mb,
        },
    }


def _get_llm_status(settings) -> str:
    """Get a safe, human-readable LLM status string."""
    if settings.llm_provider == "none":
        return "not configured"
    if not settings.is_llm_configured():
        return f"{settings.llm_provider} — API key missing"
    return f"{settings.llm_provider} — configured"


def _get_llm_model_name(settings) -> str | None:
    """Get the configured LLM model name (safe to expose)."""
    if settings.llm_provider == "gemini":
        return settings.gemini_model
    if settings.llm_provider == "openai":
        return settings.openai_model
    if settings.llm_provider == "ollama":
        return settings.ollama_model
    return None
