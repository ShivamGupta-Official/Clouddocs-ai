"""
CloudDocs AI — FastAPI Application Entry Point

Creates and configures the FastAPI application:
- Registers API routes
- Creates required data directories on startup
- Configures CORS for the Streamlit frontend
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("clouddocs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Runs setup tasks on startup and cleanup on shutdown.
    Using lifespan instead of deprecated @app.on_event decorators.
    """
    # --- Startup ---
    settings = get_settings()
    logger.info("Starting CloudDocs AI v0.1.0")

    # Create data directories if they don't exist
    for dir_path in [
        settings.data_dir_path,
        settings.upload_dir_path,
        settings.chroma_dir_path,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info("Data directory ready: %s", dir_path)

    # Log LLM configuration status (without exposing keys)
    if settings.is_llm_configured():
        logger.info("LLM provider: %s", settings.llm_provider)
    else:
        logger.warning(
            "No LLM configured (provider=%s). "
            "The app will work for document upload and retrieval, "
            "but cannot generate answers. Set LLM_PROVIDER in .env to enable.",
            settings.llm_provider,
        )

    yield

    # --- Shutdown ---
    logger.info("Shutting down CloudDocs AI")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: The configured application instance.
    """
    app = FastAPI(
        title="CloudDocs AI",
        description=(
            "A Retrieval-Augmented Generation (RAG) API for document "
            "question-answering. Upload PDF/TXT documents, ask questions, "
            "and get answers grounded in your documents."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # --- CORS ---
    # Allow the Streamlit frontend (typically port 8501) to call the API.
    # In production, restrict this to your actual frontend domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Register Routers ---
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(chat_router)

    return app


# Create the app instance (used by uvicorn)
app = create_app()
