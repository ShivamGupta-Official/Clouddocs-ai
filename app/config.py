"""
CloudDocs AI — Application Configuration

Loads and validates all settings from environment variables.
Uses pydantic-settings to provide type checking and default values.

Usage:
    from app.config import get_settings
    settings = get_settings()
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root is one level up from this file (app/config.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    Each field corresponds to an environment variable. For example,
    `embedding_model` reads from the EMBEDDING_MODEL env var.

    Defaults are provided for all non-secret settings so the app can
    start without a .env file (in retrieval-only mode with no LLM).
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore unknown env vars
    )

    # --- Embedding Model ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- LLM Provider ---
    # Options: "gemini", "openai", "ollama", "none"
    llm_provider: str = "none"

    # Google Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # --- Application Settings ---
    max_upload_size_mb: int = 50
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5
    relevance_threshold: float = 0.3

    # --- Server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Data Directories (relative to project root) ---
    data_dir: str = "data"
    upload_dir: str = "data/uploads"
    chroma_dir: str = "data/chromadb"
    sqlite_path: str = "data/metadata.db"

    # --- Resolved Paths (computed from the above) ---

    @property
    def data_dir_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def upload_dir_path(self) -> Path:
        return PROJECT_ROOT / self.upload_dir

    @property
    def chroma_dir_path(self) -> Path:
        return PROJECT_ROOT / self.chroma_dir

    @property
    def sqlite_db_path(self) -> Path:
        return PROJECT_ROOT / self.sqlite_path

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def is_llm_configured(self) -> bool:
        """Check if an LLM provider is properly configured."""
        if self.llm_provider == "none":
            return False
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            return False
        if self.llm_provider == "openai" and not self.openai_api_key:
            return False
        # Ollama doesn't need an API key, just a running server
        return self.llm_provider in ("gemini", "openai", "ollama")

    def validate_chunk_settings(self) -> None:
        """Validate that chunk settings are safe (no infinite loops, etc.)."""
        if self.chunk_size < 100:
            raise ValueError(f"chunk_size must be >= 100, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size}) to avoid infinite loops"
            )


@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings (cached — created once per process).

    Returns:
        Settings: The validated application settings.
    """
    settings = Settings()
    settings.validate_chunk_settings()
    return settings
