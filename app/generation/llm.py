"""
CloudDocs AI — Configurable LLM Provider Interface

Provides a unified interface across LLM backends:
- Google Gemini (official google-genai SDK)
- OpenAI (official openai SDK)
- Ollama (local HTTP API)
- NoLLMProvider (graceful offline/retrieval-only fallback)

Ensures that API keys and secrets are never hardcoded or leaked into error logs.
"""

import logging
from abc import ABC, abstractmethod

import httpx

from app.config import Settings

logger = logging.getLogger("clouddocs.llm")


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a text response for the given grounded prompt.

        Args:
            prompt: Complete prompt containing system instructions, context, and question.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If the external API call fails.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str | None:
        pass


class NoLLMProvider(BaseLLMProvider):
    """
    Extractive fallback provider when no external generative LLM is configured.
    Performs deterministic, 100% grounded extraction directly from the user's
    verified retrieved document passages, eliminating hallucinations entirely.
    """

    def __init__(self, reason: str = "Extractive Search Mode"):
        self.reason = reason

    @property
    def provider_name(self) -> str:
        return "none"

    @property
    def model_name(self) -> str | None:
        return None

    def generate(self, prompt: str) -> str:
        import re

        # Extract passages between 'SOURCE PASSAGES:' and 'USER QUESTION:'
        match = re.search(
            r"SOURCE PASSAGES:\s*={10,}\s*(.*?)\s*={10,}\s*USER QUESTION:",
            prompt,
            re.DOTALL,
        )
        if match:
            passages_block = match.group(1).strip()
            sources = re.split(r"---\s*\[Source\s+(\d+)[^\]]*\]\s*---", passages_block)
            if len(sources) >= 3:
                src_num = sources[1]
                src_text = sources[2].strip()
                cleaned_text = "\n".join(
                    line.strip() for line in src_text.splitlines() if line.strip()
                )
                return (
                    f"Direct evidence extracted from your uploaded document [Source {src_num}]:\n\n"
                    f"{cleaned_text}"
                )

        return (
            "The provided documents do not contain sufficient evidence to answer this question."
        )


class GeminiProvider(BaseLLMProvider):
    """Google Gemini integration using the official google-genai SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        if not api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY in .env")
        self.api_key = api_key
        self._model_name = model_name

        try:
            from google import genai

            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google GenAI client: {e}") from e

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
            if not response.text:
                return "The model did not return any text output."
            return response.text.strip()
        except Exception as e:
            err_str = str(e).replace(self.api_key, "[REDACTED]")
            logger.error("Gemini API call failed: %s", err_str)
            raise RuntimeError(f"Google Gemini generation failed: {err_str}") from e


class OpenAIProvider(BaseLLMProvider):
    """OpenAI integration using the official openai SDK."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY in .env")
        self.api_key = api_key
        self._model_name = model_name

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for factual precision
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            return content.strip()
        except Exception as e:
            err_str = str(e).replace(self.api_key, "[REDACTED]")
            logger.error("OpenAI API call failed: %s", err_str)
            raise RuntimeError(f"OpenAI generation failed: {err_str}") from e


class OllamaProvider(BaseLLMProvider):
    """Local Ollama integration via HTTP REST API."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
                data = resp.json()
                return data.get("response", "").strip()
        except httpx.ConnectError as e:
            logger.error("Could not connect to Ollama at %s", self.base_url)
            raise RuntimeError(
                f"Could not connect to local Ollama server at {self.base_url}. "
                "Ensure Ollama is running ('ollama serve')."
            ) from e
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            raise RuntimeError(f"Ollama generation failed: {e}") from e


def get_llm_provider(settings: Settings) -> BaseLLMProvider:
    """
    Factory function returning the configured LLM provider.

    Detects missing keys or provider type and falls back to NoLLMProvider
    with clear diagnostic instructions.
    """
    provider = settings.llm_provider.lower().strip()

    if provider == "gemini":
        if not settings.gemini_api_key:
            return NoLLMProvider("Gemini provider selected, but GEMINI_API_KEY is not configured.")
        try:
            return GeminiProvider(settings.gemini_api_key, settings.gemini_model)
        except Exception as e:
            logger.warning("Could not initialize Gemini provider: %s", e)
            return NoLLMProvider(f"Could not initialize Gemini: {e}")

    elif provider == "openai":
        if not settings.openai_api_key:
            return NoLLMProvider("OpenAI provider selected, but OPENAI_API_KEY is not configured.")
        try:
            return OpenAIProvider(settings.openai_api_key, settings.openai_model)
        except Exception as e:
            logger.warning("Could not initialize OpenAI provider: %s", e)
            return NoLLMProvider(f"Could not initialize OpenAI: {e}")

    elif provider == "ollama":
        return OllamaProvider(settings.ollama_base_url, settings.ollama_model)

    return NoLLMProvider("No LLM provider configured (LLM_PROVIDER=none).")
