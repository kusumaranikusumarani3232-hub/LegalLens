"""Configuration management for LegalLens."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load local .env if present
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


class AppConfig:
    """Application and LLM settings."""

    # Provider Options: "gemini", "groq", "openai", "mock"
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "groq").lower()

    # API Keys
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Models
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Document Processing Parameters
    CHUNK_SIZE: int = 1000  # characters per chunk
    CHUNK_OVERLAP: int = 150  # character overlap
    MAX_FILE_SIZE_MB: int = 15  # Max upload size
    MAX_DOC_CHARS_ANALYSIS: int = 60000  # Max document characters for direct context

    @classmethod
    def get_active_provider(cls, override_provider: Optional[str] = None, override_key: Optional[str] = None) -> str:
        """Determines the most appropriate provider based on config and available keys."""
        target = (override_provider or cls.DEFAULT_PROVIDER).lower()

        # If override_key is provided for the target, use it
        if override_key:
            return target

        # Check credentials
        if target == "groq" and cls.GROQ_API_KEY:
            return "groq"
        if target == "gemini" and cls.GEMINI_API_KEY:
            return "gemini"
        if target == "openai" and cls.OPENAI_API_KEY:
            return "openai"

        # Fallback to whichever provider has an available key
        if cls.GROQ_API_KEY:
            return "groq"
        if cls.GEMINI_API_KEY:
            return "gemini"
        if cls.OPENAI_API_KEY:
            return "openai"

        # If no keys are present anywhere, fallback to smart offline mock
        return "mock"


config = AppConfig()
