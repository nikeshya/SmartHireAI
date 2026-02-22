"""
Centralized configuration for the Resume QA RAG system.
Loads from Streamlit secrets (cloud) or .env (local).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Load .env for local development
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Also try Streamlit secrets (for Streamlit Cloud deployment)
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"):
            if key in st.secrets:
                os.environ.setdefault(key, st.secrets[key])
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESUME_CATEGORIES: list[str] = [
    "Data Science",
    "Web Development",
    "DevOps",
    "Mobile Development",
    "Machine Learning",
    "Cloud Engineering",
]

EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536
LLM_MODEL: str = "gpt-4o-mini"
DEFAULT_TOP_K: int = 5
PINECONE_METRIC: str = "cosine"
DATA_CSV_PATH: Path = _PROJECT_ROOT / "data" / "resumes.csv"


# ---------------------------------------------------------------------------
# Validated Settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Validated application settings sourced from environment variables."""

    openai_api_key: str = Field(..., description="OpenAI API key")
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_index_name: str = Field(
        default="resume-rag", description="Pinecone index name"
    )

    @field_validator("openai_api_key")
    @classmethod
    def _openai_key_not_empty(cls, v: str) -> str:
        if not v or v.startswith("sk-your"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to .env or Streamlit secrets."
            )
        return v

    @field_validator("pinecone_api_key")
    @classmethod
    def _pinecone_key_not_empty(cls, v: str) -> str:
        if not v or v.startswith("your-"):
            raise ValueError(
                "PINECONE_API_KEY is not set. Add it to .env or Streamlit secrets."
            )
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Return a cached Settings instance. Raises on invalid config."""
    return Settings()  # type: ignore[call-arg]
