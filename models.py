"""
Pydantic models and schemas for the Resume QA RAG pipeline.
Provides strict validation for data ingestion, query processing, and response formatting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Data Ingestion Models
# ---------------------------------------------------------------------------
class ResumeRecord(BaseModel):
    """Validates a single resume row from the CSV dataset."""

    id: str = Field(..., min_length=1, description="Unique resume identifier")
    name: str = Field(..., min_length=1, description="Candidate full name")
    category: str = Field(..., description="Resume category (e.g. Data Science)")
    skills: str = Field(..., min_length=1, description="Comma-separated skill list")
    experience_years: int = Field(
        ..., ge=0, le=50, description="Years of professional experience"
    )
    education: str = Field(..., description="Highest education qualification")
    resume_text: str = Field(
        ..., min_length=10, description="Full resume text for embedding"
    )

    @field_validator("category")
    @classmethod
    def _valid_category(cls, v: str) -> str:
        from config import RESUME_CATEGORIES

        if v not in RESUME_CATEGORIES:
            raise ValueError(
                f"Invalid category '{v}'. Must be one of: {RESUME_CATEGORIES}"
            )
        return v


# ---------------------------------------------------------------------------
# Query Models
# ---------------------------------------------------------------------------
class QueryInput(BaseModel):
    """Validates user query input."""

    query: str = Field(
        ..., min_length=3, max_length=1000, description="User's natural language query"
    )


class ClassificationResult(BaseModel):
    """Result of classifying a query into a resume category."""

    category: str = Field(..., description="Predicted resume category")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence score"
    )

    @field_validator("category")
    @classmethod
    def _valid_category(cls, v: str) -> str:
        from config import RESUME_CATEGORIES

        if v not in RESUME_CATEGORIES:
            raise ValueError(
                f"Invalid category '{v}'. Must be one of: {RESUME_CATEGORIES}"
            )
        return v


# ---------------------------------------------------------------------------
# Retrieval Models
# ---------------------------------------------------------------------------
class RetrievedDocument(BaseModel):
    """A single document retrieved from Pinecone."""

    id: str = Field(..., description="Document / vector ID in Pinecone")
    score: float = Field(..., description="Cosine similarity score")
    name: str = Field(default="Unknown", description="Candidate name")
    category: str = Field(default="Unknown", description="Resume category")
    skills: str = Field(default="", description="Skills from metadata")
    experience_years: int = Field(default=0, description="Years of experience")
    resume_text: str = Field(default="", description="Resume content snippet")


# ---------------------------------------------------------------------------
# Generation Models
# ---------------------------------------------------------------------------
class GeneratedAnswer(BaseModel):
    """LLM-generated answer from the retrieved context."""

    answer: str = Field(..., description="Generated answer text")
    model: str = Field(..., description="LLM model used for generation")
    prompt_tokens: int = Field(default=0, description="Prompt tokens consumed")
    completion_tokens: int = Field(default=0, description="Completion tokens generated")


# ---------------------------------------------------------------------------
# Pipeline Trace & Response
# ---------------------------------------------------------------------------
class PipelineTrace(BaseModel):
    """Captures timing and intermediate results for full transparency."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Pipeline execution timestamp",
    )
    classification_time_ms: float = Field(
        default=0.0, description="Time for category classification (ms)"
    )
    embedding_time_ms: float = Field(
        default=0.0, description="Time to embed the query (ms)"
    )
    retrieval_time_ms: float = Field(
        default=0.0, description="Time for Pinecone vector search (ms)"
    )
    generation_time_ms: float = Field(
        default=0.0, description="Time for LLM answer generation (ms)"
    )
    total_time_ms: float = Field(
        default=0.0, description="Total end-to-end pipeline time (ms)"
    )
    embedding_model: str = Field(default="", description="Embedding model used")
    llm_model: str = Field(default="", description="LLM model used")
    top_k: int = Field(default=5, description="Number of documents retrieved")
    documents_found: int = Field(
        default=0, description="Actual number of documents returned"
    )


class PipelineResponse(BaseModel):
    """Complete response combining all pipeline stages."""

    query: str = Field(..., description="Original user query")
    classification: ClassificationResult
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    answer: GeneratedAnswer
    trace: PipelineTrace
