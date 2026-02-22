"""
Pinecone vector search with metadata filtering.
Embeds the user's query and retrieves the top-K most similar resumes
filtered by the predicted category.
"""

from __future__ import annotations

import logging

from openai import OpenAI
from pinecone import Pinecone

from config import DEFAULT_TOP_K, EMBEDDING_MODEL
from models import RetrievedDocument

logger = logging.getLogger(__name__)


def retrieve(
    openai_client: OpenAI,
    pc: Pinecone,
    index_name: str,
    query: str,
    category: str,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[list[RetrievedDocument], float]:
    """
    Embed the query and search Pinecone with a category metadata filter.

    Returns:
        Tuple of (list of RetrievedDocument, embedding_time_ms).
    """
    import time

    # 1. Embed the query
    logger.info("Embedding query for retrieval...")
    t0 = time.perf_counter()
    embed_response = openai_client.embeddings.create(
        input=[query], model=EMBEDDING_MODEL
    )
    query_embedding = embed_response.data[0].embedding
    embedding_time_ms = (time.perf_counter() - t0) * 1000
    logger.info("Query embedded in %.1f ms", embedding_time_ms)

    # 2. Search Pinecone with metadata filter
    index = pc.Index(index_name)
    logger.info(
        "Searching Pinecone (top_k=%d, category='%s')...", top_k, category
    )

    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"category": {"$eq": category}},
    )

    # 3. Parse results into Pydantic models
    documents: list[RetrievedDocument] = []
    for match in search_results.matches:
        meta = match.metadata or {}
        doc = RetrievedDocument(
            id=match.id,
            score=round(match.score, 4),
            name=meta.get("name", "Unknown"),
            category=meta.get("category", "Unknown"),
            skills=meta.get("skills", ""),
            experience_years=int(meta.get("experience_years", 0)),
            resume_text=meta.get("resume_text", ""),
        )
        documents.append(doc)

    logger.info("Retrieved %d documents.", len(documents))
    return documents, embedding_time_ms
