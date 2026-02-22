"""
Pipeline orchestrator — ties classify → retrieve → generate together.
Captures timing for every stage and returns a fully traced PipelineResponse.
"""

from __future__ import annotations

import logging
import time

from openai import OpenAI
from pinecone import Pinecone

from classifier import classify_query
from config import DEFAULT_TOP_K, EMBEDDING_MODEL, LLM_MODEL, get_settings
from generator import generate_answer
from models import PipelineResponse, PipelineTrace, QueryInput
from retriever import retrieve

logger = logging.getLogger(__name__)


def run_pipeline(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> PipelineResponse:
    """
    Execute the full RAG pipeline:
      1. Validate query
      2. Classify into a category
      3. Retrieve matching resumes from Pinecone
      4. Generate a grounded answer

    Returns a PipelineResponse with full trace information.
    """
    pipeline_start = time.perf_counter()

    # -- 0. Validate query ---------------------------------------------------
    validated = QueryInput(query=query)
    logger.info("Pipeline started for query: %s", validated.query[:80])

    # -- Initialize clients ---------------------------------------------------
    settings = get_settings()
    openai_client = OpenAI(api_key=settings.openai_api_key)
    pc = Pinecone(api_key=settings.pinecone_api_key)

    # -- 1. Classify ----------------------------------------------------------
    t0 = time.perf_counter()
    classification = classify_query(openai_client, validated.query)
    classification_time = (time.perf_counter() - t0) * 1000
    logger.info(
        "Classification: %s (%.0f ms)", classification.category, classification_time
    )

    # -- 2. Retrieve ----------------------------------------------------------
    t0 = time.perf_counter()
    documents, embedding_time_ms = retrieve(
        openai_client=openai_client,
        pc=pc,
        index_name=settings.pinecone_index_name,
        query=validated.query,
        category=classification.category,
        top_k=top_k,
    )
    retrieval_time = (time.perf_counter() - t0) * 1000
    logger.info("Retrieval: %d docs (%.0f ms)", len(documents), retrieval_time)

    # -- 3. Generate ----------------------------------------------------------
    t0 = time.perf_counter()
    answer = generate_answer(openai_client, validated.query, documents)
    generation_time = (time.perf_counter() - t0) * 1000
    logger.info("Generation: %.0f ms", generation_time)

    # -- 4. Build trace -------------------------------------------------------
    total_time = (time.perf_counter() - pipeline_start) * 1000
    trace = PipelineTrace(
        classification_time_ms=round(classification_time, 1),
        embedding_time_ms=round(embedding_time_ms, 1),
        retrieval_time_ms=round(retrieval_time, 1),
        generation_time_ms=round(generation_time, 1),
        total_time_ms=round(total_time, 1),
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        top_k=top_k,
        documents_found=len(documents),
    )

    return PipelineResponse(
        query=validated.query,
        classification=classification,
        retrieved_documents=documents,
        answer=answer,
        trace=trace,
    )
