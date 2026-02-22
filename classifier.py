"""
Query category classifier using OpenAI GPT-4o-mini.
Classifies a user's natural-language query into one of the predefined
resume categories so that metadata filtering can be applied.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import LLM_MODEL, RESUME_CATEGORIES
from models import ClassificationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_CLASSIFICATION_SYSTEM_PROMPT = f"""\
You are a query classifier for a resume search system.
Given a user's query, classify it into exactly ONE of the following categories:
{json.dumps(RESUME_CATEGORIES)}

Respond with a JSON object containing:
- "category": the predicted category (must be exactly one of the categories above)
- "confidence": a float between 0 and 1 indicating your confidence

Example response:
{{"category": "Data Science", "confidence": 0.92}}

Rules:
- Always pick the MOST relevant category.
- If the query is ambiguous, pick the closest match and lower the confidence.
- Never invent new categories.
- Respond ONLY with valid JSON, no extra text.
"""


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def classify_query(
    client: OpenAI,
    query: str,
    model: str = LLM_MODEL,
) -> ClassificationResult:
    """
    Classify a user query into a resume category.

    Uses structured JSON output from GPT-4o-mini with retry logic.
    """
    logger.info("Classifying query: %s", query[:80])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=100,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    logger.debug("Raw classification response: %s", raw)

    parsed = json.loads(raw)

    # Validate with Pydantic
    result = ClassificationResult(
        category=parsed.get("category", RESUME_CATEGORIES[0]),
        confidence=float(parsed.get("confidence", 0.5)),
    )

    logger.info(
        "Classification result: %s (confidence: %.2f)",
        result.category,
        result.confidence,
    )
    return result
