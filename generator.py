"""
Answer generation using GPT-4o-mini.
Builds a context string from retrieved resume documents and generates
a grounded answer that references the actual data.
"""

from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import LLM_MODEL
from models import GeneratedAnswer, RetrievedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_GENERATION_SYSTEM_PROMPT = """\
You are a professional resume analyst assistant. You answer questions about \
candidates based ONLY on the resume documents provided in the context below.

Rules:
1. Answer ONLY using the information in the provided context.
2. If the context doesn't contain enough information, say so clearly.
3. Reference specific candidates by name when relevant.
4. Provide structured, professional responses.
5. Highlight key qualifications, skills, and experience that match the query.
6. If comparing candidates, create a clear comparison format.
7. Never make up information not present in the context.
"""

_CONTEXT_TEMPLATE = """\
--- RETRIEVED RESUMES ---
{context}
--- END OF RESUMES ---

User Question: {query}
"""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def _build_context(documents: list[RetrievedDocument], max_chars: int = 12000) -> str:
    """Build a context string from retrieved documents, respecting token limits."""
    parts: list[str] = []
    total_chars = 0

    for i, doc in enumerate(documents, 1):
        entry = (
            f"[Resume {i}] Name: {doc.name}\n"
            f"Category: {doc.category}\n"
            f"Skills: {doc.skills}\n"
            f"Experience: {doc.experience_years} years\n"
            f"Resume: {doc.resume_text}\n"
        )
        if total_chars + len(entry) > max_chars:
            logger.warning(
                "Context truncated at document %d to stay within %d chars.",
                i,
                max_chars,
            )
            break
        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def generate_answer(
    client: OpenAI,
    query: str,
    documents: list[RetrievedDocument],
    model: str = LLM_MODEL,
) -> GeneratedAnswer:
    """
    Generate a grounded answer from retrieved resume documents.
    """
    if not documents:
        return GeneratedAnswer(
            answer="No relevant resumes were found for your query. "
            "Try rephrasing or broadening your search.",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
        )

    context = _build_context(documents)
    user_message = _CONTEXT_TEMPLATE.format(context=context, query=query)

    logger.info("Generating answer with %s...", model)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    content = response.choices[0].message.content or "Unable to generate an answer."
    usage = response.usage

    answer = GeneratedAnswer(
        answer=content,
        model=model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )

    logger.info(
        "Answer generated (%d prompt tokens, %d completion tokens).",
        answer.prompt_tokens,
        answer.completion_tokens,
    )
    return answer
