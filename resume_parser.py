"""
Automatic resume parser using GPT-4o-mini.
Extracts structured data (name, category, skills, experience, education)
from raw resume text — so the user never has to manually fill a CSV.
"""

from __future__ import annotations

import json
import logging
import uuid

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import LLM_MODEL, RESUME_CATEGORIES
from models import ResumeRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------
_EXTRACTION_SYSTEM_PROMPT = f"""\
You are an expert resume parser. Given raw resume text, extract the following fields as a JSON object:

{{
  "name": "Candidate's full name",
  "category": "One of: {json.dumps(RESUME_CATEGORIES)}",
  "skills": "Comma-separated list of key technical skills",
  "experience_years": <integer, total years of professional experience>,
  "education": "Highest education qualification (e.g. B.Tech Computer Science)",
  "resume_text": "A clean, concise summary of the full resume (max 500 words)"
}}

Rules:
- "category" MUST be exactly one of the listed categories. Pick the best fit.
- "experience_years" must be an integer. If unclear, estimate from context.
- "skills" should focus on technical/professional skills, comma-separated.
- "resume_text" should be a cleaned version of the resume — remove formatting artifacts, keep content.
- If a field cannot be determined, use reasonable defaults (e.g. name="Unknown Candidate", experience_years=0).
- Respond ONLY with valid JSON, no extra text.
"""


# ---------------------------------------------------------------------------
# File text extractors
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    from PyPDF2 import PdfReader
    from io import BytesIO

    reader = PdfReader(BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    from docx import Document
    from io import BytesIO

    doc = Document(BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain text file."""
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Route to the correct extractor based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif lower.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Use PDF, DOCX, or TXT.")


# ---------------------------------------------------------------------------
# GPT-4o-mini extraction
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def parse_resume_text(
    client: OpenAI,
    raw_text: str,
    model: str = LLM_MODEL,
) -> ResumeRecord:
    """
    Use GPT-4o-mini to extract structured resume data from raw text.
    Returns a validated ResumeRecord.
    """
    # Truncate very long resumes to stay within token limits
    truncated = raw_text[:8000] if len(raw_text) > 8000 else raw_text

    logger.info("Parsing resume text (%d chars)...", len(truncated))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": truncated},
        ],
        temperature=0.0,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )

    raw_json = response.choices[0].message.content or "{}"
    parsed = json.loads(raw_json)

    # Generate a unique ID
    record_id = f"R{uuid.uuid4().hex[:6].upper()}"

    record = ResumeRecord(
        id=record_id,
        name=parsed.get("name", "Unknown Candidate"),
        category=parsed.get("category", RESUME_CATEGORIES[0]),
        skills=parsed.get("skills", ""),
        experience_years=int(parsed.get("experience_years", 0)),
        education=parsed.get("education", "Not specified"),
        resume_text=parsed.get("resume_text", truncated[:500]),
    )

    logger.info("Parsed resume: %s (%s)", record.name, record.category)
    return record


def process_uploaded_file(
    client: OpenAI,
    file_bytes: bytes,
    filename: str,
) -> ResumeRecord:
    """
    Full pipeline: file bytes → extract text → GPT parse → ResumeRecord.
    """
    raw_text = extract_text(file_bytes, filename)
    if not raw_text.strip():
        raise ValueError(f"No text could be extracted from {filename}")
    return parse_resume_text(client, raw_text)
