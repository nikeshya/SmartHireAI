"""
Data ingestion pipeline: CSV → OpenAI embeddings → Pinecone upsert.
Reads the resume CSV, validates each row, generates embeddings, and upserts
vectors with metadata into the Pinecone index.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from config import (
    DATA_CSV_PATH,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    PINECONE_METRIC,
    get_settings,
)
from models import ResumeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_validate_csv(csv_path: Path = DATA_CSV_PATH) -> list[ResumeRecord]:
    """Load the CSV and validate every row with Pydantic."""
    df = pd.read_csv(csv_path)
    records: list[ResumeRecord] = []
    errors: list[str] = []

    for idx, row in df.iterrows():
        try:
            record = ResumeRecord(**row.to_dict())
            records.append(record)
        except Exception as e:
            errors.append(f"Row {idx}: {e}")

    if errors:
        print(f"⚠️  Validation warnings ({len(errors)} rows skipped):")
        for err in errors:
            print(f"   {err}")

    print(f"✅ Validated {len(records)} / {len(df)} resume records.")
    return records


def get_or_create_index(pc: Pinecone, index_name: str) -> None:
    """Create Pinecone index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        print(f"📌 Index '{index_name}' already exists.")
        return

    print(f"🆕 Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSIONS,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        print("   ⏳ Waiting for index to be ready...")
        time.sleep(2)
    print(f"   ✅ Index '{index_name}' is ready.")


def generate_embeddings(
    client: OpenAI, texts: list[str], model: str = EMBEDDING_MODEL
) -> list[list[float]]:
    """Generate embeddings for a batch of texts using OpenAI."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Main Ingestion
# ---------------------------------------------------------------------------

def ingest(csv_path: Path = DATA_CSV_PATH, batch_size: int = 10) -> None:
    """
    End-to-end ingestion: validate CSV → embed → upsert to Pinecone.
    Idempotent — skips records that already exist in the index.
    """
    settings = get_settings()
    openai_client = OpenAI(api_key=settings.openai_api_key)
    pc = Pinecone(api_key=settings.pinecone_api_key)

    # 1. Validate CSV
    records = load_and_validate_csv(csv_path)
    if not records:
        print("❌ No valid records to ingest.")
        return

    # 2. Ensure index exists
    get_or_create_index(pc, settings.pinecone_index_name)
    index = pc.Index(settings.pinecone_index_name)

    # 3. Check which records already exist
    existing_ids = set()
    try:
        # Fetch existing IDs to avoid duplicates
        all_ids = [r.id for r in records]
        fetch_result = index.fetch(ids=all_ids)
        existing_ids = set(fetch_result.vectors.keys())
        if existing_ids:
            print(f"⏭️  Skipping {len(existing_ids)} already-ingested records.")
    except Exception:
        pass  # Index might be empty

    new_records = [r for r in records if r.id not in existing_ids]
    if not new_records:
        print("✅ All records already ingested. Nothing to do.")
        return

    print(f"📤 Ingesting {len(new_records)} new records...")

    # 4. Batch embed and upsert
    for i in range(0, len(new_records), batch_size):
        batch = new_records[i : i + batch_size]
        texts = [r.resume_text for r in batch]

        print(f"   Batch {i // batch_size + 1}: Embedding {len(batch)} records...")
        embeddings = generate_embeddings(openai_client, texts)

        vectors = []
        for record, embedding in zip(batch, embeddings):
            vectors.append(
                {
                    "id": record.id,
                    "values": embedding,
                    "metadata": {
                        "name": record.name,
                        "category": record.category,
                        "skills": record.skills,
                        "experience_years": record.experience_years,
                        "education": record.education,
                        "resume_text": record.resume_text,
                    },
                }
            )

        index.upsert(vectors=vectors)
        print(f"   ✅ Upserted {len(batch)} vectors.")

    # 5. Summary
    stats = index.describe_index_stats()
    print(f"\n🎉 Ingestion complete! Total vectors in index: {stats.total_vector_count}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        ingest()
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        sys.exit(1)
