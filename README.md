# SmartHire AI — Intelligent Resume Search

A production-ready **RAG (Retrieval-Augmented Generation)** system for intelligent resume search and candidate analysis. Upload resumes → AI auto-parses them → Ask anything about candidates in natural language.

**Tech Stack:** OpenAI Embeddings · Pinecone Vector Search · GPT-4o-mini · Metadata Filtering · Streamlit · Pydantic

## ✨ Features

- **📤 Drag & Drop Upload** — Upload PDF/DOCX/TXT resume files, AI extracts all structured data automatically
- **🏷️ Auto-Classification** — GPT-4o-mini classifies each query into the right category
- **🔍 Semantic Search** — Finds relevant candidates even with different wording (not just keyword matching)
- **🧬 Metadata Filtering** — Searches only within the relevant category for precise results
- **💡 AI Answers** — Generates detailed, grounded answers referencing actual resume data
- **🔬 Pipeline Trace** — Full transparency into every step with timing metrics

## Architecture

```
User Query → Category Classification (GPT-4o-mini)
           → Query Embedding (text-embedding-3-small)
           → Pinecone Vector Search (filtered by category)
           → Answer Generation (GPT-4o-mini with retrieved context)
           → Streamlit UI (answer + documents + pipeline trace)
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env with your real API keys
```

### 3. Run the App
```bash
streamlit run app.py
```

Upload resume files through the UI — the system handles everything automatically!

## Resume Categories

| Category | Examples |
|----------|---------|
| Data Science | ML, analytics, statistics |
| Web Development | React, Django, Node.js |
| DevOps | Kubernetes, Docker, CI/CD |
| Mobile Development | Flutter, React Native, Swift |
| Machine Learning | PyTorch, NLP, Computer Vision |
| Cloud Engineering | AWS, GCP, Azure |

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (upload + query tabs) |
| `config.py` | Settings & environment config |
| `models.py` | Pydantic schemas for all pipeline stages |
| `resume_parser.py` | GPT-powered resume text extraction |
| `ingestion.py` | Embedding & Pinecone upsert pipeline |
| `classifier.py` | Query → category classification |
| `retriever.py` | Vector search with metadata filter |
| `generator.py` | Context-grounded answer generation |
| `pipeline.py` | Full pipeline orchestration with timing |

## Deployment

Deployed on **Streamlit Community Cloud**. API keys are managed via Streamlit Secrets.

## License

MIT
