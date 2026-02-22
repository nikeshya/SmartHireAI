"""
Streamlit UI for the SmartHire AI system.
Two sections:
  1. Upload Resumes — drag & drop PDF/DOCX/TXT files, auto-parsed and ingested
  2. Ask Questions — query the uploaded resumes with natural language
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import DATA_CSV_PATH, DEFAULT_TOP_K, RESUME_CATEGORIES, get_settings

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartHire AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 2rem; max-width: 1200px; }

    .hero-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem;
    }
    .hero-subtitle { color: #8892b0; font-size: 1.05rem; margin-bottom: 1.5rem; }

    .resume-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3d);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .resume-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    .card-name { font-size: 1.15rem; font-weight: 600; color: #ccd6f6; margin-bottom: 0.3rem; }
    .card-meta { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
    .badge {
        display: inline-block; padding: 0.2rem 0.65rem; border-radius: 20px;
        font-size: 0.78rem; font-weight: 500;
    }
    .badge-category { background: rgba(102,126,234,0.15); color: #667eea; border: 1px solid rgba(102,126,234,0.3); }
    .badge-score { background: rgba(100,255,218,0.1); color: #64ffda; border: 1px solid rgba(100,255,218,0.3); }
    .badge-exp { background: rgba(255,183,77,0.1); color: #ffb74d; border: 1px solid rgba(255,183,77,0.3); }
    .badge-success { background: rgba(76,175,80,0.15); color: #66bb6a; border: 1px solid rgba(76,175,80,0.3); }
    .card-skills { color: #8892b0; font-size: 0.88rem; line-height: 1.5; }

    .answer-box {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 1px solid rgba(100, 255, 218, 0.15); border-left: 4px solid #64ffda;
        border-radius: 10px; padding: 1.4rem 1.6rem; color: #c9d1d9;
        font-size: 0.95rem; line-height: 1.7; margin-bottom: 1rem;
    }

    .trace-container {
        background: linear-gradient(145deg, #1a1a2e, #222240);
        border: 1px solid rgba(102,126,234,0.15); border-radius: 10px; padding: 1.2rem 1.5rem;
    }
    .trace-row { display: flex; justify-content: space-between; padding: 0.45rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.88rem; }
    .trace-label { color: #8892b0; }
    .trace-value { color: #ccd6f6; font-weight: 500; }
    .trace-value-highlight { color: #64ffda; font-weight: 600; }

    .upload-status {
        background: linear-gradient(145deg, #1a2e1a, #223322);
        border: 1px solid rgba(100,255,100,0.15); border-radius: 10px;
        padding: 1rem 1.3rem; margin: 0.5rem 0;
    }

    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }
    section[data-testid="stSidebar"] .stMarkdown p { color: #8892b0; }

    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px; padding: 0.5rem 1.8rem;
        font-weight: 600; font-size: 0.9rem; transition: all 0.3s;
        white-space: nowrap; min-width: fit-content;
    }
    div.stButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 6px 20px rgba(102,126,234,0.35);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    top_k = st.slider("Number of results (top-K)", 1, 10, DEFAULT_TOP_K)

    st.markdown("---")
    st.markdown("### Categories")
    for cat in RESUME_CATEGORIES:
        st.markdown(f"- {cat}")

    st.markdown("---")

    # Show how many resumes are loaded
    csv_count = 0
    if DATA_CSV_PATH.exists():
        try:
            df = pd.read_csv(DATA_CSV_PATH)
            csv_count = len(df)
        except Exception:
            pass
    st.markdown(f"### Resumes Loaded: **{csv_count}**")

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown(
        """
    1. **Upload** resume files (PDF/DOCX/TXT)
    2. AI **auto-parses** name, skills, category
    3. Resumes **embedded** & stored in Pinecone
    4. **Ask questions** in natural language
    """
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">SmartHire AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    "Upload resumes, AI auto-parses & indexes them, ask anything about candidates in natural language"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tab layout: Upload | Ask Questions
# ---------------------------------------------------------------------------
tab_upload, tab_query = st.tabs(["Upload Resumes", "Ask Questions"])

# ============================= UPLOAD TAB =================================
with tab_upload:
    st.markdown("### Drop your resume files here")
    st.markdown("Supported formats: **PDF**, **DOCX**, **TXT** — upload as many as you want.")

    uploaded_files = st.file_uploader(
        "Upload resume files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")

        if st.button("Process & Ingest All Resumes", use_container_width=True):
            try:
                settings = get_settings()
                from openai import OpenAI
                from pinecone import Pinecone

                from ingestion import generate_embeddings, get_or_create_index
                from resume_parser import process_uploaded_file

                openai_client = OpenAI(api_key=settings.openai_api_key)
                pc = Pinecone(api_key=settings.pinecone_api_key)

                # Ensure index exists
                get_or_create_index(pc, settings.pinecone_index_name)
                index = pc.Index(settings.pinecone_index_name)

                # Load existing CSV (or create new)
                existing_records = []
                if DATA_CSV_PATH.exists():
                    try:
                        existing_df = pd.read_csv(DATA_CSV_PATH)
                        existing_records = existing_df.to_dict("records")
                    except Exception:
                        pass

                progress = st.progress(0, text="Starting...")
                success_count = 0
                new_records = []

                for i, uploaded_file in enumerate(uploaded_files):
                    progress.progress(
                        (i) / len(uploaded_files),
                        text=f"Processing {uploaded_file.name}...",
                    )

                    try:
                        # 1. Parse resume with GPT
                        file_bytes = uploaded_file.read()
                        record = process_uploaded_file(
                            openai_client, file_bytes, uploaded_file.name
                        )

                        # 2. Embed the resume text
                        embeddings = generate_embeddings(
                            openai_client, [record.resume_text]
                        )

                        # 3. Upsert to Pinecone
                        index.upsert(
                            vectors=[
                                {
                                    "id": record.id,
                                    "values": embeddings[0],
                                    "metadata": {
                                        "name": record.name,
                                        "category": record.category,
                                        "skills": record.skills,
                                        "experience_years": record.experience_years,
                                        "education": record.education,
                                        "resume_text": record.resume_text,
                                    },
                                }
                            ]
                        )

                        # 4. Add to CSV records
                        new_records.append(record.model_dump())
                        success_count += 1

                        st.markdown(
                            f"""<div class="resume-card">
                                <div class="card-name">{record.name} — Ingested</div>
                                <div class="card-meta">
                                    <span class="badge badge-category">{record.category}</span>
                                    <span class="badge badge-exp">{record.experience_years} yrs</span>
                                    <span class="badge badge-success">Done</span>
                                </div>
                                <div class="card-skills"><strong>Skills:</strong> {record.skills}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")

                progress.progress(1.0, text="Done!")

                # 5. Save all records to CSV
                if new_records:
                    all_records = existing_records + new_records
                    all_df = pd.DataFrame(all_records)
                    DATA_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                    all_df.to_csv(DATA_CSV_PATH, index=False)

                st.success(
                    f"Done! {success_count}/{len(uploaded_files)} resumes processed, "
                    f"embedded, and stored in Pinecone. Total resumes in system: "
                    f"{len(existing_records) + len(new_records)}"
                )
                st.info("Now go to the **Ask Questions** tab and start querying.")

            except ValueError as e:
                st.error(f"Configuration Error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

    # Show current CSV contents
    st.markdown("---")
    st.markdown("### Currently Loaded Resumes")
    if DATA_CSV_PATH.exists():
        try:
            df = pd.read_csv(DATA_CSV_PATH)
            if not df.empty:
                display_df = df[["id", "name", "category", "skills", "experience_years"]].copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No resumes loaded yet. Upload some files above.")
        except Exception:
            st.info("No resumes loaded yet. Upload some files above.")
    else:
        st.info("No resumes loaded yet. Upload some files above.")


# ============================= QUERY TAB ==================================
with tab_query:
    st.markdown("### Ask anything about your candidates")

    query = st.text_input(
        "Your question",
        placeholder="e.g. Find data scientists with experience in deep learning and NLP",
        label_visibility="collapsed",
    )

    search_clicked = st.button("Search")

    # Session state for history
    if "history" not in st.session_state:
        st.session_state.history = []

    if search_clicked and query:
        try:
            with st.spinner("Classifying, searching, generating answer..."):
                from pipeline import run_pipeline

                result = run_pipeline(query, top_k=top_k)

            st.session_state.history.append(result)

            # -- Answer --
            st.markdown("### Answer")
            st.markdown(
                f'<div class="answer-box">{result.answer.answer}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"**Predicted Category:** "
                f'<span class="badge badge-category">{result.classification.category}</span> '
                f"&nbsp; Confidence: **{result.classification.confidence:.0%}**",
                unsafe_allow_html=True,
            )

            # -- Retrieved documents --
            st.markdown("### Retrieved Resumes")
            if not result.retrieved_documents:
                st.info("No matching resumes found for this category.")
            else:
                for doc in result.retrieved_documents:
                    st.markdown(
                        f"""<div class="resume-card">
                            <div class="card-name">{doc.name}</div>
                            <div class="card-meta">
                                <span class="badge badge-category">{doc.category}</span>
                                <span class="badge badge-score">Score: {doc.score:.4f}</span>
                                <span class="badge badge-exp">{doc.experience_years} yrs exp</span>
                            </div>
                            <div class="card-skills"><strong>Skills:</strong> {doc.skills}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    with st.expander(f"Full resume — {doc.name}"):
                        st.write(doc.resume_text)

            # -- Pipeline trace --
            st.markdown("### Pipeline Trace")
            trace = result.trace
            st.markdown(
                f"""<div class="trace-container">
                    <div class="trace-row"><span class="trace-label">Total Time</span>
                        <span class="trace-value-highlight">{trace.total_time_ms:.0f} ms</span></div>
                    <div class="trace-row"><span class="trace-label">Classification</span>
                        <span class="trace-value">{trace.classification_time_ms:.0f} ms</span></div>
                    <div class="trace-row"><span class="trace-label">Embedding</span>
                        <span class="trace-value">{trace.embedding_time_ms:.0f} ms</span></div>
                    <div class="trace-row"><span class="trace-label">Retrieval</span>
                        <span class="trace-value">{trace.retrieval_time_ms:.0f} ms</span></div>
                    <div class="trace-row"><span class="trace-label">Generation</span>
                        <span class="trace-value">{trace.generation_time_ms:.0f} ms</span></div>
                    <div class="trace-row"><span class="trace-label">Docs Found</span>
                        <span class="trace-value">{trace.documents_found} / {trace.top_k}</span></div>
                    <div class="trace-row"><span class="trace-label">Models</span>
                        <span class="trace-value">{trace.embedding_model} + {trace.llm_model}</span></div>
                    <div class="trace-row"><span class="trace-label">Tokens</span>
                        <span class="trace-value">{result.answer.prompt_tokens} prompt + {result.answer.completion_tokens} completion</span></div>
                </div>""",
                unsafe_allow_html=True,
            )

        except ValueError as e:
            st.error(f"Configuration Error: {e}")
        except Exception as e:
            st.error(f"Pipeline Error: {e}")
            st.exception(e)

    elif search_clicked and not query:
        st.warning("Please enter a query before searching.")

    # -- History --
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("### Query History")
        for i, past in enumerate(reversed(st.session_state.history[:-1]), 1):
            with st.expander(f"Query {i}: {past.query[:60]}..."):
                st.markdown(f"**Category:** {past.classification.category}")
                st.markdown(f"**Answer:** {past.answer.answer[:300]}...")
                st.markdown(f"**Time:** {past.trace.total_time_ms:.0f} ms")
