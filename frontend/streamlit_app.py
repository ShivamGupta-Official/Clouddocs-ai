"""
CloudDocs AI — Streamlit User Interface

A professional, minimal, document-focused UI:
- Dashboard: Real document and chunk metrics
- Documents: File upload (PDF/TXT), status monitoring, document deletion
- AI Chat: Grounded conversation with expandable genuine source citations
- Settings & System Status: Live component health and model configuration
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import streamlit as st
from api_client import CloudDocsAPIClient
from vector_viz import generate_3d_vector_space_figure

from app.config import get_settings
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import VectorStore


@st.cache_resource
def get_viz_engine():
    """Cache embedding engine and ChromaDB vector store for instant 3D projection."""
    settings = get_settings()
    vs = VectorStore(settings.chroma_dir_path)
    engine = EmbeddingEngine(settings.embedding_model)
    return vs, engine


# Page configuration
st.set_page_config(
    page_title="CloudDocs AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom high-contrast CSS styling
st.markdown(
    """
    <style>
    /* Clean, high-contrast application styling */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Headings and titles */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    p, span, label, div {
        color: #1e293b;
    }

    /* Metric cards: ensure dark bold text on white cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }

    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] label,
    div[data-testid="stMetricLabel"] div {
        color: #475569 !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] div,
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
    }

    /* Action buttons: high contrast */
    div.stButton > button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease-in-out !important;
    }

    div.stButton > button:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    button[kind="primary"],
    div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        font-weight: 600 !important;
    }

    button[kind="primary"]:hover,
    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }

    /* Citation box */
    .citation-box {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 4px solid #2563eb !important;
        padding: 14px 18px !important;
        margin-top: 10px !important;
        margin-bottom: 10px !important;
        border-radius: 6px !important;
        font-size: 0.92rem !important;
        color: #0f172a !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    }

    .citation-box b {
        color: #1d4ed8 !important;
        font-weight: 600 !important;
    }

    .citation-box i {
        color: #64748b !important;
    }

    .citation-box p {
        color: #334155 !important;
        margin-top: 6px !important;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        color: #0f172a !important;
        margin-bottom: 12px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div {
        color: #0f172a !important;
    }

    /* Status badges */
    .status-completed {
        background-color: #dcfce7 !important;
        color: #15803d !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85em !important;
        display: inline-block !important;
    }

    .status-failed {
        background-color: #fee2e2 !important;
        color: #b91c1c !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85em !important;
        display: inline-block !important;
    }

    .status-processing {
        background-color: #fef3c7 !important;
        color: #b45309 !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85em !important;
        display: inline-block !important;
    }

    /* Expanders & Accordions */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] details summary span {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Sidebar high contrast */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize API client
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
client = CloudDocsAPIClient(base_url=API_URL)

# Initialize session state for navigation and chat history
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar navigation
with st.sidebar:
    st.title("📄 CloudDocs AI")
    st.caption("Retrieval-Augmented Generation")

    st.markdown("---")

    page_options = ["Dashboard", "Documents", "AI Chat", "3D Vector Space", "Settings & Status"]
    current_idx = (
        page_options.index(st.session_state.current_page)
        if st.session_state.current_page in page_options
        else 0
    )
    selected_page = st.radio("Navigation", page_options, index=current_idx)
    st.session_state.current_page = selected_page

    st.markdown("---")
    # Quick health check in sidebar
    health = client.get_health()
    if health["success"]:
        st.markdown("🟢 **Backend:** Connected")
    else:
        st.markdown("🔴 **Backend:** Offline")
        st.caption("Ensure FastAPI is running at port 8000")


# ==============================================================================
# SCREEN 1: DASHBOARD
# ==============================================================================
if st.session_state.current_page == "Dashboard":
    st.title("System Dashboard")
    st.markdown(
        "Welcome to **CloudDocs AI**. Upload PDF and TXT documents to build a verified, "
        "local vector knowledge base, and ask questions with strict source attribution."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Fetch real document metrics
    doc_res = client.list_documents()
    docs = doc_res["data"] if doc_res["success"] else []

    total_docs = len(docs)
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    completed_docs = sum(1 for d in docs if d.get("status") == "COMPLETED")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Indexed Chunks", total_chunks)
    with col3:
        st.metric("Ready for Search", completed_docs)
    with col4:
        st.metric("API Status", "Healthy" if health["success"] else "Unavailable")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Recent Documents")
        if not docs:
            st.info("No documents indexed yet. Upload a PDF or TXT document to begin.")
        else:
            recent_docs = docs[:5]
            for doc in recent_docs:
                status_class = f"status-{doc.get('status', 'processing').lower()}"
                st.markdown(
                    f"**{doc.get('filename')}** &nbsp; "
                    f"<span class='{status_class}'>{doc.get('status')}</span> &nbsp; "
                    f"*{doc.get('chunk_count', 0)} chunks*",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"ID: {doc.get('id')} • Size: {doc.get('file_size_bytes', 0) / 1024:.1f} KB"
                )
                st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

    with col_right:
        st.subheader("Quick Actions")
        st.markdown("Get started with your workflow:")
        if st.button("📁 Go to Document Library", use_container_width=True):
            st.session_state.current_page = "Documents"
            st.rerun()

        if st.button("💬 Start AI Question-Answering", use_container_width=True):
            st.session_state.current_page = "AI Chat"
            st.rerun()


# ==============================================================================
# SCREEN 2: DOCUMENTS (UPLOAD & MANAGEMENT)
# ==============================================================================
elif st.session_state.current_page == "Documents":
    st.title("Document Library")
    st.markdown("Manage uploaded documents, extract text, and index vector embeddings.")

    # Upload section
    with st.expander("⬆️ Upload New Document", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=["pdf", "txt"],
            help="Maximum size: 50MB. Digital text required (scanned PDFs without OCR will be flagged).",
        )

        if uploaded_file is not None and st.button("Upload and Index Document", type="primary"):
            with st.spinner(f"Extracting and indexing '{uploaded_file.name}'..."):
                file_bytes = uploaded_file.read()
                result = client.upload_document(uploaded_file.name, file_bytes)

                if result["success"]:
                    doc_data = result["data"]
                    st.success(
                        f"✅ Successfully indexed '{doc_data.get('filename')}' with {doc_data.get('chunk_count')} chunks!"
                    )
                    st.rerun()
                else:
                    st.error(f"❌ Ingestion failed: {result.get('error')}")

    st.markdown("---")

    # Document list table
    st.subheader("Indexed Documents")
    doc_res = client.list_documents()

    if not doc_res["success"]:
        st.error(f"Failed to load documents: {doc_res.get('error')}")
    else:
        docs = doc_res["data"]
        if not docs:
            st.info("No documents currently stored. Upload your first document above.")
        else:
            for d in docs:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                    with col1:
                        st.markdown(f"**{d.get('filename')}**")
                        st.caption(f"ID: {d.get('id')}")
                    with col2:
                        st.write(d.get("file_type"))
                    with col3:
                        size_kb = d.get("file_size_bytes", 0) / 1024
                        st.write(f"{size_kb:.1f} KB")
                    with col4:
                        st.write(f"{d.get('chunk_count', 0)} chunks")
                    with col5:
                        # Delete action
                        if st.button("Delete", key=f"del_{d.get('id')}"):
                            del_res = client.delete_document(d.get("id"))
                            if del_res["success"]:
                                st.success(f"Deleted '{d.get('filename')}'.")
                                st.rerun()
                            else:
                                st.error(f"Failed to delete: {del_res.get('error')}")

                    st.markdown("<hr style='margin: 4px 0 12px 0;'>", unsafe_allow_html=True)


# ==============================================================================
# SCREEN 3: AI CHAT
# ==============================================================================
elif st.session_state.current_page == "AI Chat":
    st.title("Document Q&A Chat")
    st.markdown("Ask questions grounded strictly in your indexed documents.")

    # Control parameters in expander
    with st.expander("Search Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Max Passages (Top-K)", min_value=1, max_value=10, value=4)
        with col2:
            threshold = st.slider(
                "Relevance Threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.05
            )

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # If message contains citations, render source inspection
            if "citations" in msg and msg["citations"]:
                with st.expander("📚 Source Citations & Evidence", expanded=False):
                    for cite in msg["citations"]:
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <b>[Source {cite.get("citation_index")}] {cite.get("filename")}</b> — Page {cite.get("page_number")}
                                <i>(Relevance: {cite.get("relevance_score", 0):.2%})</i><br>
                                <p style="margin-top: 6px; color: #374151;">"{cite.get("snippet")}"</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            if msg.get("insufficient_evidence", False):
                st.warning(
                    "⚠️ The uploaded documents do not contain sufficient evidence to answer this question."
                )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response via backend API
        with (
            st.chat_message("assistant"),
            st.spinner("Searching documents and formulating response..."),
        ):
            response = client.ask_question(
                question=prompt,
                top_k=top_k,
                threshold=threshold,
            )

            if response["success"]:
                data = response["data"]
                answer = data.get("answer", "")
                citations = data.get("citations", [])
                has_evidence = data.get("has_sufficient_evidence", True)
                latency = data.get("latency_seconds", 0)

                st.markdown(answer)

                if citations:
                    with st.expander("📚 Source Citations & Evidence", expanded=True):
                        for cite in citations:
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <b>[Source {cite.get("citation_index")}] {cite.get("filename")}</b> — Page {cite.get("page_number")}
                                    <i>(Relevance: {cite.get("relevance_score", 0):.2%})</i><br>
                                    <p style="margin-top: 6px; color: #374151;">"{cite.get("snippet")}"</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with st.expander("🌐 Live 3D Vector Space & Similarity Radar", expanded=False):
                        try:
                            vs, engine = get_viz_engine()
                            v_data = vs.collection.get(
                                include=["embeddings", "documents", "metadatas"]
                            )
                            q_emb = engine.embed_query(prompt)
                            retrieved_ids = [
                                c.get("chunk_id") for c in citations if c.get("chunk_id")
                            ]
                            fig_chat_3d = generate_3d_vector_space_figure(
                                chunks_data=v_data,
                                query_text=prompt,
                                query_embedding=q_emb,
                                retrieved_chunk_ids=retrieved_ids,
                            )
                            if fig_chat_3d:
                                st.plotly_chart(fig_chat_3d, use_container_width=True)
                                st.caption(
                                    "💡 Rotate with mouse, scroll to zoom, and hover over 3D coordinates to explore semantic proximity."
                                )
                        except Exception as e:
                            st.warning(f"Could not load 3D projection: {e}")

                if not has_evidence:
                    st.warning(
                        "⚠️ Insufficient evidence in uploaded documents to verify this answer."
                    )

                st.caption(f"Provider: {data.get('llm_provider')} • Latency: {latency:.2f}s")

                # Save to conversation state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "insufficient_evidence": not has_evidence,
                    }
                )
            else:
                error_msg = f"Failed to retrieve answer: {response.get('error')}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                    }
                )

    # Clear chat button
    if st.session_state.messages and st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()


# ==============================================================================
# SCREEN 4: 3D VECTOR SPACE & SEMANTIC RADAR
# ==============================================================================
elif st.session_state.current_page == "3D Vector Space":
    st.title("🌐 3D Vector Space & Semantic Radar")
    st.markdown(
        "Interactive 3D geometric visualization of high-dimensional document embeddings. "
        "See how 384-dimensional semantic vectors are reduced using Principal Component Analysis (PCA) "
        "and how search queries locate nearest neighbors in geometric space."
    )

    try:
        vs, engine = get_viz_engine()
        v_data = vs.collection.get(include=["embeddings", "documents", "metadatas"])
        chunks_count = len(v_data.get("ids", []))

        if chunks_count == 0:
            st.info(
                "No document vectors indexed yet. Upload a PDF or TXT file in Document Library to explore the 3D vector space."
            )
        else:
            col_search, col_stats = st.columns([3, 1])
            with col_search:
                test_query = st.text_input(
                    "🔍 Enter a question or concept to project into 3D space:",
                    value=st.session_state.get(
                        "last_query", "cyber threats in unidirectional traffic"
                    ),
                    placeholder="e.g. passive monitoring, problem statement, zero-return...",
                )
            with col_stats:
                top_k_viz = st.slider(
                    "Nearest Neighbors (K)",
                    min_value=1,
                    max_value=min(10, chunks_count),
                    value=min(4, chunks_count),
                )

            query_emb = None
            retrieved_ids = []

            if test_query and test_query.strip():
                st.session_state.last_query = test_query.strip()
                with st.spinner("Projecting query into 3D vector space..."):
                    query_emb = engine.embed_query(test_query.strip())
                    search_res = vs.search(query_emb, top_k=top_k_viz)
                    retrieved_ids = [r["chunk_id"] for r in search_res]

            fig_3d = generate_3d_vector_space_figure(
                chunks_data=v_data,
                query_text=test_query.strip() if test_query else None,
                query_embedding=query_emb,
                retrieved_chunk_ids=retrieved_ids,
            )

            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
                st.caption(
                    "🖱️ **3D Controls:** Left-click + drag to rotate • Right-click + drag to pan • "
                    "Scroll to zoom in/out • Hover over any node to inspect text and slide numbers."
                )

            # Educational breakdown & raw vector inspector
            st.markdown("---")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.subheader("💡 How Vector Similarity Works")
                st.markdown(
                    """
                    1. **Dense Embeddings**: Each text chunk is converted into a 384-dimensional vector where semantically related concepts point in nearly identical geometric directions.
                    2. **Cosine Similarity**: Distance between question and document vectors is calculated geometrically:
                       $$\\text{Cosine Similarity} = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|}$$
                    3. **PCA Dimensionality Reduction**: Projects the 384-D coordinates into 3 principal orthogonal axes ($X, Y, Z$) preserving maximal semantic variance.
                    """
                )
            with col_e2:
                st.subheader("🔬 Raw 384-D Vector Inspector")
                if query_emb is not None:
                    st.write(f"**Query Vector Dimension:** `{len(query_emb)} floats`")
                    st.write(f"**First 8 Dimensions:** `{np.round(query_emb[:8], 4).tolist()}`")
                    st.write(f"**Vector L2 Norm:** `{np.linalg.norm(query_emb):.4f}`")
                st.write(f"**Indexed Document Vectors:** `{chunks_count}`")
                st.write("**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`")

    except Exception as e:
        st.error(f"Error initializing 3D visualization: {e}")


# ==============================================================================
# SCREEN 5: SETTINGS & SYSTEM STATUS
# ==============================================================================
elif st.session_state.current_page == "Settings & Status":
    st.title("Settings & System Status")
    st.markdown("Monitor real-time component health and configuration parameters.")

    health_res = client.get_health()

    if not health_res["success"]:
        st.error(f"Cannot reach backend: {health_res.get('error')}")
    else:
        hdata = health_res["data"]
        comps = hdata.get("components", {})
        config = hdata.get("configuration", {})

        st.subheader("Component Health")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("FastAPI Server", comps.get("api", "unknown"))
        with col2:
            st.metric("SQLite Metadata DB", comps.get("metadata_db", "unknown"))
        with col3:
            st.metric("ChromaDB Vector Store", comps.get("vector_store", "unknown"))
        with col4:
            st.metric("LLM Provider", comps.get("llm", "unknown"))

        st.markdown("---")

        st.subheader("Active Configuration")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write(f"**Embedding Model:** `{config.get('embedding_model')}`")
            st.write(
                f"**Configured LLM:** `{config.get('llm_provider')} ({config.get('llm_model')})`"
            )
            st.write(f"**Max Upload Size:** `{config.get('max_upload_size_mb')} MB`")
        with col_c2:
            st.write(f"**Chunk Size:** `{config.get('chunk_size')} characters`")
            st.write(f"**Chunk Overlap:** `{config.get('chunk_overlap')} characters`")
            st.write(f"**Default Top-K:** `{config.get('retrieval_top_k')} chunks`")

        st.markdown("---")
        st.subheader("Configuring External LLMs")
        st.markdown(
            """
            To enable generative answers via Google Gemini or OpenAI:
            1. Open your local `.env` file in the project root (`clouddocs-ai/.env`).
            2. To use **Google Gemini**:
               ```bash
               LLM_PROVIDER=gemini
               GEMINI_API_KEY=your_actual_gemini_api_key
               GEMINI_MODEL=gemini-2.0-flash
               ```
            3. To use **OpenAI**:
               ```bash
               LLM_PROVIDER=openai
               OPENAI_API_KEY=your_actual_openai_api_key
               OPENAI_MODEL=gpt-4o-mini
               ```
            4. Restart the backend server for changes to take effect.
            """
        )
