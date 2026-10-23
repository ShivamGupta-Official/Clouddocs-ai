"""
CloudDocs AI — Streamlit User Interface

A professional, minimal, document-focused UI:
- Dashboard: Real document and chunk metrics
- Documents: File upload (PDF/TXT), status monitoring, document deletion
- AI Chat: Grounded conversation with expandable genuine source citations
- Settings & System Status: Live component health and model configuration
"""

import os

import streamlit as st
from api_client import CloudDocsAPIClient

# Page configuration
st.set_page_config(
    page_title="CloudDocs AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal CSS styling
st.markdown(
    """
    <style>
    /* Clean, professional styling */
    .stApp {
        background-color: #fafafa;
        color: #1f2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Header typography */
    h1, h2, h3 {
        color: #111827;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }

    /* Citation box */
    .citation-box {
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        margin-top: 8px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9em;
    }

    /* Status badges */
    .status-completed {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85em;
    }
    .status-failed {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85em;
    }
    .status-processing {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85em;
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

    page_options = ["Dashboard", "Documents", "AI Chat", "Settings & Status"]
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
                "Relevance Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05
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
