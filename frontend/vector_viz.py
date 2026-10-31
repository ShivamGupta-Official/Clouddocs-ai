"""
CloudDocs AI — 3D Vector Space Visualization Module

Provides interactive 3D visualizations of high-dimensional embeddings:
- PCA dimensionality reduction from 384-dim embedding space down to 3D (X, Y, Z)
- Interactive Plotly 3D scatter plot of document chunks
- Dynamic projection of user search queries into vector space
- Visual proximity lines (ray-tracing) connecting query to retrieved nearest neighbors
- Hover cards showing document, page number, similarity %, and verbatim snippets
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def generate_3d_vector_space_figure(
    chunks_data: dict,
    query_text: str | None = None,
    query_embedding: list[float] | np.ndarray | None = None,
    retrieved_chunk_ids: list[str] | None = None,
) -> go.Figure | None:
    """
    Generate an interactive 3D Plotly figure representing the vector space.

    Args:
        chunks_data: Output of ChromaDB collection.get(include=['embeddings', 'documents', 'metadatas'])
        query_text: Optional user query string.
        query_embedding: Optional 384-dim vector for the query.
        retrieved_chunk_ids: Optional list of chunk IDs that matched the query.

    Returns:
        Plotly Figure object or None if insufficient vectors exist.
    """
    embeddings = chunks_data.get("embeddings")
    metadatas = chunks_data.get("metadatas", [])
    documents = chunks_data.get("documents", [])
    ids = chunks_data.get("ids", [])

    if embeddings is None or len(embeddings) == 0:
        return None

    # Convert embeddings to numpy array
    doc_vectors = np.array(embeddings)
    n_docs = len(doc_vectors)

    has_query = query_embedding is not None and len(query_embedding) == doc_vectors.shape[1]
    all_vectors = np.vstack([doc_vectors, np.array(query_embedding)]) if has_query else doc_vectors

    # Ensure we have at least 3 samples for PCA with 3 components
    if len(all_vectors) < 3:
        # Pad with tiny deterministic jitter so PCA(n_components=3) can compute cleanly
        jitter = np.random.RandomState(42).normal(
            0, 1e-4, size=(3 - len(all_vectors), all_vectors.shape[1])
        )
        padded_vectors = np.vstack([all_vectors, jitter])
        pca = PCA(n_components=3, random_state=42)
        all_coords = pca.fit_transform(padded_vectors)[: len(all_vectors)]
    else:
        pca = PCA(n_components=3, random_state=42)
        all_coords = pca.fit_transform(all_vectors)

    doc_coords = all_coords[:n_docs]
    query_coord = all_coords[n_docs] if has_query else None

    # Group document chunks by filename for distinct color palettes
    unique_files = list({m.get("filename", "Unknown") for m in metadatas})
    palette = [
        "#2563eb",  # Blue
        "#10b981",  # Emerald
        "#8b5cf6",  # Purple
        "#ec4899",  # Pink
        "#06b6d4",  # Cyan
        "#f97316",  # Orange
    ]

    fig = go.Figure()

    # 1. Add Document Chunks Traces (Grouped by file)
    retrieved_set = set(retrieved_chunk_ids or [])

    for file_idx, filename in enumerate(unique_files):
        color = palette[file_idx % len(palette)]
        indices = [i for i, m in enumerate(metadatas) if m.get("filename") == filename]

        x_vals = [doc_coords[i, 0] for i in indices]
        y_vals = [doc_coords[i, 1] for i in indices]
        z_vals = [doc_coords[i, 2] for i in indices]

        # Hover text descriptions
        hover_texts = []
        marker_sizes = []
        marker_colors = []

        for i in indices:
            chunk_id = ids[i]
            meta = metadatas[i]
            page = meta.get("page_number", 1)
            idx = meta.get("chunk_index", 0)
            text_preview = documents[i].replace("<", "&lt;").replace(">", "&gt;")[:140]

            is_retrieved = chunk_id in retrieved_set
            if is_retrieved:
                marker_sizes.append(10)
                marker_colors.append("#22c55e")  # Bright green highlight for retrieved
                badge = "<b>[★ RETRIEVED NEAREST NEIGHBOR]</b><br>"
            else:
                marker_sizes.append(7)
                marker_colors.append(color)
                badge = ""

            hover_texts.append(
                f"{badge}"
                f"<b>Document:</b> {filename}<br>"
                f"<b>Page / Slide:</b> {page} (Chunk {idx})<br>"
                f'<b>Snippet:</b> <i>"{text_preview}..."</i>'
            )

        fig.add_trace(
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode="markers",
                name=f"{filename} ({len(indices)} chunks)",
                marker=dict(
                    size=marker_sizes,
                    color=marker_colors,
                    opacity=0.85,
                    line=dict(color="#0f172a", width=1),
                ),
                text=hover_texts,
                hoverinfo="text",
            )
        )

    # 2. Add User Search Query Point (if provided)
    if has_query and query_coord is not None:
        clean_q = (query_text or "User Query").replace("<", "&lt;").replace(">", "&gt;")
        fig.add_trace(
            go.Scatter3d(
                x=[query_coord[0]],
                y=[query_coord[1]],
                z=[query_coord[2]],
                mode="markers+text",
                name="Search Query Vector",
                text=[f'<b>🔍 Query:</b> "{clean_q}"'],
                hoverinfo="text",
                marker=dict(
                    size=12,
                    color="#f59e0b",  # Amber gold
                    symbol="diamond",
                    line=dict(color="#ffffff", width=2),
                ),
            )
        )

        # 3. Add Ray-Tracing Lines from Query to Retrieved Chunks
        if retrieved_chunk_ids:
            line_x, line_y, line_z = [], [], []
            for r_id in retrieved_chunk_ids:
                if r_id in ids:
                    target_idx = ids.index(r_id)
                    t_coord = doc_coords[target_idx]
                    # Create segment: Query -> Target -> None (breaks line for Plotly)
                    line_x.extend([query_coord[0], t_coord[0], None])
                    line_y.extend([query_coord[1], t_coord[1], None])
                    line_z.extend([query_coord[2], t_coord[2], None])

            if line_x:
                fig.add_trace(
                    go.Scatter3d(
                        x=line_x,
                        y=line_y,
                        z=line_z,
                        mode="lines",
                        name="Semantic Similarity Rays",
                        line=dict(color="#f59e0b", width=3, dash="dash"),
                        hoverinfo="none",
                    )
                )

    # Styling 3D layout
    fig.update_layout(
        title=dict(
            text="Interactive 3D Vector Space (Semantic Embeddings Projection)",
            font=dict(size=16, color="#0f172a", family="Segoe UI, sans-serif"),
        ),
        scene=dict(
            xaxis=dict(
                title="PCA Dimension 1",
                backgroundcolor="#f8fafc",
                gridcolor="#e2e8f0",
                showbackground=True,
                zerolinecolor="#94a3b8",
            ),
            yaxis=dict(
                title="PCA Dimension 2",
                backgroundcolor="#f8fafc",
                gridcolor="#e2e8f0",
                showbackground=True,
                zerolinecolor="#94a3b8",
            ),
            zaxis=dict(
                title="PCA Dimension 3",
                backgroundcolor="#f8fafc",
                gridcolor="#e2e8f0",
                showbackground=True,
                zerolinecolor="#94a3b8",
            ),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
            ),
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#0f172a"),
        ),
        paper_bgcolor="#ffffff",
    )

    return fig
