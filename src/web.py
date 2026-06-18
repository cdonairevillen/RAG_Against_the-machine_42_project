import os
import sys
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROCESSED_DIR = "data/processed"
REPO_ROOT = "data/raw/vllm-0.10.1"
EVAL_DOCS = "data/output/search_results/dataset_docs_public.json"
EVAL_CODE = "data/output/search_results/dataset_code_public.json"
GT_DOCS = "data/datasets/AnsweredQuestions/dataset_docs_public.json"
GT_CODE = "data/datasets/AnsweredQuestions/dataset_code_public.json"

st.set_page_config(
    page_title="RAG against the Machine",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:\\
            wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0a0a0a;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0a0a0a;
    }

    [data-testid="stHeader"] {
        background-color: #0a0a0a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #0a0a0a;
    }

    /* Title */
    .rag-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ff3b3b;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .rag-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #555;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* Answer box */
    .answer-box {
        background-color: #111;
        border: 1px solid #1e1e1e;
        border-left: 3px solid #ff3b3b;
        border-radius: 4px;
        padding: 1.2rem 1.4rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        line-height: 1.7;
        color: #d0d0d0;
        margin-bottom: 1.5rem;
    }

    /* Source card */
    .source-card {
        background-color: #0f0f0f;
        border: 1px solid #1a1a1a;
        border-radius: 4px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }

    .source-path {
        color: #ff3b3b;
        font-weight: 700;
        margin-bottom: 0.3rem;
        word-break: break-all;
    }

    .source-meta {
        color: #444;
        margin-bottom: 0.4rem;
    }

    .source-text {
        color: #666;
        font-size: 0.7rem;
        line-height: 1.5;
        border-top: 1px solid #1a1a1a;
        padding-top: 0.4rem;
        margin-top: 0.4rem;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Metric card */
    .metric-card {
        background-color: #0f0f0f;
        border: 1px solid #1a1a1a;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #444;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.6rem;
    }

    .recall-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.4rem;
        gap: 0.6rem;
    }

    .recall-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #555;
        width: 70px;
        flex-shrink: 0;
    }

    .recall-bar-bg {
        flex: 1;
        height: 4px;
        background-color: #1a1a1a;
        border-radius: 2px;
        overflow: hidden;
    }

    .recall-bar-fill {
        height: 100%;
        background-color: #ff3b3b;
        border-radius: 2px;
    }

    .recall-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #888;
        width: 36px;
        text-align: right;
        flex-shrink: 0;
    }

    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #333;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.8rem;
        margin-top: 1.2rem;
    }

    .status-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #2ecc71;
        margin-right: 0.4rem;
    }

    .status-line {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #444;
        margin-bottom: 0.3rem;
    }

    /* Input */
    .stTextInput > div > div > input {
        background-color: #111 !important;
        border: 1px solid #222 !important;
        border-radius: 4px !important;
        color: #e0e0e0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        padding: 0.7rem 1rem !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #ff3b3b !important;
        box-shadow: none !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #333 !important;
    }

    /* Button */
    .stButton > button {
        background-color: #ff3b3b !important;
        color: #0a0a0a !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.4rem !important;
        letter-spacing: 0.05em !important;
    }

    .stButton > button:hover {
        background-color: #cc2f2f !important;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #1a1a1a;
        margin: 1.5rem 0;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #ff3b3b !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #111 !important;
        border-color: #222 !important;
        color: #e0e0e0 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


def load_retriever() -> object:
    """Load Retriever with reranker from disk.

    Returns:
        Loaded Retriever instance cached in session state.
    """
    from src.retriever import Retriever
    retriever = Retriever.from_disk(PROCESSED_DIR)
    retriever.load_reranker()
    return retriever


def load_generator() -> object:
    """Load Generator (Qwen3-0.6B) from HuggingFace cache.

    Returns:
        Loaded Generator instance cached in session state.
    """
    from src.generator import Generator
    return Generator(repo_root=REPO_ROOT)


def load_recall_metrics() -> dict:
    """Load pre-computed Recall@k metrics from evaluation output files.

    Returns:
        Dict with 'docs' and 'code' keys, each containing recall scores,
        or empty dicts if files are not found.
    """
    from src.evaluator import Evaluator

    metrics: dict = {"docs": {}, "code": {}}
    evaluator = Evaluator()

    for key, student_path, gt_path in [
        ("docs", EVAL_DOCS, GT_DOCS),
        ("code", EVAL_CODE, GT_CODE),
    ]:
        if os.path.isfile(student_path) and os.path.isfile(gt_path):
            try:
                result = evaluator.compute_recall(
                    student_path=student_path,
                    ground_truth_path=gt_path,
                    k=10,
                )
                metrics[key] = result.get("recall_at_k", {})
            except Exception:
                pass

    return metrics


def render_recall_bars(recall: dict, label: str,
                       col: DeltaGenerator) -> None:
    """Render a set of Recall@k bars in a Streamlit column.

    Args:
        recall: Dict mapping k (int) to recall score (float).
        label: Section label to display above the bars.
        col: Streamlit column object to render into.
    """
    with col:
        st.markdown(
            f'<div class="metric-label">{label}</div>',
            unsafe_allow_html=True
        )
        if not recall:
            st.markdown(
                '<div class="status-line">No eval data found.<br>'
                'Run make eval-docs / eval-code first.</div>',
                unsafe_allow_html=True
            )
            return

        for k, score in sorted(recall.items()):
            pct = int(score * 100)
            bar_width = int(score * 100)
            st.markdown(f"""
<div class="recall-row">
    <span class="recall-label">Recall@{k}</span>
    <div class="recall-bar-bg">
        <div class="recall-bar-fill" style="width:{bar_width}%"></div>
    </div>
    <span class="recall-value">{pct}%</span>
</div>""", unsafe_allow_html=True)


def render_source_card(index: int, file_path: str, first_char: int,
                       last_char: int, text_preview: str) -> None:
    """Render a single source chunk card.

    Args:
        index: 1-based position in the results list.
        file_path: Relative path to the source file.
        first_char: Start character index in the file.
        last_char: End character index in the file.
        text_preview: First 300 characters of the chunk text.
    """
    short_path = file_path.replace("data/raw/vllm-0.10.1/", "")
    preview = text_preview[:].replace("<", "&lt;").replace(">", "&gt;")

    st.markdown(f"""
<div class="source-card">
    <div class="source-path">[{index}] {short_path}</div>
    <div class="source-meta">chars {first_char}–{last_char}</div>
    <div class="source-text">{preview}...</div>
</div>""", unsafe_allow_html=True)


def main() -> None:
    """Main Streamlit application entry point."""

    #  Header
    st.markdown(
        '<div class="rag-title">⚡ RAG against the Machine</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="rag-subtitle">vLLM knowledge base · Qwen3-0.6B</div>',
        unsafe_allow_html=True
    )

    #  Load models (cached across reruns)
    if "retriever" not in st.session_state:
        with st.spinner("Loading retriever and reranker..."):
            try:
                st.session_state.retriever = load_retriever()
            except Exception as exc:
                st.error(
                    f"Could not load index: {exc}\n\n"
                    "Run `make index` before starting the web interface."
                )
                st.stop()

    if "generator" not in st.session_state:
        with st.spinner("Loading Qwen3-0.6B..."):
            try:
                st.session_state.generator = load_generator()
            except Exception as exc:
                st.error(f"Could not load generator: {exc}")
                st.stop()

    if "recall_metrics" not in st.session_state:
        st.session_state.recall_metrics = load_recall_metrics()

    retriever = st.session_state.retriever
    generator = st.session_state.generator
    metrics = st.session_state.recall_metrics

    # Layout: main | sidebar
    col_main, col_right = st.columns([3, 1], gap="large")

    with col_right:
        # System status
        st.markdown(
            '<div class="section-label">System</div>',
            unsafe_allow_html=True
        )
        embed_active = (
            retriever.embeddings is not None
            and retriever.embed_model is not None
        )
        reranker_active = retriever.reranker is not None
        n_chunks = len(retriever.chunks)

        st.markdown(f"""
<div class="metric-card">
    <div class="status-line">
        <span class="status-dot"></span>{n_chunks:,} chunks indexed
    </div>
    <div class="status-line">
        <span class="status-dot"
            style="background:{'#2ecc71' if reranker_active else '#333'}">
        </span>Reranker {'active' if reranker_active else 'off'}
    </div>
    <div class="status-line">
        <span class="status-dot"
            style="background:{'#2ecc71' if embed_active else '#333'}">
        </span>Embeddings {'active' if embed_active else 'off'}
    </div>
</div>""", unsafe_allow_html=True)

        # Recall metrics
        st.markdown(
            '<div class="section-label">Retrieval metrics</div>',
            unsafe_allow_html=True
        )
        render_recall_bars(metrics.get("docs", {}), "Docs dataset", col_right)
        render_recall_bars(metrics.get("code", {}), "Code dataset", col_right)

    with col_main:
        # Query input
        k_val = st.selectbox(
            "Results (k)",
            options=[3, 5, 10],
            index=1,
            label_visibility="collapsed",
        )

        query = st.text_input(
            "query",
            placeholder="Ask anything about the vLLM codebase...",
            label_visibility="collapsed",
        )

        run = st.button("Search & Answer", use_container_width=False)

        if run and query and query.strip():
            # Retrieve
            with st.spinner("Retrieving relevant chunks..."):
                sources = retriever.search_for_generation(
                    query, k=k_val
                )

            if not sources:
                st.markdown(
                    '<div class="answer-box">'
                    "I've not found relevant information about "
                    "this subject in my files.</div>",
                    unsafe_allow_html=True
                )
            else:
                # Generate
                with st.spinner("Generating answer..."):
                    answer = generator.answer(query, sources)

                if not answer or answer.strip() in (
                    "No question provided.",
                    "No relevant sources found.",
                    "Could not read source content from disk.",
                ):
                    answer = (
                        "I've not found relevant information "
                        "about this subject in my files."
                    )

                # Answer
                st.markdown(
                    '<div class="section-label">Answer</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True
                )

                # Sources
                not_found = "Not found in the provided sources" in answer
                if not not_found:
                    st.markdown(
                        '<div class="section-label">Sources</div>',
                        unsafe_allow_html=True
                    )
                    for i, source in enumerate(sources, 1):
                        # Read chunk text for preview
                        text_preview = ""
                        abs_path = os.path.join(
                            REPO_ROOT, source.file_path
                        )
                        if not os.path.isfile(abs_path):
                            abs_path = source.file_path
                        if os.path.isfile(abs_path):
                            try:
                                with open(
                                    abs_path, "r",
                                    encoding="utf-8", errors="ignore"
                                ) as fh:
                                    content = fh.read()
                                text_preview = content[
                                    source.first_character_index:
                                    source.last_character_index
                                ]
                            except (OSError, IndexError):
                                pass

                        render_source_card(
                            index=i,
                            file_path=source.file_path,
                            first_char=source.first_character_index,
                            last_char=source.last_character_index,
                            text_preview=text_preview,
                        )

        elif run and (not query or not query.strip()):
            st.warning("Write a question first.")


if __name__ == "__main__":
    main()
