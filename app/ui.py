"""Streamlit UI component and presentation logic for LegalLens."""

import os
from pathlib import Path
import streamlit as st

from app.config import config
from app.document_processor import DocumentProcessor, DocumentProcessingError
from app.rag import LegalRAGIndex
from app.legal_analyzer import LegalAnalyzer
from app.comparator import DocumentComparator
from app.prompts import QA_PROMPT
from app.llm_provider import LLMProvider
from app.safety import SafetyGuardrail, LEGAL_DISCLAIMER_TEXT

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CUSTOM_CSS = """
<style>
/* Main typography and container */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header & branding */
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #f8fafc;
    padding: 1.75rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 20px -2px rgba(0,0,0,0.15);
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 0;
}

/* Badges & Pills */
.pill-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-right: 0.5rem;
    margin-bottom: 0.25rem;
}
.badge-payment { background-color: #dbeafe; color: #1e40af; }
.badge-term { background-color: #fef3c7; color: #92400e; }
.badge-termination { background-color: #fee2e2; color: #991b1b; }
.badge-liability { background-color: #fce7f3; color: #9d174d; }
.badge-notice { background-color: #e0e7ff; color: #3730a3; }
.badge-default { background-color: #f1f5f9; color: #334155; }

/* Custom Cards */
.legal-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.legal-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.attention-card {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}

.source-tag {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 500;
    background: #f8fafc;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
    display: inline-block;
    margin-top: 0.5rem;
}

/* Stat Box */
.stat-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.stat-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.stat-val {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
}
</style>
"""


def get_badge_class(category: str) -> str:
    cat = (category or "").lower()
    if any(k in cat for k in ["payment", "fee", "deposit", "money"]):
        return "badge-payment"
    if any(k in cat for k in ["term", "renewal", "duration"]):
        return "badge-term"
    if any(k in cat for k in ["termination", "cancellation", "penalty"]):
        return "badge-termination"
    if any(k in cat for k in ["liability", "indemnification"]):
        return "badge-liability"
    if any(k in cat for k in ["notice", "deadline"]):
        return "badge-notice"
    return "badge-default"


def init_session_state():
    """Initializes Streamlit session variables."""
    if "doc_text" not in st.session_state:
        st.session_state.doc_text = ""
    if "doc_pages" not in st.session_state:
        st.session_state.doc_pages = []
    if "doc_metadata" not in st.session_state:
        st.session_state.doc_metadata = {}
    if "doc_filename" not in st.session_state:
        st.session_state.doc_filename = ""
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "rag_index" not in st.session_state:
        st.session_state.rag_index = None
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None
    if "checklist_state" not in st.session_state:
        st.session_state.checklist_state = {}


def render_sidebar():
    """Renders user controls, API keys, provider selector, and sample contract buttons."""
    with st.sidebar:
        st.subheader("⚙️ AI Engine Settings")

        current_provider = config.get_active_provider()
        provider_options = ["gemini", "groq", "openai", "mock"]
        provider_index = provider_options.index(current_provider) if current_provider in provider_options else 0

        chosen_provider = st.selectbox(
            "Selected Provider",
            options=provider_options,
            index=provider_index,
            format_func=lambda x: {
                "gemini": "Google Gemini (Gemini 2.0 / 1.5)",
                "groq": "Groq (Llama 3.3 70B)",
                "openai": "OpenAI (GPT-4o Mini)",
                "mock": "Offline / Demo Engine (No API key required)"
            }.get(x, x),
            help="Select your AI model provider or use the smart Offline Demo engine."
        )

        api_key_input = ""
        if chosen_provider != "mock":
            env_key = getattr(config, f"{chosen_provider.upper()}_API_KEY", "") or ""
            api_key_input = st.text_input(
                f"{chosen_provider.title()} API Key",
                value=env_key,
                type="password",
                placeholder=f"Enter {chosen_provider.title()} API Key...",
                help="Key is kept securely in memory for this session and never saved to disk."
            )
            if not api_key_input:
                st.caption(f"💡 No key set. You can run in **Offline / Demo Engine** mode, or add your key above.")

        st.markdown("---")
        st.subheader("📂 Instant Sample Contracts")
        st.caption("Load verified realistic legal documents with one click:")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Lease v1", use_container_width=True):
                load_sample_file("residential_lease_v1.txt")
        with col2:
            if st.button("📝 Lease v2", use_container_width=True):
                load_sample_file("residential_lease_v2_revised.txt")

        if st.button("💼 Freelance Agreement", use_container_width=True):
            load_sample_file("freelance_consulting_agreement.txt")

        st.markdown("---")
        if st.button("🔄 Clear Current Session", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.markdown("---")
        st.caption("🔒 **Privacy Guarantee**")
        st.caption("Your uploaded documents are processed in-memory for this session only and are never saved permanently.")

    return chosen_provider, api_key_input


def load_sample_file(filename: str):
    """Loads a bundled sample contract into session state."""
    path = DATA_DIR / filename
    if path.exists():
        text = path.read_text(encoding="utf-8")
        full_text, pages, meta = DocumentProcessor.extract_text(text.encode("utf-8"), filename)
        chunks = DocumentProcessor.chunk_document(pages)
        st.session_state.doc_text = full_text
        st.session_state.doc_pages = pages
        st.session_state.doc_metadata = meta
        st.session_state.doc_filename = filename
        st.session_state.rag_index = LegalRAGIndex(chunks)
        st.session_state.analysis = None
        st.session_state.qa_history = []
        st.toast(f"Loaded sample: {filename}", icon="📄")
        st.rerun()


def run_document_analysis(provider: str, api_key: str):
    """Executes single-document analysis."""
    if not st.session_state.doc_text:
        return

    with st.spinner("Analyzing legal document with LegalLens..."):
        try:
            analyzer = LegalAnalyzer(provider=provider, api_key=api_key)
            analysis = analyzer.analyze(
                st.session_state.doc_text,
                st.session_state.doc_metadata
            )
            st.session_state.analysis = analysis
            st.success("Analysis complete!")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            with st.expander("🛠️ Developer Debug & Error Details"):
                import traceback
                st.code(traceback.format_exc(), language="python")


def render_understand_tab(provider: str, api_key: str):
    """Renders Single-Document Understanding Tab."""
    st.markdown("### 📄 Upload & Understand Legal Document")
    st.markdown("Upload any rental lease, employment contract, service agreement, or NDA to receive a plain-language breakdown.")

    uploaded_file = st.file_uploader(
        "Upload document (PDF, Word DOCX, or Plain Text TXT)",
        type=["pdf", "docx", "txt"],
        key="main_doc_uploader"
    )

    if uploaded_file is not None:
        if st.session_state.doc_filename != uploaded_file.name:
            try:
                bytes_data = uploaded_file.read()
                full_text, pages, meta = DocumentProcessor.extract_text(bytes_data, uploaded_file.name)
                chunks = DocumentProcessor.chunk_document(pages)

                st.session_state.doc_text = full_text
                st.session_state.doc_pages = pages
                st.session_state.doc_metadata = meta
                st.session_state.doc_filename = uploaded_file.name
                st.session_state.rag_index = LegalRAGIndex(chunks)
                st.session_state.analysis = None
                st.session_state.qa_history = []
                st.rerun()
            except DocumentProcessingError as e:
                st.error(f"⚠️ Document error: {str(e)}")
                return
            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred while reading the file: {str(e)}")
                return

    if not st.session_state.doc_text:
        st.info("👆 Upload a document above or pick one of the sample agreements in the sidebar to get started.")
        return

    st.markdown(f"**Loaded Document:** `{st.session_state.doc_filename}` ({st.session_state.doc_metadata.get('page_count', 1)} pages, ~{st.session_state.doc_metadata.get('word_count', 0):,} words)")

    if st.session_state.analysis is None:
        if st.button("🚀 Analyze This Document", type="primary", use_container_width=True):
            run_document_analysis(provider, api_key)
            st.rerun()
        return

    analysis = st.session_state.analysis

    # 1. Document Overview Grid
    st.markdown("#### 1. Document Overview")
    meta = analysis.metadata

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Document Type</div>
            <div class="stat-val">{meta.document_type}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Approximate Length</div>
            <div class="stat-val">{meta.approximate_length}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Effective Date</div>
            <div class="stat-val">{meta.effective_date}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Parties Involved</div>
            <div class="stat-val" style="font-size:0.95rem;">{meta.parties_involved}</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Expiration / Renewal</div>
            <div class="stat-val" style="font-size:0.95rem;">{meta.expiration_date}</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Governing Jurisdiction</div>
            <div class="stat-val" style="font-size:0.95rem;">{meta.governing_jurisdiction}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Plain-English Summary
    st.markdown("#### 2. Plain-English Summary")
    sum_data = analysis.summary

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class="legal-card">
            <h5 style="color:#0f172a; margin-bottom:0.4rem;">📘 What is this document?</h5>
            <p style="color:#334155; font-size:0.95rem; margin-bottom:0;">{sum_data.what_is_this}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="legal-card">
            <h5 style="color:#0f172a; margin-bottom:0.4rem;">🤝 What does the other party agree to?</h5>
            <p style="color:#334155; font-size:0.95rem; margin-bottom:0;">{sum_data.what_does_other_party_agree_to}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown(f"""
        <div class="legal-card">
            <h5 style="color:#0f172a; margin-bottom:0.4rem;">✍️ What am I agreeing to?</h5>
            <p style="color:#334155; font-size:0.95rem; margin-bottom:0;">{sum_data.what_am_i_agreeing_to}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="legal-card" style="background:#fffbeb; border-color:#fef3c7;">
            <h5 style="color:#92400e; margin-bottom:0.4rem;">🔍 What should I pay attention to?</h5>
            <p style="color:#78350f; font-size:0.95rem; margin-bottom:0;">{sum_data.what_should_i_pay_attention_to}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Attention Points
    if analysis.attention_points:
        st.markdown("#### 3. Provisions Worth Reviewing")
        st.caption("Neutral observations on terms that warrant careful consideration:")
        for ap in analysis.attention_points:
            st.markdown(f"""
            <div class="attention-card">
                <div style="font-weight:700; color:#92400e; font-size:1.02rem; margin-bottom:0.25rem;">
                    ⚠️ {ap.item}
                </div>
                <p style="margin-bottom:0.35rem; color:#451a03; font-size:0.93rem;">
                    <strong>Observation:</strong> {ap.observation}
                </p>
                <p style="margin-bottom:0.25rem; color:#451a03; font-size:0.93rem;">
                    <strong>Why it matters:</strong> {ap.why_it_matters}
                </p>
                <div class="source-tag">Source: {ap.source}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

    # 4. Obligations & Deadlines
    st.markdown("#### 4. Actionable Obligations & Deadlines")
    your_obs = [o for o in analysis.obligations if "your" in o.party.lower()]
    other_obs = [o for o in analysis.obligations if "your" not in o.party.lower()]

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.markdown("##### 📌 Your Obligations")
        if your_obs:
            for o in your_obs:
                st.markdown(f"""
                <div class="legal-card" style="border-left: 4px solid #3b82f6;">
                    <div style="font-weight:600; color:#1e293b; margin-bottom:0.2rem;">{o.obligation}</div>
                    <div style="font-size:0.85rem; color:#475569;"><strong>Timeline / Deadline:</strong> {o.deadline_or_timeline}</div>
                    <div class="source-tag">Source: {o.source}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No explicit obligations for your party identified.")

    with col_o2:
        st.markdown("##### 🏢 Other Party's Obligations")
        if other_obs:
            for o in other_obs:
                st.markdown(f"""
                <div class="legal-card" style="border-left: 4px solid #10b981;">
                    <div style="font-weight:600; color:#1e293b; margin-bottom:0.2rem;">{o.obligation}</div>
                    <div style="font-size:0.85rem; color:#475569;"><strong>Timeline / Deadline:</strong> {o.deadline_or_timeline}</div>
                    <div class="source-tag">Source: {o.source}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No explicit counterparty obligations identified.")

    st.markdown("---")

    # 5. Important Clauses
    st.markdown("#### 5. Key Clauses & Practical Meanings")
    st.caption("Categorized clause extracts with plain-language explanations:")

    for idx, c in enumerate(analysis.clauses):
        badge_cls = get_badge_class(c.category)
        with st.expander(f"🔹 {c.clause_title} [{c.category}]", expanded=(idx < 2)):
            st.markdown(f'<span class="pill-badge {badge_cls}">{c.category}</span>', unsafe_allow_html=True)
            st.markdown(f"**What the document says:**\n> {c.what_the_document_says}")
            st.markdown(f"**In simple words:**\n{c.in_simple_words}")
            st.markdown(f"**Why you may want to pay attention:**\n{c.why_you_may_want_to_pay_attention}")
            st.markdown(f'<div class="source-tag">Source: {c.source}</div>', unsafe_allow_html=True)


def render_compare_tab(provider: str, api_key: str):
    """Renders Two-Document Comparison Tab."""
    st.markdown("### ⚖️ Compare Two Document Versions")
    st.markdown("Upload two versions of a contract (e.g. Original vs Revised) to identify meaningful changes, altered obligations, and practical implications.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Document A (Original)")
        file_a = st.file_uploader("Upload Original Document", type=["pdf", "docx", "txt"], key="compare_doc_a")
    with col2:
        st.markdown("##### Document B (Revised)")
        file_b = st.file_uploader("Upload Revised Document", type=["pdf", "docx", "txt"], key="compare_doc_b")

    # Quick button to load comparison samples
    if st.button("📂 Load Sample Lease Comparison (v1 vs v2)", key="load_compare_samples"):
        path_a = DATA_DIR / "residential_lease_v1.txt"
        path_b = DATA_DIR / "residential_lease_v2_revised.txt"
        if path_a.exists() and path_b.exists():
            st.session_state.compare_text_a = path_a.read_text(encoding="utf-8")
            st.session_state.compare_name_a = "Lease Agreement v1"
            st.session_state.compare_text_b = path_b.read_text(encoding="utf-8")
            st.session_state.compare_name_b = "Revised Lease Agreement v2"
            st.session_state.comparison_result = None
            st.rerun()

    text_a = getattr(st.session_state, "compare_text_a", "")
    text_b = getattr(st.session_state, "compare_text_b", "")
    name_a = getattr(st.session_state, "compare_name_a", "Document A")
    name_b = getattr(st.session_state, "compare_name_b", "Document B")

    if file_a:
        try:
            t_a, _, _ = DocumentProcessor.extract_text(file_a.read(), file_a.name)
            text_a = t_a
            name_a = file_a.name
            st.session_state.compare_text_a = text_a
            st.session_state.compare_name_a = name_a
        except Exception as e:
            st.error(f"Error reading Document A: {e}")

    if file_b:
        try:
            t_b, _, _ = DocumentProcessor.extract_text(file_b.read(), file_b.name)
            text_b = t_b
            name_b = file_b.name
            st.session_state.compare_text_b = text_b
            st.session_state.compare_name_b = name_b
        except Exception as e:
            st.error(f"Error reading Document B: {e}")

    if text_a and text_b:
        st.info(f"Ready to compare **{name_a}** with **{name_b}**")
        if st.button("🔍 Compare Versions Now", type="primary"):
            with st.spinner("Analyzing contractual changes between versions..."):
                try:
                    comparator = DocumentComparator(provider=provider, api_key=api_key)
                    res = comparator.compare(text_a, text_b, doc_a_name=name_a, doc_b_name=name_b)
                    st.session_state.comparison_result = res
                    st.success("Comparison complete!")
                except Exception as e:
                    st.error(f"Comparison error: {e}")

    res = st.session_state.comparison_result
    if res:
        st.markdown("---")
        st.markdown("#### 📊 Comparison Summary")
        st.markdown(f"""
        <div class="legal-card" style="border-left: 4px solid #6366f1;">
            <p style="font-size:1.05rem; font-weight:600; color:#1e1b4b; margin-bottom:0.5rem;">Executive Summary:</p>
            <p style="color:#312e81; margin-bottom:0.5rem;">{res.summary_of_differences}</p>
            <div style="background:#e0e7ff; padding:0.5rem 0.75rem; border-radius:6px; font-size:0.9rem; color:#3730a3;">
                <strong>Key Takeaway:</strong> {res.overall_takeaway}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📝 Meaningful Changes Identified")
        if res.changes:
            for idx, change in enumerate(res.changes):
                badge_cls = get_badge_class(change.category)
                with st.container():
                    st.markdown(f"""
                    <div class="legal-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                            <span class="pill-badge {badge_cls}">{change.category}</span>
                            <span style="font-size:0.8rem; color:#64748b;">Change #{idx+1}</span>
                        </div>
                        <h5 style="color:#0f172a; margin-bottom:0.4rem;">{change.what_changed}</h5>
                        
                        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:1rem; margin-bottom:0.75rem;">
                            <div style="background:#f1f5f9; padding:0.75rem; border-radius:6px; font-size:0.88rem;">
                                <strong style="color:#475569;">Previous Version ({change.source_previous}):</strong><br>
                                {change.previous_version}
                            </div>
                            <div style="background:#eff6ff; padding:0.75rem; border-radius:6px; font-size:0.88rem; border-left:3px solid #3b82f6;">
                                <strong style="color:#1d4ed8;">New Version ({change.source_new}):</strong><br>
                                {change.new_version}
                            </div>
                        </div>

                        <div style="background:#fffbeb; padding:0.6rem 0.85rem; border-radius:6px; font-size:0.88rem; color:#92400e;">
                            <strong>What this could mean:</strong> {change.what_this_could_mean}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No material differences were found between the documents.")


def render_qa_tab(provider: str, api_key: str):
    """Renders Grounded Q&A Interface."""
    st.markdown("### 💬 Ask Questions About Your Document")
    st.markdown("Questions are answered using **strict document grounding** with exact page and section citations. If a fact is absent, LegalLens will explicitly inform you rather than guessing.")

    if not st.session_state.doc_text:
        st.info("Please upload a document or load a sample contract in the sidebar first to use Document Q&A.")
        return

    # Quick prompt buttons
    st.markdown("**Sample Grounded Questions:**")
    q_cols = st.columns(3)
    sample_queries = [
        "Can I terminate this agreement early?",
        "What are the payment and deposit terms?",
        "Who is responsible for repairs and maintenance?"
    ]
    selected_sample = None
    for i, q in enumerate(sample_queries):
        if q_cols[i].button(q, key=f"quick_q_{i}", use_container_width=True):
            selected_sample = q

    # Chat history display
    for msg in st.session_state.qa_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.caption(f"📍 Sources: {', '.join(msg['citations'])}")

    user_query = st.chat_input("Ask any question regarding this document...") or selected_sample

    if user_query:
        # Display user message
        st.session_state.qa_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve grounded context
        with st.spinner("Searching document context..."):
            rag = st.session_state.rag_index
            if not rag:
                chunks = DocumentProcessor.chunk_document(st.session_state.doc_pages)
                rag = LegalRAGIndex(chunks)
                st.session_state.rag_index = rag

            context, citations = rag.get_grounded_context(user_query, top_k=3)

            # Generate grounded response
            llm = LLMProvider(provider=provider, api_key=api_key)
            prompt = QA_PROMPT.format(context=context, question=user_query)
            try:
                response_text = llm.generate(prompt, temperature=0.1)
                sanitized_text = SafetyGuardrail.sanitize_text(response_text)
            except Exception as e:
                sanitized_text = f"An error occurred while querying the model: {str(e)}"

        with st.chat_message("assistant"):
            st.markdown(sanitized_text)
            if citations:
                st.caption(f"📍 Sources: {', '.join(citations)}")

        st.session_state.qa_history.append({
            "role": "assistant",
            "content": sanitized_text,
            "citations": citations
        })


def render_checklist_tab():
    """Renders Interactive Action Checklist."""
    st.markdown("### ✅ Actionable Pre-Signing & Execution Checklist")
    st.markdown("Track and verify critical conditions before signing or committing to the agreement.")

    if st.session_state.analysis is None:
        st.info("Please analyze a document in the 'Understand' tab first to generate your customized checklist.")
        return

    checklist = st.session_state.analysis.checklist
    if not checklist:
        st.info("No specific checklist items were derived for this document.")
        return

    # Compute progress
    total_items = len(checklist)
    completed_count = sum(1 for idx, _ in enumerate(checklist) if st.session_state.checklist_state.get(f"task_{idx}", False))
    progress = completed_count / max(1, total_items)

    st.progress(progress, text=f"Completion Progress: {completed_count} of {total_items} items reviewed ({int(progress*100)}%)")

    # Group by category
    categories = sorted(list(set(item.category for item in checklist)))
    for cat in categories:
        st.markdown(f"#### 📋 {cat}")
        cat_items = [(idx, item) for idx, item in enumerate(checklist) if item.category == cat]
        for idx, item in cat_items:
            key = f"task_{idx}"
            checked = st.checkbox(
                f"**{item.task}**",
                value=st.session_state.checklist_state.get(key, item.completed),
                key=key,
                help=item.detail
            )
            st.session_state.checklist_state[key] = checked
            st.caption(f"💡 *Why this matters:* {item.detail}")
            st.write("")


def render_lawyer_questions_tab():
    """Renders 'What Should I Ask a Lawyer?' Feature."""
    st.markdown("### 👨‍⚖️ What Should I Ask a Legal Professional?")
    st.markdown("Document-grounded, practical questions you can bring to a consultation with a qualified attorney or the counterparty before signing.")

    if st.session_state.analysis is None:
        st.info("Please analyze a document in the 'Understand' tab first to generate tailored consultation questions.")
        return

    questions = st.session_state.analysis.lawyer_questions
    if not questions:
        st.info("No specific consultation questions were generated for this document.")
        return

    for idx, q in enumerate(questions):
        st.markdown(f"""
        <div class="legal-card" style="border-left: 4px solid #8b5cf6;">
            <div style="font-size:0.8rem; font-weight:600; color:#6d28d9; text-transform:uppercase; margin-bottom:0.25rem;">
                {q.category}
            </div>
            <h5 style="color:#1e1b4b; margin-bottom:0.35rem;">❓ {q.question}</h5>
            <p style="color:#4c1d95; font-size:0.9rem; margin-bottom:0;">
                <strong>Context from document:</strong> {q.context}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Exportable list of questions
    all_q_text = "\n\n".join(
        f"{idx+1}. [{q.category}] {q.question}\n   Context: {q.context}"
        for idx, q in enumerate(questions)
    )
    st.download_button(
        label="📥 Download Questions List (TXT)",
        data=f"LegalLens — Consultation Questions for {st.session_state.doc_filename}\n\n{all_q_text}\n\n{LEGAL_DISCLAIMER_TEXT}",
        file_name="legallens_lawyer_questions.txt",
        mime="text/plain"
    )


def render_app():
    """Root UI presentation function."""
    st.set_page_config(
        page_title="LegalLens — Understand it before you sign it",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ LegalLens</h1>
        <p><strong>Understand it before you sign it.</strong> AI-powered legal document understanding, comparison, and practical guidance for everyone.</p>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer Banner
    st.markdown(SafetyGuardrail.get_disclaimer_banner())

    # Sidebar controls
    provider, api_key = render_sidebar()

    # Main Navigation Tabs
    tab_understand, tab_compare, tab_qa, tab_checklist, tab_lawyer = st.tabs([
        "📄 Understand Document",
        "⚖️ Compare Versions",
        "💬 Ask Questions (Q&A)",
        "✅ Action Checklist",
        "👨‍⚖️ Questions for Lawyer"
    ])

    with tab_understand:
        render_understand_tab(provider, api_key)

    with tab_compare:
        render_compare_tab(provider, api_key)

    with tab_qa:
        render_qa_tab(provider, api_key)

    with tab_checklist:
        render_checklist_tab()

    with tab_lawyer:
        render_lawyer_questions_tab()

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:#64748b; font-size:0.82rem; padding:1rem 0;">
        <strong>LegalLens v1.0.0</strong> — Built for clarity, transparency, and consumer protection.<br>
        <em>{LEGAL_DISCLAIMER_TEXT}</em>
    </div>
    """, unsafe_allow_html=True)
