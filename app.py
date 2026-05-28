import os
import json
import pickle
from pathlib import Path
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
INDEX_PATH = "faiss_index/fiqh.index"
CHUNKS_PATH = "faiss_index/chunks.pkl"
CATEGORIES_PATH = "fiqh_data/fiqh_categories.json"
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
HF_REPO_ID = "ubaid-ai/fiqh-qa-bot-data"

SYSTEM_PROMPT = """You are an Islamic Fiqh Q&A Assistant created by Ubaid ur Rehman, \
an Aalim specializing in Islamic Fiqh (primarily Hanafi school, as followed in Pakistan/India).

Your knowledge base includes:
- Fatawa Darul Uloom Deoband — Mufti Mahmood ul Hasan Gangohi (رحمہ اللہ)
- Classical Hanafi fiqh texts (Hidayah, Durr al-Mukhtar, Radd al-Muhtar, Bahishti Zewar)
- Quran and Hadith references

Your role:
- Answer fiqh questions using the provided knowledge base excerpts.
- Always cite relevant Quran verses and Hadith references from the context.
- When citing Fatawa Darul Uloom, mention the source explicitly.
- Clearly mention the madhab/school of thought (primarily Hanafi).
- If the question requires a personal fatwa or a complex ruling on a specific situation, \
advise the user to consult a qualified Mufti.
- Answer in the same language the user uses (English, Urdu, or Roman Urdu).
- Be respectful, scholarly, and accurate. Do not fabricate hadith references.
- If the provided context does not contain enough information, say so clearly \
rather than inventing an answer.
- Format answers clearly with references at the end when appropriate.
- When answering, include Arabic daleel (Quran/Hadith text) where available.

IMPORTANT DISCLAIMER: This is an educational tool, not a fatwa service. \
Always recommend consulting a qualified scholar for personal rulings."""

EXAMPLE_QUESTIONS = [
    "Wudu karne ka tareeqa kya hai?",
    "What breaks the fast during Ramadan?",
    "Zakat ki nisab kitni hai?",
    "Can I pray sitting if I'm sick?",
    "Nikah ke arkaan kya hain?",
    "Is seafood halal in Hanafi fiqh?",
    "Taraweeh 20 rakat ki daleel?",
    "Sood (interest) ki hurmat kya hai?",
]

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Islamic Fiqh Q&A Assistant",
    page_icon="☪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Imports ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Reset & Base ── */
    :root {
        --teal:          #0D9488;
        --teal-dark:     #0F766E;
        --teal-light:    #CCFBF1;
        --teal-lighter:  #F0FDFA;
        --teal-border:   #99F6E4;
        --text:          #374151;
        --text-light:    #6B7280;
        --text-muted:    #9CA3AF;
        --bg:            #FFFFFF;
        --bg-subtle:     #F9FAFB;
        --border:        #E5E7EB;
        --shadow-sm:     0 1px 3px rgba(0,0,0,0.08);
        --shadow-md:     0 4px 16px rgba(0,0,0,0.08);
    }

    /* ── App shell ── */
    .stApp {
        background: #FFFFFF !important;
        color: var(--text);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 860px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    .sidebar-brand {
        text-align: center;
        padding: 0.5rem 1rem 1.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    .sidebar-icon {
        font-size: 2.8rem;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .sidebar-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--teal-dark);
        margin: 0.3rem 0 0.2rem;
    }
    .sidebar-desc {
        font-size: 0.78rem;
        color: var(--text-light);
        line-height: 1.4;
    }
    .sidebar-meta {
        margin-top: 1.5rem;
        padding: 0.75rem 1rem;
        background: var(--teal-lighter);
        border-radius: 10px;
        border: 1px solid var(--teal-border);
        font-size: 0.75rem;
        color: var(--teal-dark);
        line-height: 1.7;
    }
    .sidebar-meta strong { font-weight: 600; }

    /* ── Header ── */
    .main-header {
        text-align: center;
        padding: 2rem 1.5rem 1.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid var(--teal-lighter);
        position: relative;
        animation: none !important;
        opacity: 1 !important;
    }
    .header-arabic {
        font-size: 1.35rem;
        color: var(--teal);
        direction: rtl;
        font-weight: 600;
        margin-bottom: 0.4rem;
        letter-spacing: 0.02em;
    }
    .header-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text);
        margin: 0.2rem 0;
    }
    .header-title span { color: var(--teal); }
    .header-sub {
        font-size: 0.85rem;
        color: var(--text-light);
        margin-top: 0.4rem;
    }
    .header-line {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }
    .header-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--teal);
        opacity: 0.4;
    }
    .header-dot.center { opacity: 1; width: 8px; height: 8px; }

    /* ── Welcome Screen ── */
    .welcome-wrap {
        text-align: center;
        padding: 2.5rem 1rem 2rem;
        animation: none !important;
        opacity: 1 !important;
    }
    .welcome-icon { font-size: 3rem; margin-bottom: 0.75rem; }
    .welcome-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.3rem;
    }
    .welcome-sub {
        font-size: 0.9rem;
        color: var(--text-light);
        margin-bottom: 0.2rem;
        direction: rtl;
    }
    .welcome-hint {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 1.2rem;
    }
    .example-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 1.5rem;
        padding: 0 0.5rem;
    }

    /* ── Chat Messages ── */
    .msg-wrap-user {
        display: flex;
        justify-content: flex-end;
        margin: 0.7rem 0;
        animation: none !important;
        opacity: 1 !important;
    }
    .msg-wrap-assistant {
        display: flex;
        justify-content: flex-start;
        margin: 0.7rem 0;
        animation: none !important;
        opacity: 1 !important;
    }
    .user-bubble {
        background: var(--teal-light);
        border-radius: 18px 18px 4px 18px;
        padding: 0.85rem 1.1rem;
        max-width: 75%;
        color: #134E4A;
        font-size: 0.95rem;
        line-height: 1.55;
        box-shadow: var(--shadow-sm);
    }
    .user-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: var(--teal-dark);
        margin-bottom: 0.35rem;
        opacity: 0.75;
    }
    .assistant-card {
        background: #FFFFFF;
        border-left: 4px solid var(--teal);
        border-radius: 0 16px 16px 0;
        padding: 1rem 1.2rem;
        max-width: 88%;
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: var(--shadow-md);
        border-top: 1px solid var(--border);
        border-bottom: 1px solid var(--border);
        border-right: 1px solid var(--border);
    }
    .assistant-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: var(--teal);
        margin-bottom: 0.4rem;
        opacity: 0.85;
    }

    /* ── Sources Box ── */
    .sources-box {
        background: var(--teal-lighter);
        border: 1px solid var(--teal-border);
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-top: 0.85rem;
        font-size: 0.78rem;
        color: #0F766E;
        line-height: 1.6;
    }
    .sources-box strong { font-weight: 600; }

    /* ── Input area ── */
    .stTextInput > div > div > input {
        background: var(--bg-subtle) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1rem !important;
        transition: none !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--teal) !important;
        background: #FFFFFF !important;
        box-shadow: 0 0 0 3px rgba(13,148,136,0.12) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--teal) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 2px 8px rgba(13,148,136,0.25) !important;
        transition: none !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button:hover {
        background: var(--teal-dark) !important;
        box-shadow: 0 4px 14px rgba(13,148,136,0.3) !important;
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(13,148,136,0.2) !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: var(--bg-subtle) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-size: 0.88rem !important;
    }
    .stSelectbox label {
        color: var(--text-light) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: var(--teal) !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1.2rem 1rem 0.5rem;
        color: var(--text-muted);
        font-size: 0.76rem;
        border-top: 1px solid var(--border);
        margin-top: 2.5rem;
        line-height: 1.6;
    }
    .footer strong { color: var(--text-light); font-weight: 600; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--teal-border); border-radius: 4px; }

    /* ── Anti-flicker ── */
    .main-header, .user-bubble, .assistant-card,
    .sources-box, .footer, .welcome-wrap,
    [data-testid="stVerticalBlock"], .element-container {
        animation: none !important;
        transition: none !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Cloud Download (HuggingFace) ─────────────────────────────────────────────
def ensure_data_files():
    """Download FAISS data from HuggingFace if not available locally (cloud deployment)."""
    index_dir = Path("faiss_index")
    data_dir = Path("fiqh_data")

    required = [
        (index_dir / "fiqh.index", "fiqh.index"),
        (index_dir / "chunks.pkl", "chunks.pkl"),
    ]
    data_files = [
        (data_dir / "fiqh_qa.json", "fiqh_qa.json"),
        (data_dir / "fiqh_categories.json", "fiqh_categories.json"),
    ]

    if all(f.exists() for f, _ in required + data_files):
        return True

    try:
        from huggingface_hub import hf_hub_download

        index_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)

        for local_path, hf_filename in required + data_files:
            if not local_path.exists():
                st.info(f"⏳ Downloading {hf_filename} from HuggingFace...")
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=hf_filename,
                    repo_type="dataset",
                    local_dir=str(local_path.parent),
                    local_dir_use_symlinks=False,
                )
        return True
    except ImportError:
        return False
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False


# ── Resource Loading ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        with st.spinner("⏳ First launch: downloading Fiqh knowledge base from HuggingFace..."):
            if not ensure_data_files():
                st.error(
                    "FAISS index not found. Please run `python prepare_data.py` first.",
                    icon="⚠️",
                )
                st.stop()
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return model, index, chunks


@st.cache_data
def load_categories():
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["categories"]


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file.", icon="🔑")
        st.stop()
    return Groq(api_key=api_key)


# ── RAG Search ───────────────────────────────────────────────────────────────
def search_fiqh(query, model, index, chunks, category_filter="All", k=TOP_K):
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k * 3)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        if category_filter != "All" and chunk["category"] != category_filter:
            continue
        results.append((score, chunk))
        if len(results) >= k:
            break
    return results


def build_context(results):
    parts = []
    for i, (_, chunk) in enumerate(results, 1):
        parts.append(
            f"[Source {i}] Category: {chunk['category']}\n"
            f"Q: {chunk['question']}\n"
            f"A: {chunk['answer']}\n"
            f"Quran Refs: {', '.join(chunk['quran_refs']) if chunk['quran_refs'] else 'N/A'}\n"
            f"Hadith Refs: {', '.join(chunk['hadith_refs']) if chunk['hadith_refs'] else 'N/A'}\n"
            f"Madhab: {chunk['madhab']} | Source: {chunk['source']}"
        )
    return "\n\n---\n\n".join(parts)


# ── LLM Response ─────────────────────────────────────────────────────────────
def get_answer(client, query, context, chat_history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in chat_history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({
        "role": "user",
        "content": (
            f"Based on the following fiqh knowledge base:\n\n{context}\n\n"
            f"Please answer this question: {query}"
        ),
    })
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except RateLimitError:
        return (
            "⚠️ **Rate limit reached.** The free Groq API allows a limited number of "
            "requests per minute/day. Please wait a minute and try again, or check your "
            "Groq console to upgrade your plan."
        )


# ── Rendering Helpers ─────────────────────────────────────────────────────────
def render_message(role, content, sources=None):
    if role == "user":
        st.markdown(
            f"<div class='msg-wrap-user'>"
            f"<div class='user-bubble'>"
            f"<div class='user-label'>You</div>"
            f"{content}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        refs_html = ""
        if sources:
            for _, chunk in sources[:3]:
                quran = ", ".join(chunk["quran_refs"]) if chunk["quran_refs"] else "—"
                hadith = (", ".join(chunk["hadith_refs"][:2])
                          if chunk["hadith_refs"] else "—")
                refs_html += (
                    f"<b>{chunk['category']}</b> · "
                    f"Quran: {quran} · Hadith: {hadith} · "
                    f"Source: {chunk['source']}<br>"
                )
        sources_section = (
            f"<div class='sources-box'><strong>📖 References Used</strong><br>{refs_html}</div>"
            if refs_html else ""
        )
        st.markdown(
            f"<div class='msg-wrap-assistant'>"
            f"<div class='assistant-card'>"
            f"<div class='assistant-label'>☪ Fiqh Assistant</div>"
            f"{content}"
            f"{sources_section}"
            f"</div></div>",
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(categories):
    with st.sidebar:
        st.markdown("""
        <div class='sidebar-brand'>
            <div class='sidebar-icon'>☪️</div>
            <div class='sidebar-title'>Islamic Fiqh Assistant</div>
            <div class='sidebar-desc'>
                AI-powered Q&amp;A grounded in classical Hanafi scholarship and
                Fatawa Darul Uloom Deoband
            </div>
            <div class='sidebar-meta'>
                <strong>School:</strong> Hanafi · حنفی مذہب<br>
                <strong>LLM:</strong> Llama 3.3 70B (Groq)<br>
                <strong>Embeddings:</strong> all-MiniLM-L6-v2<br>
                <strong>Sources:</strong> Darul Uloom Deoband + Classical texts
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Filter by Category**")
        cat_names = ["All"] + [c["name"] for c in categories]
        selected = st.selectbox("Category", cat_names, label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources_map = {}
            st.rerun()

    return selected


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model, index, chunks = load_resources()
    categories = load_categories()
    client = get_groq_client()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "sources_map" not in st.session_state:
        st.session_state.sources_map = {}

    selected_category = render_sidebar(categories)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='main-header'>
        <div class='header-arabic'>مساعد الفقه الإسلامي</div>
        <div class='header-title'>☪ Islamic <span>Fiqh</span> Q&amp;A Assistant</div>
        <div class='header-sub'>
            Hanafi school · Fatawa Darul Uloom Deoband · Quran &amp; Hadith references
        </div>
        <div class='header-line'>
            <div class='header-dot'></div>
            <div class='header-dot'></div>
            <div class='header-dot center'></div>
            <div class='header-dot'></div>
            <div class='header-dot'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chat history or welcome screen ────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div class='welcome-wrap'>
            <div class='welcome-icon'>📖</div>
            <div class='welcome-title'>Assalamu Alaykum — How can I help?</div>
            <div class='welcome-sub'>السلام علیکم! اسلامی فقہ کے بارے میں کوئی بھی سوال پوچھیں</div>
            <div class='welcome-hint'>Try one of the examples below, or type your own question</div>
            <div class='example-grid'>
        """, unsafe_allow_html=True)

        # Render example question buttons in a tight grid
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"eq_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

        st.markdown("</div></div>", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            sources = st.session_state.sources_map.get(msg.get("id"))
            render_message(
                msg["role"], msg["content"],
                sources if msg["role"] == "assistant" else None,
            )

    # ── Input area ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([7, 1])
    with col1:
        default_val = st.session_state.pending_question or ""
        user_input = st.text_input(
            "Ask a fiqh question",
            value=default_val,
            placeholder="e.g., What breaks wudu? · Zakat ki nisab kya hai?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col2:
        send = st.button("Send ➤", use_container_width=True)

    if st.session_state.pending_question:
        st.session_state.pending_question = None

    # ── Process query ─────────────────────────────────────────────────────────
    if (send or user_input) and user_input.strip():
        query = user_input.strip()

        msg_id = len(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": query, "id": msg_id})
        render_message("user", query)

        with st.spinner("Searching Islamic knowledge base..."):
            results = search_fiqh(query, model, index, chunks, selected_category)
            context = build_context(results)
            answer = get_answer(client, query, context, st.session_state.messages[:-1])

        ans_id = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer, "id": ans_id})
        st.session_state.sources_map[ans_id] = results
        render_message("assistant", answer, results)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='footer'>
        ⚠️ <strong>Educational tool only — not a fatwa service.</strong>
        Consult a qualified Mufti for personal rulings. ·
        Primarily follows the <strong>Hanafi</strong> school. ·
        Created by <strong>Ubaid ur Rehman</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
