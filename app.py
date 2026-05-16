import os
import json
import pickle
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
INDEX_PATH = "faiss_index/fiqh.index"
CHUNKS_PATH = "faiss_index/chunks.pkl"
CATEGORIES_PATH = "fiqh_data/fiqh_categories.json"
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

SYSTEM_PROMPT = """You are an Islamic Fiqh Q&A Assistant created by Ubaid ur Rehman, \
an Aalim specializing in Islamic Fiqh (primarily Hanafi school, as followed in Pakistan/India). \

Your role:
- Answer fiqh questions using the provided knowledge base excerpts.
- Always cite relevant Quran verses and Hadith references from the context.
- Clearly mention the madhab/school of thought (primarily Hanafi).
- If the question requires a personal fatwa or a complex ruling on a specific situation, \
  advise the user to consult a qualified Mufti.
- Answer in the same language the user uses (English or Urdu).
- Be respectful, scholarly, and accurate. Do not fabricate hadith references.
- If the provided context does not contain enough information, say so clearly \
  rather than inventing an answer.
- Format answers clearly with references at the end when appropriate.

IMPORTANT DISCLAIMER: This is an educational tool, not a fatwa service. \
Always recommend consulting a qualified scholar for personal rulings."""

EXAMPLE_QUESTIONS = [
    "Wudu karne ka tareeqa kya hai?",
    "What breaks the fast during Ramadan?",
    "Zakat ki nisab kitni hai?",
    "Can I pray sitting if I'm sick?",
    "Nikah ke arkaan kya hain?",
    "Is seafood halal in Hanafi fiqh?",
    "What nullifies salah?",
    "Who are the eligible recipients of Zakat?",
    "What is the ruling on riba (interest)?",
    "How many days does hayd last in Hanafi fiqh?",
]

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Islamic Fiqh Q&A Assistant",
    page_icon="☪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Primary purple theme */
    :root {
        --primary: #4A148C;
        --primary-light: #7B1FA2;
        --primary-lighter: #CE93D8;
        --accent: #E040FB;
        --bg-dark: #1a0a2e;
        --bg-card: #2d1b4e;
        --text-light: #f3e5f5;
    }

    .stApp {
        background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 50%, #1a0a2e 100%);
        color: var(--text-light);
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
        border-bottom: 2px solid #7B1FA2;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2.2rem;
        color: #CE93D8;
        margin: 0;
        text-shadow: 0 0 20px #7B1FA2;
    }
    .main-header p {
        color: #b39ddb;
        font-size: 0.95rem;
        margin: 0.4rem 0 0;
    }
    .arabic-title {
        font-size: 1.4rem;
        color: #E040FB;
        direction: rtl;
        margin-bottom: 0.3rem;
    }

    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #4A148C, #6A1B9A);
        border-radius: 18px 18px 4px 18px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 85%;
        margin-left: auto;
        color: #f3e5f5;
        box-shadow: 0 2px 10px rgba(74,20,140,0.4);
    }
    .assistant-message {
        background: linear-gradient(135deg, #1e0a35, #2d1b4e);
        border: 1px solid #7B1FA2;
        border-radius: 18px 18px 18px 4px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 90%;
        color: #f3e5f5;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .message-label {
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        letter-spacing: 0.5px;
    }
    .user-label { color: #CE93D8; }
    .assistant-label { color: #E040FB; }

    /* Sources box */
    .sources-box {
        background: rgba(74,20,140,0.2);
        border: 1px solid #4A148C;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-top: 0.8rem;
        font-size: 0.82rem;
        color: #b39ddb;
    }
    .sources-box strong { color: #CE93D8; }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0a2e, #2d1b4e) !important;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: rgba(224,64,251,0.08);
        border: 1px solid #7B1FA2;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        color: #b39ddb;
        margin: 0.5rem 0;
    }
    .disclaimer-box strong { color: #E040FB; }

    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #2d1b4e !important;
        border: 1px solid #7B1FA2 !important;
        color: #f3e5f5 !important;
        border-radius: 10px !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #4A148C, #7B1FA2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6A1B9A, #9C27B0) !important;
        box-shadow: 0 4px 15px rgba(74,20,140,0.5) !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #2d1b4e !important;
        border: 1px solid #7B1FA2 !important;
        color: #f3e5f5 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #6d4c8c;
        font-size: 0.78rem;
        border-top: 1px solid #4A148C;
        margin-top: 2rem;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a0a2e; }
    ::-webkit-scrollbar-thumb { background: #4A148C; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Resource Loading ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
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
def search_fiqh(query: str, model, index, chunks, category_filter: str = "All", k: int = TOP_K):
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
    context_parts = []
    for i, (score, chunk) in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}] Category: {chunk['category']}\n"
            f"Q: {chunk['question']}\n"
            f"A: {chunk['answer']}\n"
            f"Quran Refs: {', '.join(chunk['quran_refs']) if chunk['quran_refs'] else 'N/A'}\n"
            f"Hadith Refs: {', '.join(chunk['hadith_refs']) if chunk['hadith_refs'] else 'N/A'}\n"
            f"Madhab: {chunk['madhab']} | Source: {chunk['source']}"
        )
    return "\n\n---\n\n".join(context_parts)


# ── LLM Response ─────────────────────────────────────────────────────────────
def get_answer(client, query: str, context: str, chat_history: list):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in chat_history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    user_msg = (
        f"Based on the following fiqh knowledge base:\n\n{context}\n\n"
        f"Please answer this question: {query}"
    )
    messages.append({"role": "user", "content": user_msg})
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── UI ────────────────────────────────────────────────────────────────────────
def render_sidebar(categories):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 0.5rem 0 1rem;'>
            <div style='font-size:2.5rem;'>☪️</div>
            <div style='color:#CE93D8; font-weight:700; font-size:1.1rem;'>Islamic Fiqh Assistant</div>
            <div style='color:#9c27b0; font-size:0.8rem;'>Hanafi School | حنفی مذہب</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📚 Filter by Category")
        cat_names = ["All"] + [c["name"] for c in categories]
        selected = st.selectbox("Category", cat_names, label_visibility="collapsed")

        st.markdown("### 💡 Example Questions")
        for q in EXAMPLE_QUESTIONS[:6]:
            if st.button(q, key=f"eq_{q[:20]}", use_container_width=True):
                st.session_state.pending_question = q

        st.markdown("---")
        st.markdown("""
        <div class='disclaimer-box'>
            <strong>⚠️ Important Disclaimers</strong><br><br>
            🔹 This is an <strong>AI-powered educational tool</strong>, <strong>NOT a fatwa service</strong>.<br><br>
            🔹 For rulings specific to your personal situation, please consult a <strong>qualified Mufti</strong>.<br><br>
            🔹 Primarily follows the <strong>Hanafi school</strong> of jurisprudence.<br><br>
            🔹 Citations are from established Islamic texts — verify with scholarly sources.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='text-align:center; color:#6d4c8c; font-size:0.75rem;'>
            Created by <strong style='color:#CE93D8;'>Ubaid ur Rehman</strong><br>
            Aalim | Islamic Fiqh Specialist
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return selected


def render_message(role: str, content: str, sources=None):
    if role == "user":
        st.markdown(f"""
        <div class='user-message'>
            <div class='message-label user-label'>🧑 You</div>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='assistant-message'>
            <div class='message-label assistant-label'>☪️ Fiqh Assistant</div>
            {content}
        """, unsafe_allow_html=True)
        if sources:
            refs_html = ""
            for _, chunk in sources[:3]:
                quran = ", ".join(chunk["quran_refs"]) if chunk["quran_refs"] else "—"
                hadith = ", ".join(chunk["hadith_refs"][:2]) if chunk["hadith_refs"] else "—"
                refs_html += (
                    f"<b>{chunk['category']}</b> | "
                    f"Quran: {quran} | Hadith: {hadith} | "
                    f"Source: {chunk['source']}<br>"
                )
            st.markdown(f"""
            <div class='sources-box'>
                <strong>📖 References Used:</strong><br>{refs_html}
            </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)


def main():
    model, index, chunks = load_resources()
    categories = load_categories()
    client = get_groq_client()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "sources_map" not in st.session_state:
        st.session_state.sources_map = {}

    selected_category = render_sidebar(categories)

    # Header
    st.markdown("""
    <div class='main-header'>
        <div class='arabic-title'>مساعد الفقه الإسلامي</div>
        <h1>☪️ Islamic Fiqh Q&A Assistant</h1>
        <p>Ask questions about Islamic jurisprudence — Answers based on Hanafi school with Quran & Hadith references</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem; color:#7B1FA2;'>
                <div style='font-size:3rem; margin-bottom:1rem;'>📖</div>
                <div style='font-size:1.1rem; color:#CE93D8; font-weight:600;'>
                    Assalamu Alaykum! Ask any question about Islamic Fiqh.
                </div>
                <div style='color:#9c27b0; margin-top:0.5rem; font-size:0.9rem;'>
                    السلام علیکم! اسلامی فقہ کے بارے میں کوئی بھی سوال پوچھیں۔
                </div>
                <div style='margin-top:1.5rem; color:#6d4c8c; font-size:0.85rem;'>
                    Use the example questions in the sidebar or type your own below.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                sources = st.session_state.sources_map.get(msg.get("id"))
                render_message(msg["role"], msg["content"], sources if msg["role"] == "assistant" else None)

    # Input area
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([6, 1])
    with col1:
        pending = st.session_state.pending_question or ""
        user_input = st.text_input(
            "Ask a fiqh question...",
            value=pending,
            placeholder="e.g., What breaks wudu? | Zakat ki nisab kya hai?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col2:
        send = st.button("Send ➤", use_container_width=True)

    if st.session_state.pending_question:
        st.session_state.pending_question = None

    if (send or user_input) and user_input.strip():
        query = user_input.strip()

        msg_id = len(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": query, "id": msg_id})

        with st.spinner("Searching Islamic knowledge base..."):
            results = search_fiqh(query, model, index, chunks, selected_category)
            context = build_context(results)
            answer = get_answer(client, query, context, st.session_state.messages[:-1])

        ans_id = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer, "id": ans_id})
        st.session_state.sources_map[ans_id] = results

        st.rerun()

    # Footer
    st.markdown("""
    <div class='footer'>
        ⚠️ <strong>Disclaimer:</strong> This AI tool is for <strong>educational purposes only</strong> and is
        <strong>NOT a fatwa service</strong>. For personal Islamic rulings, consult a qualified Mufti. |
        Primarily follows <strong>Hanafi</strong> school of jurisprudence. |
        Created by <strong>Ubaid ur Rehman</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
