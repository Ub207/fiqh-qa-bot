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
    "Photography/tasweer ka shari hukm?",
    "Talaq ke kitni iqsaam hain?",
]

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Islamic Fiqh Q&A Assistant",
    page_icon="☪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Light Purple Theme + Anti-Flicker) ───────────────────────────
st.markdown("""
<style>
    /* ── Color Palette ── */
    :root {
        --primary:         #6A1B9A;
        --primary-light:   #9C27B0;
        --primary-lighter: #E1BEE7;
        --accent:          #AB47BC;
        --bg-main:         #F3E5F5;
        --bg-card:         #FFFFFF;
        --bg-sidebar:      #EDE7F6;
        --text-primary:    #311B47;
        --text-secondary:  #6A1B9A;
        --text-muted:      #7E57C2;
        --border:          #CE93D8;
        --gold:            #F9A825;
        --shadow:          rgba(106,27,154,0.1);
    }

    /* ── Anti-Flicker ── */
    .stApp {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    .main-header, .user-message, .assistant-message,
    .disclaimer-box, .sources-box, .footer {
        animation: none !important;
        transition: none !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    .element-container {
        animation: none !important;
        transition: none !important;
    }

    /* ── App Background ── */
    .stApp {
        background: linear-gradient(135deg, #F3E5F5 0%, #EDE7F6 50%, #F3E5F5 100%);
        color: var(--text-primary);
    }

    /* ── Header ── */
    .main-header {
        text-align: center;
        padding: 1.5rem 1rem 1.2rem;
        background: linear-gradient(135deg, #6A1B9A, #9C27B0);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(106,27,154,0.2);
    }
    .main-header h1 {
        font-size: 2rem;
        color: #FFFFFF;
        margin: 0.3rem 0;
        text-shadow: none;
    }
    .main-header p {
        color: #E1BEE7;
        font-size: 0.9rem;
        margin: 0.3rem 0 0;
    }
    .arabic-title {
        font-size: 1.4rem;
        color: #F9A825;
        direction: rtl;
        margin-bottom: 0.2rem;
        font-weight: 600;
    }

    /* ── Chat Messages ── */
    .user-message {
        background: linear-gradient(135deg, #7B1FA2, #9C27B0);
        border-radius: 18px 18px 4px 18px;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0 0.6rem auto;
        max-width: 82%;
        color: #FFFFFF;
        box-shadow: 0 2px 12px rgba(106,27,154,0.2);
    }
    .assistant-message {
        background: #FFFFFF;
        border: 1px solid #CE93D8;
        border-radius: 18px 18px 18px 4px;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0;
        max-width: 90%;
        color: #311B47;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .message-label {
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .user-label    { color: #E1BEE7; }
    .assistant-label { color: #9C27B0; }

    /* ── Sources Box ── */
    .sources-box {
        background: #F3E5F5;
        border: 1px solid #CE93D8;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        margin-top: 0.8rem;
        font-size: 0.8rem;
        color: #4A148C;
    }
    .sources-box strong { color: #6A1B9A; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EDE7F6, #E1BEE7) !important;
    }

    /* ── Disclaimer Box ── */
    .disclaimer-box {
        background: #FFF8E1;
        border: 1px solid #F9A825;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        color: #4A148C;
        margin: 0.5rem 0;
    }
    .disclaimer-box strong { color: #9C27B0; }

    /* ── Input ── */
    .stTextInput > div > div > input {
        background: #FFFFFF !important;
        border: 2px solid #CE93D8 !important;
        color: #311B47 !important;
        border-radius: 12px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #9C27B0 !important;
        box-shadow: 0 0 0 3px rgba(156,39,176,0.15) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7B1FA2, #9C27B0) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(106,27,154,0.25) !important;
        transition: none !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #9C27B0, #AB47BC) !important;
        box-shadow: 0 4px 16px rgba(106,27,154,0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: #FFFFFF !important;
        border: 1px solid #CE93D8 !important;
        color: #311B47 !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #7E57C2;
        font-size: 0.78rem;
        border-top: 1px solid #CE93D8;
        margin-top: 2rem;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #F3E5F5; }
    ::-webkit-scrollbar-thumb { background: #CE93D8; border-radius: 3px; }
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
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


# ── Rendering Helpers ─────────────────────────────────────────────────────────
def render_message(role, content, sources=None):
    if role == "user":
        st.markdown(
            f"<div class='user-message'>"
            f"<div class='message-label user-label'>🧑 You</div>"
            f"{content}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='assistant-message'>"
            f"<div class='message-label assistant-label'>☪️ Fiqh Assistant</div>"
            f"{content}",
            unsafe_allow_html=True,
        )
        if sources:
            refs_html = ""
            for _, chunk in sources[:3]:
                quran = ", ".join(chunk["quran_refs"]) if chunk["quran_refs"] else "—"
                hadith = (", ".join(chunk["hadith_refs"][:2])
                          if chunk["hadith_refs"] else "—")
                refs_html += (
                    f"<b>{chunk['category']}</b> | "
                    f"Quran: {quran} | Hadith: {hadith} | "
                    f"Source: {chunk['source']}<br>"
                )
            st.markdown(
                f"<div class='sources-box'>"
                f"<strong>📖 References Used:</strong><br>{refs_html}"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("</div>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(categories):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:0.5rem 0 1rem;'>
            <div style='font-size:2.5rem;'>☪️</div>
            <div style='color:#6A1B9A; font-weight:700; font-size:1.1rem;'>
                Islamic Fiqh Assistant
            </div>
            <div style='color:#7E57C2; font-size:0.8rem;'>Hanafi School | حنفی مذہب</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📚 Filter by Category")
        cat_names = ["All"] + [c["name"] for c in categories]
        selected = st.selectbox("Category", cat_names, label_visibility="collapsed")

        st.markdown("### 💡 Example Questions")
        for q in EXAMPLE_QUESTIONS[:6]:
            if st.button(q, key=f"eq_{q[:20]}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

        st.markdown("---")
        st.markdown("""
        <div class='disclaimer-box'>
            <strong>⚠️ Important Disclaimers</strong><br><br>
            🔹 This is an <strong>AI-powered educational tool</strong>,
            <strong>NOT a fatwa service</strong>.<br><br>
            🔹 For personal rulings, consult a <strong>qualified Mufti</strong>.<br><br>
            🔹 Primarily follows the <strong>Hanafi school</strong>.<br><br>
            🔹 Sources: Fatawa Darul Uloom Deoband &amp; classical Hanafi texts.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='text-align:center; color:#9C27B0; font-size:0.75rem;'>
            Created by <strong style='color:#6A1B9A;'>Ubaid ur Rehman</strong><br>
            Aalim | Islamic Fiqh Specialist
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
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

    # Header
    st.markdown("""
    <div class='main-header'>
        <div class='arabic-title'>مساعد الفقه الإسلامي</div>
        <h1>☪️ Islamic Fiqh Q&amp;A Assistant</h1>
        <p>Answers based on Hanafi school · Fatawa Darul Uloom Deoband · Quran &amp; Hadith references</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Render saved chat history ─────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; padding:3rem 1rem; color:#7E57C2;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>📖</div>
            <div style='font-size:1.1rem; color:#6A1B9A; font-weight:600;'>
                Assalamu Alaykum! Ask any question about Islamic Fiqh.
            </div>
            <div style='color:#7E57C2; margin-top:0.5rem; font-size:0.9rem;'>
                السلام علیکم! اسلامی فقہ کے بارے میں کوئی بھی سوال پوچھیں۔
            </div>
            <div style='margin-top:1.5rem; color:#AB47BC; font-size:0.85rem;'>
                Use the example questions in the sidebar or type your own below.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            sources = st.session_state.sources_map.get(msg.get("id"))
            render_message(
                msg["role"], msg["content"],
                sources if msg["role"] == "assistant" else None,
            )

    # ── Input area ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([6, 1])
    with col1:
        default_val = st.session_state.pending_question or ""
        user_input = st.text_input(
            "Ask a fiqh question",
            value=default_val,
            placeholder="e.g., What breaks wudu? | Zakat ki nisab kya hai?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col2:
        send = st.button("Send ➤", use_container_width=True)

    # Clear pending after it has been placed in the input box
    if st.session_state.pending_question:
        st.session_state.pending_question = None

    # ── Process query (NO st.rerun — render inline immediately) ──────────────
    if (send or user_input) and user_input.strip():
        query = user_input.strip()

        # Save user message to state
        msg_id = len(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": query, "id": msg_id})

        # Render user message right now
        render_message("user", query)

        # Get answer
        with st.spinner("Searching Islamic knowledge base..."):
            results = search_fiqh(query, model, index, chunks, selected_category)
            context = build_context(results)
            answer = get_answer(client, query, context, st.session_state.messages[:-1])

        # Save assistant message to state
        ans_id = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer, "id": ans_id})
        st.session_state.sources_map[ans_id] = results

        # Render assistant message right now — no st.rerun() needed
        render_message("assistant", answer, results)

    # Footer
    st.markdown("""
    <div class='footer'>
        ⚠️ <strong>Disclaimer:</strong> Educational tool only — <strong>NOT a fatwa service</strong>.
        Consult a qualified Mufti for personal rulings. |
        Primarily follows <strong>Hanafi</strong> school. |
        Sources: <strong>Fatawa Darul Uloom Deoband</strong> &amp; classical Hanafi texts. |
        Created by <strong>Ubaid ur Rehman</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
