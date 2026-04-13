import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import tempfile
import os

st.set_page_config(
    page_title="Diabetes Testimonial Chatbot",
    layout="wide",
    page_icon="🌿"
)

# -----------------------------
# Gemini setup
# -----------------------------
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -----------------------------
# Load and prepare data
# -----------------------------
@st.cache_resource
def load_rag_system():
    csv_path = "diabetes_testimonials_only.csv"

    df = pd.read_csv(csv_path)
    df = df[["title", "url", "transcript"]].copy()

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["transcript"] = df["transcript"].fillna("").astype(str).str.strip()

    df = df[df["transcript"] != ""].reset_index(drop=True)

    documents = []
    for i, row in df.iterrows():
        doc_text = f"""TITLE: {row['title']}
URL: {row['url']}
TRANSCRIPT:
{row['transcript']}"""

        documents.append({
            "doc_id": i,
            "title": row["title"],
            "url": row["url"],
            "text": doc_text
        })

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    doc_texts = [d["text"] for d in documents]
    doc_embeddings = embed_model.encode(
        doc_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(doc_embeddings.astype("float32"))

    return documents, embed_model, index

documents, embed_model, index = load_rag_system()

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, top_k=3):
    q_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = documents[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return results

# -----------------------------
# RAG
# -----------------------------
def ask_rag(query, top_k=3):
    results = retrieve(query, top_k=top_k)

    context = "\n\n".join([
        f"""SOURCE {i+1}
TITLE: {r['title']}
URL: {r['url']}
CONTENT:
{r['text']}"""
        for i, r in enumerate(results)
    ])

    prompt = f"""
You are a testimonial-based assistant.

Rules:
1. Answer only from the provided testimonial context.
2. If the answer is not clearly present, say: "Not found clearly in the testimonials."
3. Do not give medical advice.
4. Mention relevant source title and URL.
5. Keep the answer clear and short.

User question:
{query}

Context:
{context}
"""

    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": [
            {"title": r["title"], "url": r["url"], "score": r["score"]}
            for r in results
        ]
    }

# -----------------------------
# Voice helpers
# -----------------------------
def speech_to_text(audio_bytes, lang_code="en-IN"):
    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language=lang_code)
        return text
    except Exception as e:
        return f"Speech recognition failed: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        return tmp_file.name

# -----------------------------
# THEME / UI
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background:
            linear-gradient(rgba(15, 18, 22, 0.50), rgba(15, 18, 22, 0.50)),
            url('https://stillwater-main.onrender.com/images/c.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1250px;
    }

    .main-shell {
        padding: 0.2rem 0.2rem 1rem 0.2rem;
    }

    .hero-wrap {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.18);
        border-radius: 26px;
        padding: 1.15rem 1.4rem 1.2rem 1.4rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .hero-logo {
        height: 54px;
        margin: 0 auto 0.4rem auto;
        display: block;
        object-fit: contain;
    }

    .hero-title {
        color: #f7f4ee;
        font-size: 2.05rem;
        line-height: 1.15;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: rgba(255, 255, 255, 0.88);
        font-size: 0.98rem;
        line-height: 1.3;
        font-weight: 500;
        margin-top: 0.45rem;
    }

    .panel-card {
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.14);
        border-radius: 22px;
        padding: 1.2rem;
        color: white;
        margin-top: 0.6rem;
    }

    .section-title {
        color: #ffffff;
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        line-height: 1.2;
    }

    .stSelectbox label,
    .stTextInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 14px !important;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.88) !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        color: #1b1b1b !important;
        min-height: 48px;
    }

    .stTextInput > div > div > input::placeholder {
        color: #6b7280;
    }

    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.88) !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 46px;
        border: none;
        border-radius: 14px;
        font-weight: 700;
        transition: all 0.25s ease;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: rgba(255, 255, 255, 0.82);
        color: #25313d;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.96);
    }

    .ask-row .stButton > button,
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #d6b36a, #b38a3d);
        color: white;
    }

    .answer-box {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        color: white;
        line-height: 1.55;
        margin-top: 0.5rem;
    }

    .sources-box {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        color: white;
    }

    .sources-box a {
        color: #f5d58b !important;
        text-decoration: none;
    }

    .sources-box a:hover {
        text-decoration: underline;
    }

    .audio-box {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 0.75rem;
        margin-top: 0.35rem;
    }

    .stAudio {
        border-radius: 14px;
        overflow: hidden;
    }

    hr {
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.15);
        margin: 1rem 0;
    }

    .mini-note {
        color: rgba(255, 255, 255, 0.78);
        font-size: 0.92rem;
        line-height: 1.35;
        margin-top: -0.2rem;
        margin-bottom: 0.7rem;
    }

    @media (max-width: 768px) {
        .hero-wrap {
            padding: 1rem 1rem 1.05rem 1rem;
            border-radius: 20px;
        }

        .hero-logo {
            height: 44px;
            margin-bottom: 0.3rem;
        }

        .hero-title {
            font-size: 1.55rem;
        }

        .hero-subtitle {
            font-size: 0.9rem;
        }

        .panel-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="main-shell">
    <div class="hero-wrap">
        <img src="https://www.stillwater.you/images/logo.png" class="hero-logo" alt="StillWater Logo">
        <div class="hero-title">Diabetes Testimonial Chatbot</div>
        <div class="hero-subtitle">🌿 AI-powered insights from real patient stories</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# LANGUAGE
# -----------------------------
col_lang1, col_lang2 = st.columns(2)

with col_lang1:
    input_lang = st.selectbox("🌐 Input Language", ["English", "Hindi"])

with col_lang2:
    output_lang = st.selectbox("🔊 Output Voice Language", ["English", "Hindi"])

lang_map = {
    "English": {"stt": "en-IN", "tts": "en"},
    "Hindi": {"stt": "hi-IN", "tts": "hi"}
}

preset_questions = [
    "Find testimonials where people reduced diabetes medicine after switching to plant-based diet",
    "Find patient stories that talk about plant-based diet helping diabetes",
    "Find testimonials where patients describe how long they had diabetes"
]

left, right = st.columns([1, 2], gap="large")

# -----------------------------
# LEFT PANEL
# -----------------------------
with left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎤 Voice Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="mini-note">Record your question using voice input.</div>', unsafe_allow_html=True)

    # Fixed audio_recorder call
    audio_bytes = audio_recorder(
        text="Click to record",
        pause_threshold=2.0
    )

    if audio_bytes:
        with st.spinner("Processing speech..."):
            recognized_text = speech_to_text(
                audio_bytes,
                lang_code=lang_map[input_lang]["stt"]
            )
            st.session_state["selected_query"] = recognized_text
        st.success(f"Recognized: {recognized_text}")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# RIGHT PANEL
# -----------------------------
with right:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">💡 Sample Questions</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Reduce medicines"):
            st.session_state["selected_query"] = preset_questions[0]

    with c2:
        if st.button("Plant diet helps"):
            st.session_state["selected_query"] = preset_questions[1]

    with c3:
        if st.button("Duration"):
            st.session_state["selected_query"] = preset_questions[2]

    st.markdown("<hr>", unsafe_allow_html=True)

    default_query = st.session_state.get("selected_query", "")
    query = st.text_input("💬 Ask your question", value=default_query)

    if st.button("🚀 Ask") and query.strip():
        with st.spinner("Thinking..."):
            result = ask_rag(query.strip(), top_k=3)

        st.markdown('<div class="section-title">🧠 Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📚 Sources</div>', unsafe_allow_html=True)

        for i, src in enumerate(result["sources"], start=1):
            st.markdown(
                f"""
                <div class="sources-box">
                    <strong>{i}. {src['title']}</strong><br>
                    🔗 <a href="{src['url']}" target="_blank">{src['url']}</a><br>
                    ⭐ Score: {round(src['score'], 4)}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔊 Voice Output</div>', unsafe_allow_html=True)

        audio_file = text_to_speech(
            result["answer"],
            lang=lang_map[output_lang]["tts"]
        )

        with open(audio_file, "rb") as f:
            out_audio_bytes = f.read()
            st.markdown('<div class="audio-box">', unsafe_allow_html=True)
            st.audio(out_audio_bytes, format="audio/mp3")
            st.markdown('</div>', unsafe_allow_html=True)

        if os.path.exists(audio_file):
            os.remove(audio_file)

    st.markdown('</div>', unsafe_allow_html=True)
