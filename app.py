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
    page_title="SHARAN Conversational AI",
    layout="wide",
    page_icon="🎤"
)

# -----------------------------
# Gemini setup
# -----------------------------
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -----------------------------
# LANGUAGE CONFIG
# -----------------------------
if "language" not in st.session_state:
    st.session_state.language = "English"

def get_text(key):
    texts = {
        "english": {
            "title": "SHARAN Conversational AI",
            "voice_input": "Voice Input",
            "sample_questions": "Sample Questions",
            "reduce_medicines": "Reduce Medicines",
            "type2_reversed": "Type 2 Reversed",
            "diabetes_reversed": "Can Type 2 be Reversed?",
            "ask_question": "Ask your question",
            "ask_btn": "Ask",
            "answer": "Answer",
            "sources": "Sources",
            "voice_output": "Voice Output",
            "recognized": "Recognized:",
            "processing": "Processing speech...",
            "thinking": "Thinking...",
            "score": "Score:"
        },
        "hindi": {
            "title": "SHARAN बातचीत AI",
            "voice_input": "वॉइस इनपुट",
            "sample_questions": "नमूना प्रश्न",
            "reduce_medicines": "दवा कम करें",
            "type2_reversed": "टाइप 2 उलटा",
            "diabetes_reversed": "क्या टाइप 2 उलटा हो सकता है?",
            "ask_question": "अपना प्रश्न पूछें",
            "ask_btn": "पूछें",
            "answer": "उत्तर",
            "sources": "स्रोत",
            "voice_output": "वॉइस आउटपुट",
            "recognized": "पहचाना गया:",
            "processing": "आवाज प्रोसेस हो रही...",
            "thinking": "सोच रहे हैं...",
            "score": "स्कोर:"
        }
    }
    return texts[st.session_state.language.lower()][key]

lang_map = {
    "English": {"stt": "en-IN", "tts": "en"},
    "Hindi": {"stt": "hi-IN", "tts": "hi"}
}

# ❌ REMOVED TOP LANGUAGE DROPDOWN

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
# Retrieval & RAG
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
    .block-container {
        padding-top: 100px !important;
    }

    .lang-toggle-box {
        position: fixed;
        bottom: 18px;
        left: 18px;
        z-index: 9999;
        background: rgba(0,0,0,0.45);
        padding: 6px 10px;
        border-radius: 18px;
        font-size: 12px;
        color: white;
        backdrop-filter: blur(6px);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER (UNCHANGED)
# -----------------------------
st.markdown("""
<div class="hero-wrap">
    <img src="https://www.stillwater.you/images/logo.png" class="hero-logo">
    <div class="hero-title">""" + get_text("title") + """</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# MAIN UI (UNCHANGED)
# -----------------------------
st.markdown('<div class="panel-card">', unsafe_allow_html=True)

st.markdown(f'<div class="section-title">💡 {get_text("sample_questions")}</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    if st.button(get_text("reduce_medicines")):
        st.session_state["selected_query"] = "reduce medicines"

with c2:
    if st.button(get_text("type2_reversed")):
        st.session_state["selected_query"] = "type 2 reversed"

with c3:
    if st.button(get_text("diabetes_reversed")):
        st.session_state["selected_query"] = "can type 2 be reversed"

query = st.text_input(get_text("ask_question"))

audio_bytes = audio_recorder()

if audio_bytes:
    query = speech_to_text(audio_bytes)

if st.button(get_text("ask_btn")):
    result = ask_rag(query)
    st.write(result["answer"])

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# ✅ BOTTOM LANGUAGE TOGGLE
# -----------------------------
st.markdown('<div class="lang-toggle-box">🌐 हिंदी</div>', unsafe_allow_html=True)

if st.checkbox("Hindi", key="lang_toggle"):
    st.session_state.language = "Hindi"
else:
    st.session_state.language = "English"
