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

# -----------------------------
# Load RAG
# -----------------------------
@st.cache_resource
def load_rag_system():
    df = pd.read_csv("diabetes_testimonials_only.csv")
    df = df[["title", "url", "transcript"]].fillna("").astype(str)

    documents = []
    for i, row in df.iterrows():
        documents.append({
            "doc_id": i,
            "title": row["title"],
            "url": row["url"],
            "text": row["transcript"]
        })

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(
        [d["text"] for d in documents],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    return documents, embed_model, index

documents, embed_model, index = load_rag_system()

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, top_k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype("float32"), top_k)

    return [
        {**documents[idx], "score": float(score)}
        for score, idx in zip(scores[0], indices[0]) if idx != -1
    ]

def ask_rag(query):
    results = retrieve(query)

    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""
Answer only from context.
If unclear say NOT FOUND.

Question: {query}
Context: {context}
"""

    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": results
    }

# -----------------------------
# Voice
# -----------------------------
def speech_to_text(audio_bytes, lang_code="en-IN"):
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name

    try:
        with sr.AudioFile(path) as src:
            audio = r.record(src)
            return r.recognize_google(audio, language=lang_code)
    except:
        return "Error"
    finally:
        os.remove(path)

def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        return f.name

# -----------------------------
# UI FIXES (IMPORTANT)
# -----------------------------
st.markdown("""
<style>

/* PUSH DOWN UI */
.block-container {
    padding-top: 120px !important;
}

/* BOTTOM LANGUAGE TOGGLE */
.lang-toggle {
    position: fixed;
    bottom: 20px;
    left: 20px;
    background: rgba(0,0,0,0.5);
    padding: 6px 12px;
    border-radius: 20px;
    color: white;
    font-size: 12px;
    backdrop-filter: blur(6px);
    z-index: 9999;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(f"""
<div style="text-align:center; color:white; font-size:28px; font-weight:bold;">
{get_text("title")}
</div>
""", unsafe_allow_html=True)

# -----------------------------
# INPUT
# -----------------------------
query = st.text_input(get_text("ask_question"))

audio = audio_recorder()

if audio:
    query = speech_to_text(audio)

# -----------------------------
# ASK
# -----------------------------
if st.button(get_text("ask_btn")):
    result = ask_rag(query)

    st.write(result["answer"])

# -----------------------------
# BOTTOM LANGUAGE TOGGLE
# -----------------------------
st.markdown('<div class="lang-toggle">🌐 हिंदी</div>', unsafe_allow_html=True)

if st.checkbox("Switch to Hindi"):
    st.session_state.language = "Hindi"
else:
    st.session_state.language = "English"
