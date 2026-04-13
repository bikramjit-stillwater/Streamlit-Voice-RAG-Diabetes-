# -----------------------------
# SAME IMPORTS (UNCHANGED)
# -----------------------------
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
# Gemini setup (UNCHANGED)
# -----------------------------
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -----------------------------
# LANGUAGE CONFIG (UNCHANGED)
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
# DATA + RAG (UNCHANGED)
# -----------------------------
@st.cache_resource
def load_rag_system():
    df = pd.read_csv("diabetes_testimonials_only.csv")
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
    embeddings = embed_model.encode(
        [d["text"] for d in documents],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    return documents, embed_model, index

documents, embed_model, index = load_rag_system()

def retrieve(query, top_k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, top_k)

    results = []
    for s, i in zip(scores[0], indices[0]):
        if i != -1:
            d = documents[i].copy()
            d["score"] = float(s)
            results.append(d)
    return results

def ask_rag(query):
    results = retrieve(query)
    context = "\n\n".join([r["text"] for r in results])
    response = model.generate_content(f"Answer using only context:\n{context}\n\nQ:{query}")
    return {
        "answer": response.text,
        "sources": results
    }

# -----------------------------
# VOICE (UNCHANGED)
# -----------------------------
def speech_to_text(audio_bytes, lang_code):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        with sr.AudioFile(path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio, language=lang_code)
    except Exception as e:
        return str(e)
    finally:
        os.remove(path)

# -----------------------------
# 🔥 UI CSS (UPDATED)
# -----------------------------
st.markdown("""
<style>

/* INLINE INPUT + MIC */
.mic-inline {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 46px;
    margin-top: 22px;
}

/* remove mic box */
div[data-testid="stAudioRecorder"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

div[data-testid="stAudioRecorder"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* hide labels */
div[data-testid="stAudioRecorder"] p,
div[data-testid="stAudioRecorder"] span {
    display: none !important;
}

/* mic size */
div[data-testid="stAudioRecorder"] svg {
    width: 28px !important;
    height: 28px !important;
}

/* THEME COLORS */
@media (prefers-color-scheme: light) {
    div[data-testid="stAudioRecorder"] svg {
        fill: black !important;
    }
    .lang-checkbox label {
        color: black !important;
    }
}

@media (prefers-color-scheme: dark) {
    div[data-testid="stAudioRecorder"] svg {
        fill: white !important;
    }
    .lang-checkbox label {
        color: white !important;
    }
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# INPUT + MIC INLINE
# -----------------------------
default_query = st.session_state.get("selected_query", "")

col1, col2 = st.columns([10,1])

with col1:
    query = st.text_input("💬 Ask your question", value=default_query)

with col2:
    st.markdown('<div class="mic-inline">', unsafe_allow_html=True)
    audio_bytes = audio_recorder(text="")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# AUDIO INPUT HANDLING
# -----------------------------
if audio_bytes:
    recognized = speech_to_text(audio_bytes, lang_map[st.session_state.language]["stt"])
    query = recognized
    st.success(f"Recognized: {recognized}")

# -----------------------------
# ASK BUTTON
# -----------------------------
if st.button("🚀 Ask"):
    if query:
        result = ask_rag(query)
        st.write(result["answer"])
