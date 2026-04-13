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
    df = df[["title", "url", "transcript"]].dropna()

    documents = []
    for i, row in df.iterrows():
        text = f"TITLE: {row['title']}\nURL: {row['url']}\n{row['transcript']}"
        documents.append({"doc_id": i, "title": row["title"], "url": row["url"], "text": text})

    model_emb = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model_emb.encode([d["text"] for d in documents], normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    return documents, model_emb, index

documents, embed_model, index = load_rag_system()

def retrieve(query, top_k=3):
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, top_k)

    results = []
    for s, i in zip(scores[0], indices[0]):
        results.append({**documents[i], "score": float(s)})
    return results

def ask_rag(query):
    results = retrieve(query)
    context = "\n".join([r["text"] for r in results])

    prompt = f"""
Answer only from context. No medical advice.

Q: {query}
Context:
{context}
"""
    return model.generate_content(prompt).text, results

# -----------------------------
# VOICE
# -----------------------------
def speech_to_text(audio_bytes, lang):
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name

    with sr.AudioFile(path) as source:
        audio = r.record(source)
        return r.recognize_google(audio, language=lang)

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        return f.name

# -----------------------------
# 🎨 CSS (UPDATED)
# -----------------------------
st.markdown("""
<style>

/* Floating mic */
.mic-float {
    position: fixed;
    bottom: 25px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
}

/* Remove all boxes */
div[data-testid="stAudioRecorder"],
div[data-testid="stAudioRecorder"] > div,
div[data-testid="stAudioRecorder"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Hide labels */
div[data-testid="stAudioRecorder"] p,
div[data-testid="stAudioRecorder"] span {
    display: none !important;
}

/* Theme colors */
@media (prefers-color-scheme: light) {
    div[data-testid="stAudioRecorder"] svg {
        fill: black !important;
    }
}

@media (prefers-color-scheme: dark) {
    div[data-testid="stAudioRecorder"] svg {
        fill: white !important;
    }
}

/* Bigger mic */
div[data-testid="stAudioRecorder"] svg {
    width: 40px !important;
    height: 40px !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI
# -----------------------------
st.title(get_text("title"))

query = st.text_input(get_text("ask_question"))

# 🎤 FLOATING MIC
st.markdown('<div class="mic-float">', unsafe_allow_html=True)
audio_bytes = audio_recorder(text="")
st.markdown('</div>', unsafe_allow_html=True)

if audio_bytes:
    query = speech_to_text(audio_bytes, lang_map[st.session_state.language]["stt"])
    st.success(query)

if st.button(get_text("ask_btn")):
    if query:
        ans, srcs = ask_rag(query)
        st.write(ans)

        for s in srcs:
            st.write(s["title"], s["url"])
