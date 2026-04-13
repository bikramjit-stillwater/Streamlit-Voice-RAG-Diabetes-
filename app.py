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
            "input_lang": "🌐 Input Language",
            "output_lang": "🔊 Output Voice Language",
            "voice_input": "🎤 Voice Input",
            "sample_questions": "💡 Sample Questions",
            "reduce_medicines": "Reduce Medicines",
            "type2_reversed": "Type 2 Reversed",
            "diabetes_reversed": "Can Type 2 be Reversed?",
            "ask_question": "💬 Ask your question",
            "ask_btn": "🚀 Ask",
            "answer": "🧠 Answer",
            "sources": "📚 Sources",
            "voice_output": "🔊 Voice Output",
            "recognized": "Recognized:",
            "processing": "Processing speech...",
            "thinking": "Thinking...",
            "score": "Score:"
        },
        "hindi": {
            "title": "SHARAN बातचीत AI",
            "input_lang": "🌐 इनपुट भाषा",
            "output_lang": "🔊 आउटपुट आवाज भाषा",
            "voice_input": "🎤 वॉइस इनपुट",
            "sample_questions": "💡 नमूना प्रश्न",
            "reduce_medicines": "दवा कम करें",
            "type2_reversed": "टाइप 2 उलटा",
            "diabetes_reversed": "क्या टाइप 2 उलटा हो सकता है?",
            "ask_question": "💬 अपना प्रश्न पूछें",
            "ask_btn": "🚀 पूछें",
            "answer": "🧠 उत्तर",
            "sources": "📚 स्रोत",
            "voice_output": "🔊 वॉइस आउटपुट",
            "recognized": "पहचाना गया:",
            "processing": "आवाज प्रोसेस हो रही...",
            "thinking": "सोच रहे हैं...",
            "score": "स्कोर:"
        }
    }
    return texts[st.session_state.language.lower()][key]

# Language toggle in header
col1, col2 = st.columns([1, 2])
with col2:
    st.session_state.language = st.selectbox(
        "🌐 Language / भाषा",
        ["English", "Hindi"],
        index=0 if st.session_state.language == "English" else 1,
        key="lang_toggle"
    )

lang_map = {
    "English": {"stt": "en-IN", "tts": "en"},
    "Hindi": {"stt": "hi-IN", "tts": "hi"}
}

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
# Retrieval & RAG (unchanged)
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
    .stApp {
        background: linear-gradient(rgba(15, 18, 22, 0.50), rgba(15, 18, 22, 0.50)),
                    url('https://stillwater-main.onrender.com/images/c.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .hero-wrap {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 26px;
        padding: 1.15rem 1.4rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .hero-logo { height: 54px; margin-bottom: 0.4rem; display: block; }
    .hero-title {
        color: #f7f4ee; font-size: 2.05rem; font-weight: 800;
        margin: 0; letter-spacing: -0.02em;
    }
    .panel-card {
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 22px;
        padding: 1.2rem;
        color: white;
        margin-top: 0.6rem;
    }
    .section-title { color: #ffffff; font-size: 1.08rem; font-weight: 700; margin-bottom: 0.9rem; }
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.88) !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 14px !important;
        min-height: 48px; color: #1b1b1b;
    }
    .stButton >
