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

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="StillWater AI", layout="wide")

# -----------------------------
# GEMINI SETUP (UNCHANGED)
# -----------------------------
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -----------------------------
# LOAD RAG (UNCHANGED)
# -----------------------------
@st.cache_resource
def load_rag_system():
    df = pd.read_csv("diabetes_testimonials_only.csv")
    df = df[["title", "url", "transcript"]].fillna("")

    documents = []
    for i, row in df.iterrows():
        documents.append({
            "doc_id": i,
            "title": row["title"],
            "url": row["url"],
            "text": f"{row['title']}\n{row['transcript']}"
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
# RETRIEVE (UNCHANGED)
# -----------------------------
def retrieve(query, top_k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype("float32"), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            item = documents[idx].copy()
            item["score"] = float(score)
            results.append(item)
    return results

# -----------------------------
# RAG (UNCHANGED)
# -----------------------------
def ask_rag(query):
    results = retrieve(query)

    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""
Answer ONLY from testimonials.

Question: {query}

Context:
{context}
"""

    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": results
    }

# -----------------------------
# VOICE HELPERS (UNCHANGED)
# -----------------------------
def speech_to_text(audio_bytes, lang_code="en-IN"):
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        with sr.AudioFile(path) as source:
            audio = r.record(source)
            return r.recognize_google(audio, language=lang_code)
    except:
        return "Could not understand audio"
    finally:
        os.remove(path)

def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        return tmp.name

# -----------------------------
# PREMIUM UI CSS
# -----------------------------
st.markdown("""
<style>
html, body {
    background: #F8FAFC;
    font-family: 'Inter', sans-serif;
}

/* HEADER */
.header {
    background: linear-gradient(135deg,#4F8EF7,#7B61FF);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 15px;
}

/* CARD */
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    margin-top: 15px;
}

/* CHAT */
.user {
    background: #EEF2FF;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.bot {
    background: #E0F2FE;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}

/* BUTTON */
.stButton>button {
    border-radius: 10px;
    height: 45px;
    transition: 0.3s;
}
.stButton>button:hover {
    background: #4F8EF7;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="header">
<h2>🌸 StillWater AI</h2>
<p>Diabetes Testimonial Intelligence</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# HERO IMAGE
# -----------------------------
st.image("https://www.stillwater.you/images/c.png", use_container_width=True)

# -----------------------------
# LANGUAGE
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    input_lang = st.selectbox("Input Language", ["English", "Hindi"])
with col2:
    output_lang = st.selectbox("Output Voice", ["English", "Hindi"])

lang_map = {
    "English": {"stt": "en-IN", "tts": "en"},
    "Hindi": {"stt": "hi-IN", "tts": "hi"}
}

# -----------------------------
# CHAT HISTORY
# -----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# INPUT
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

audio_bytes = audio_recorder("🎙️ Speak")

if audio_bytes:
    query = speech_to_text(audio_bytes, lang_map[input_lang]["stt"])
else:
    query = st.text_input("💬 Ask your question")

if st.button("🚀 Ask") and query:
    result = ask_rag(query)
    st.session_state.chat.append(("user", query))
    st.session_state.chat.append(("bot", result))

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# CHAT DISPLAY
# -----------------------------
for role, msg in st.session_state.chat:

    if role == "user":
        st.markdown(f'<div class="user">🧑 {msg}</div>', unsafe_allow_html=True)

    else:
        st.markdown(f'<div class="bot">🤖 {msg["answer"]}</div>', unsafe_allow_html=True)

        # SOURCES
        st.markdown("**Sources:**")
        for s in msg["sources"]:
            st.markdown(f"- {s['title']} ({round(s['score'],3)})")

        # AUDIO
        audio_file = text_to_speech(msg["answer"], lang_map[output_lang]["tts"])
        st.audio(open(audio_file, "rb").read())
        os.remove(audio_file)
