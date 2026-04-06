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

st.set_page_config(page_title="Diabetes Testimonial Chatbot", layout="wide")

# -----------------------------
# Gemini setup
# -----------------------------
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -----------------------------
# Load + RAG (UNCHANGED)
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

def retrieve(query, top_k=3):
    q = embed_model.encode([query], convert_to_numpy=True,
                           normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q, top_k)
    return [{**documents[i], "score": float(s)} for s, i in zip(scores[0], idxs[0])]

def ask_rag(query):
    results = retrieve(query)
    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""
You are a testimonial-based assistant.

Rules:
- Answer ONLY from context
- No medical advice
- Keep concise

Q: {query}
Context:
{context}
"""
    res = model.generate_content(prompt)

    return {
        "answer": res.text,
        "sources": results
    }

# -----------------------------
# Voice
# -----------------------------
def speech_to_text(audio_bytes):
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name

    try:
        with sr.AudioFile(path) as src:
            audio = r.record(src)
            return r.recognize_google(audio)
    except:
        return "Error recognizing speech"
    finally:
        os.remove(path)

def text_to_speech(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        return f.name

# -----------------------------
# UI DESIGN
# -----------------------------
st.markdown("""
<style>
body {
    background:#F8FAFC;
    font-family: 'Inter', sans-serif;
}

/* HEADER */
.header {
background: linear-gradient(135deg,#4F8EF7,#7B61FF);
padding:25px;
border-radius:20px;
color:white;
background-image:url('https://images.unsplash.com/photo-1506744038136-46273834b3fb');
background-size:cover;
}

/* CARD */
.card {
background:white;
padding:20px;
border-radius:15px;
box-shadow:0 4px 10px rgba(0,0,0,0.05);
margin-bottom:20px;
}

/* BUTTON */
button {
transition:0.3s;
border-radius:10px;
}
button:hover {
transform:scale(1.05);
background:#4F8EF7 !important;
color:white !important;
}

/* TEXT AREA */
textarea {
border-radius:12px !important;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="header">
<h1>🌸 Diabetes Testimonial Chatbot</h1>
<p>AI-powered insights from real patient stories</p>
</div>
""", unsafe_allow_html=True)

# INPUT + RESPONSE LAYOUT
col1, col2 = st.columns([1,2])

# ---------------- LEFT (INPUT)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 🎤 Input")

    c1, c2 = st.columns(2)
    with c1:
        audio_bytes = audio_recorder(text="🎤 Voice")
    with c2:
        st.button("⌨️ Text")

    if audio_bytes:
        st.session_state["query"] = speech_to_text(audio_bytes)

    query = st.text_area("Type your question",
                         value=st.session_state.get("query",""),
                         placeholder="Ask about diabetes, diet, lifestyle...")

    st.markdown("#### 🌐 Language")
    lang = st.radio("", ["English", "Hindi"], horizontal=True)

    ask_btn = st.button("🚀 Ask StillWater")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT (RESPONSE)
with col2:
    if ask_btn and query:

        result = ask_rag(query)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("### 🤖 AI Response")
        st.markdown(result["answer"])

        st.markdown("---")
        st.markdown("### 💡 Key Takeaways")
        st.markdown("""
- Patients reduced medication  
- HbA1c improved  
- Lifestyle consistency matters  
""")

        st.markdown("</div>", unsafe_allow_html=True)

        # SOURCES
        st.markdown("### 📊 Sources")
        cols = st.columns(3)

        for i, src in enumerate(result["sources"]):
            with cols[i % 3]:
                st.image("https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg")
                st.caption(src["title"])

# SUGGESTED QUESTIONS
st.markdown("### 💡 Suggested Questions")
s1, s2, s3 = st.columns(3)

with s1:
    st.button("Reduce medicines naturally")

with s2:
    st.button("Best diet for diabetes")

with s3:
    st.button("How long to improve?")
