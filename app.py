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

    all_docs = []
    for i, row in df.iterrows():
        block = f"""TESTIMONIAL_ID: {i}
TITLE: {row['title']}
URL: {row['url']}
TRANSCRIPT:
{row['transcript']}

{'='*100}
"""
        all_docs.append(block)

    with open("all_patient_testimonials.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_docs))

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

    return df, documents, embed_model, index

df, documents, embed_model, index = load_rag_system()

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
        "query": query,
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
# UI (UPDATED DESIGN)
# -----------------------------

st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.header {
    text-align:center;
    padding:30px;
    border-radius:15px;
    background: linear-gradient(90deg,#4facfe,#00f2fe);
    color:white;
}
.card {
    padding:20px;
    border-radius:15px;
    background:white;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom:20px;
}
.stButton>button {
    width:100%;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="header">', unsafe_allow_html=True)
st.image("https://www.stillwater.you/images/logo.png", width=180)
st.markdown("## StillWater")
st.markdown("### Diabetes Testimonial Chatbot")
st.markdown("AI-powered insights from real patient stories")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# LANGUAGE
col_lang1, col_lang2 = st.columns(2)

with col_lang1:
    input_lang = st.selectbox("🌐 Input Language", ["English", "Hindi"])

with col_lang2:
    output_lang = st.selectbox("🔊 Output Voice Language", ["English", "Hindi"])

lang_map = {
    "English": {"stt": "en-IN", "tts": "en"},
    "Hindi": {"stt": "hi-IN", "tts": "hi"}
}

# SAMPLE QUESTIONS
preset_questions = [
    "Find testimonials where people reduced diabetes medicine after switching to plant-based diet",
    "Find patient stories that talk about plant-based diet helping diabetes",
    "Find testimonials where patients describe how long they had diabetes"
]

# LAYOUT
left, right = st.columns([1,2])

# LEFT PANEL
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎤 Voice Input")

    audio_bytes = audio_recorder(text="Click to record", pause_threshold=2.0)

    if audio_bytes:
        recognized_text = speech_to_text(audio_bytes, lang_code=lang_map[input_lang]["stt"])
        st.session_state["selected_query"] = recognized_text
        st.success(f"Recognized: {recognized_text}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💡 Sample Questions")

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

    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT PANEL
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    default_query = st.session_state.get("selected_query", "")
    query = st.text_input("💬 Ask your question", value=default_query)

    if st.button("🚀 Ask") and query.strip():
        with st.spinner("Thinking..."):
            result = ask_rag(query.strip(), top_k=3)

        st.markdown("### 🧠 Answer")
        st.markdown(result["answer"])

        st.markdown("---")
        st.markdown("### 📚 Sources")

        for i, src in enumerate(result["sources"], start=1):
            st.markdown(
                f"""**{i}. {src['title']}**  
🔗 {src['url']}  
⭐ Score: {round(src['score'], 4)}"""
            )

        st.markdown("---")
        st.markdown("### 🔊 Voice Output")

        audio_file = text_to_speech(result["answer"], lang=lang_map[output_lang]["tts"])

        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")

        if os.path.exists(audio_file):
            os.remove(audio_file)

    st.markdown("</div>", unsafe_allow_html=True)
