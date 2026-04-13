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

st.set_page_config(page_title="Diabetes Testimonial Chatbot", layout="wide", page_icon="🌿")

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
# ADVANCED UI DESIGN - WEBSITE MATCHING THEME
# -----------------------------
st.markdown("""
<style>
    /* Full page background with overlay */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
                    url('https://stillwater-main.onrender.com/images/c.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Main container styling */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        padding: 2rem;
        margin: 1rem;
    }
    
    /* Header styling */
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .header-logo {
        height: 70px;
        margin-bottom: 1rem;
        filter: drop-shadow(0 10px 20px rgba(0,0,0,0.2));
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 400;
    }
    
    /* Card styling */
    .chat-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        padding: 2.5rem;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .chat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
    }
    
    /* Language selectors */
    .lang-selector {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        text-align: center;
        color: white;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.4);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        height: 56px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Sample buttons */
    .sample-btn {
        width: 100%;
        height: 60px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .sample1 .sample-btn { 
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #c44569;
    }
    
    .sample2 .sample-btn { 
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #219653;
    }
    
    .sample3 .sample-btn { 
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #ed8936;
    }
    
    .sample-btn:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0,0,0,0.25);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 16px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        transform: scale(1.02);
    }
    
    /* Response styling */
    .response-answer {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 20px;
        padding: 2rem;
        border-left: 5px solid #667eea;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .source-item {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .source-item:hover {
        background: rgba(102, 126, 234, 0.08);
        transform: translateX(5px);
    }
    
    /* Audio recorder */
    .audio-recorder {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 5px solid #059669;
    }
    
    /* Metrics and spinners */
    .stSpinner > div {
        border-color: #667eea;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-container {
            margin: 0.5rem;
            padding: 1.5rem;
        }
        
        .header-title {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# MAIN CONTENT
# -----------------------------
# Header Section
st.markdown("""
<div class="header-section">
    <img src="https://www.stillwater.you/images/logo.png" class="header-logo" alt="Stillwater Logo">
    <div class="header-title">Diabetes Testimonial Chatbot</div>
    <div class="header-subtitle">🌿 AI-powered insights from real patient stories</div>
</div>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Language selectors
col_lang1, col_lang2 = st.columns(2)
with col_lang1:
    st.markdown('<div class="lang-selector">🌐 Input Language</div>', unsafe_allow_html=True)
    input_lang = st.selectbox("", ["English", "Hindi"], key="input_lang")

with col_lang2:
    st.markdown('<div class="lang-selector">🔊 Output Voice Language</div>', unsafe_allow_html=True)
    output_lang = st.selectbox("", ["English", "Hindi"], key="output_lang")

lang_map = {
    "English": {"stt": "en-IN", "tts": "en"},
    "Hindi": {"stt": "hi-IN", "tts": "hi"}
}

preset_questions = [
    "Find testimonials where people reduced diabetes medicine after switching to plant-based diet",
    "Find patient stories that talk about plant-based diet helping diabetes",
    "Find testimonials where patients describe how long they had diabetes"
]

# Layout
left, right = st.columns([1, 2])

# LEFT PANEL - Voice Input
with left:
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown("### 🎤 Voice Input")
    st.markdown('<div class="audio-recorder">', unsafe_allow_html=True)
    
    audio_bytes = audio_recorder(
        text="🎙️ Click & Speak", 
        pause_threshold=2.0,
        show_digits=True,
        key="recorder"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if audio_bytes:
        with st.spinner("🔄 Processing speech..."):
            recognized_text = speech_to_text(audio_bytes, lang_code=lang_map[input_lang]["stt"])
            st.session_state["selected_query"] = recognized_text
        st.success(f"✅ **Recognized:** {recognized_text}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT PANEL - Text Input & Controls
with right:
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    
    st.markdown("### 💡 Quick Questions")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown('<div class="sample1">', unsafe_allow_html=True)
        if st.button("💊 Reduce medicines", key="btn1", help=preset_questions[0]):
            st.session_state["selected_query"] = preset_questions[0]
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c2:
        st.markdown('<div class="sample2">', unsafe_allow_html=True)
        if st.button("🌱 Plant diet helps", key="btn2", help=preset_questions[1]):
            st.session_state["selected_query"] = preset_questions[1]
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c3:
        st.markdown('<div class="sample3">', unsafe_allow_html=True)
        if st.button("⏱️ Duration", key="btn3", help=preset_questions[2]):
            st.session_state["selected_query"] = preset_questions[2]
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    default_query = st.session_state.get("selected_query", "")
    query = st.text_input(
        "💬 Ask about patient testimonials...", 
        value=default_query, 
        placeholder="e.g., How did plant-based diet help diabetes patients?",
        key="query_input"
    )
    
    if st.button("🚀 Get Insights", type="primary"):
        if query.strip():
            with st.spinner("🤔 Analyzing testimonials..."):
                result = ask_rag(query.strip(), top_k=3)
            
            # Answer
            st.markdown('<div class="response-answer">', unsafe_allow_html=True)
            st.markdown(f"### 🧠 **AI Answer**")
            st.markdown(result["answer"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Sources
            st.markdown("### 📚 **Sources Found**")
            for i, src in enumerate(result["sources"], start=1):
                st.markdown(f"""
                <div class="source-item">
                    <h4><strong>{i}.</strong> {src['title']}</h4>
                    <a href="{src['url']}" target="_blank">🔗 View Testimonial</a> 
                    • ⭐ **Score:** {round(src['score']*100, 1)}%
                </div>
                """, unsafe_allow_html=True)
            
            # Voice Output
            st.markdown("### 🔊 **Listen to Answer**")
            audio_file = text_to_speech(result["answer"], lang=lang_map[output_lang]["tts"])
            
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/mp3")
            
            if os.path.exists(audio_file):
                os.remove(audio_file)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='
    text-align: center; 
    padding: 2rem; 
    color: rgba(255,255,255,0.8); 
    font-size: 0.9rem;
    background: rgba(0,0,0,0.1);
    border-radius: 20px;
    margin: 2rem 1rem 1rem 1rem;
'>
    🌿 Powered by Stillwater AI • Real stories, real insights
</div>
""", unsafe_allow_html=True)
