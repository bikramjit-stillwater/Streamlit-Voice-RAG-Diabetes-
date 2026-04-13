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
    .stApp > div > div > div {
        padding-top: 3rem !important;
    }

    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 1rem;
        max-width: 1220px;
    }

    .stApp {
        background: linear-gradient(rgba(15, 18, 22, 0.48), rgba(15, 18, 22, 0.48)),
                    url('https://stillwater-main.onrender.com/images/c.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .hero-wrap {
        background: rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 34px rgba(0, 0, 0, 0.18);
        border-radius: 24px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        margin-bottom: 1.2rem;
        margin-top: 1rem;
    }

    .hero-logo {
        height: 48px;
        margin: 0 auto 0.5rem auto;
        display: block;
        object-fit: contain;
    }

    .hero-title {
        color: #f7f4ee;
        font-size: 1.9rem;
        line-height: 1.15;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .panel-card {
        background: rgba(255, 255, 255, 0.11);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.10);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.14);
        border-radius: 22px;
        padding: 1.5rem;
        color: white;
        margin-top: 1rem;
    }

    .section-title {
        color: #ffffff;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        line-height: 1.2;
    }

    /* Default language style */
    .lang-checkbox {
        color: white !important;
    }

    .lang-checkbox .stRadio > label {
        font-size: 0.9rem !important;
        padding: 0.4rem 0.8rem !important;
        margin: 0.1rem !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.20) !important;
        border: 1px solid rgba(255, 255, 255, 0.30) !important;
        font-weight: 600 !important;
        min-height: auto !important;
        white-space: nowrap;
    }

    .lang-checkbox .stRadio > label:hover {
        background: rgba(255, 255, 255, 0.35) !important;
    }

    .stTextInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    .stTextInput > div > div > input {
        border-radius: 16px !important;
        background: rgba(255, 255, 255, 0.88) !important;
        border: 1px solid rgba(255, 255, 255, 0.20) !important;
        color: #1b1b1b !important;
        min-height: 46px;
        padding-left: 0.9rem !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #6b7280;
    }

    .stButton > button {
        width: 100%;
        min-height: 44px;
        border: none;
        border-radius: 16px;
        font-weight: 700;
        transition: all 0.25s ease;
        box-shadow: 0 8px 22px rgba(0,0,0,0.10);
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

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #d6b36a, #b38a3d);
        color: white;
    }

    .answer-box {
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        color: white;
        line-height: 1.55;
        margin-top: 0.45rem;
    }

    .sources-box {
        background: rgba(255, 255, 255, 0.09);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.7rem;
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
        padding: 0.7rem;
        margin-top: 0.35rem;
    }

    .stAudio {
        border-radius: 14px;
        overflow: hidden;
    }

    hr {
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.14);
        margin: 0.9rem 0;
    }

    /* Mic wrapper */
    .mic-wrap {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-top: 16px !important;
        padding: 8px 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Remove audio recorder surrounding box */
    div[data-testid="stAudioRecorder"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    div[data-testid="stAudioRecorder"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    div[data-testid="stAudioRecorder"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: auto !important;
        width: auto !important;
        height: auto !important;
        outline: none !important;
    }

    div[data-testid="stAudioRecorder"] button:hover,
    div[data-testid="stAudioRecorder"] button:focus,
    div[data-testid="stAudioRecorder"] button:active {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        transform: none !important;
    }

    /* Hide text labels inside recorder */
    div[data-testid="stAudioRecorder"] p,
    div[data-testid="stAudioRecorder"] span,
    div[data-testid="stAudioRecorder"] label,
    div[data-testid="stAudioRecorder"] small {
        display: none !important;
    }

    /* Default mic size */
    div[data-testid="stAudioRecorder"] svg,
    div[data-testid="stAudioRecorder"] path {
        width: 34px !important;
        height: 34px !important;
    }

    iframe {
        background: transparent !important;
    }

    /* Light theme */
    @media (prefers-color-scheme: light) {
        div[data-testid="stAudioRecorder"] svg,
        div[data-testid="stAudioRecorder"] path,
        div[data-testid="stAudioRecorder"] [data-testid*="icon"] {
            fill: #000000 !important;
            stroke: #000000 !important;
            color: #000000 !important;
        }

        .lang-checkbox .stRadio > label,
        .lang-checkbox .stRadio > label > div,
        .lang-checkbox .stRadio > label > span {
            color: #FFFFFF !important;
        }
    }

    /* Dark theme */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stAudioRecorder"] svg,
        div[data-testid="stAudioRecorder"] path,
        div[data-testid="stAudioRecorder"] [data-testid*="icon"] {
            fill: #ffffff !important;
            stroke: #ffffff !important;
            color: #ffffff !important;
        }

        .lang-checkbox .stRadio > label,
        .lang-checkbox .stRadio > label > div,
        .lang-checkbox .stRadio > label > span {
            color: #FFFFFF !important;
        }
    }

    @media (max-width: 768px) {
        .hero-wrap {
            padding: 1.2rem 1rem;
            border-radius: 20px;
            margin-top: 1.5rem;
        }

        .hero-logo {
            height: 42px;
        }

        .hero-title {
            font-size: 1.45rem;
        }

        .panel-card {
            padding: 1.2rem;
            margin-top: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero-wrap">
    <img src="https://www.stillwater.you/images/logo.png" class="hero-logo" alt="StillWater Logo">
    <div class="hero-title">""" + get_text("title") + """</div>
</div>
""", unsafe_allow_html=True)

preset_questions = [
    "Find testimonials where people reduced diabetes medicine after switching to plant-based diet",
    "Give me testimonial of patient who have reduce type 2 diabetes",
    "can type 2 diabetes be reversed"
]

# -----------------------------
# MAIN PANEL
# -----------------------------
st.markdown('<div class="panel-card">', unsafe_allow_html=True)

st.markdown('<div style="padding: 1rem 0;">', unsafe_allow_html=True)
col1, _ = st.columns([1, 4])
with col1:
    st.markdown('<div class="lang-checkbox">', unsafe_allow_html=True)
    lang_option = st.radio(
        "🌐",
        ["English", "Hindi"],
        index=0 if st.session_state.language == "English" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state.language = lang_option
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="section-title">💡 {get_text("sample_questions")}</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    if st.button(get_text("reduce_medicines")):
        st.session_state["selected_query"] = preset_questions[0]

with c2:
    if st.button(get_text("type2_reversed")):
        st.session_state["selected_query"] = preset_questions[1]

with c3:
    if st.button(get_text("diabetes_reversed")):
        st.session_state["selected_query"] = preset_questions[2]

st.markdown("<hr>", unsafe_allow_html=True)

default_query = st.session_state.get("selected_query", "")
query = st.text_input(
    f"💬 {get_text('ask_question')}",
    value=default_query
)

# Microphone - only icon style
st.markdown('<div class="mic-wrap">', unsafe_allow_html=True)
audio_bytes = audio_recorder(
    text="",
    recording_color="#000000",
    neutral_color="#000000",
    icon_name="microphone",
    icon_size="2x",
    pause_threshold=2.0
)
st.markdown('</div>', unsafe_allow_html=True)

if audio_bytes:
    with st.spinner(get_text("processing")):
        input_lang = "English" if st.session_state.language == "English" else "Hindi"
        recognized_text = speech_to_text(
            audio_bytes,
            lang_code=lang_map[input_lang]["stt"]
        )
        st.session_state["selected_query"] = recognized_text
        query = recognized_text
    st.success(f"{get_text('recognized')} {recognized_text}")

input_lang = "English" if st.session_state.language == "English" else "Hindi"
output_lang = input_lang

if st.button(f"🚀 {get_text('ask_btn')}", type="primary"):
    if query.strip():
        with st.spinner(get_text("thinking")):
            result = ask_rag(query.strip(), top_k=3)

        st.markdown(f'<div class="section-title">{get_text("answer")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{get_text("sources")}</div>', unsafe_allow_html=True)

        for i, src in enumerate(result["sources"], start=1):
            st.markdown(
                f"""
                <div class="sources-box">
                    <strong>{i}. {src['title']}</strong><br>
                    🔗 <a href="{src['url']}" target="_blank">{src['url']}</a><br>
                    ⭐ {get_text('score')} {round(src['score'], 4)}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{get_text("voice_output")}</div>', unsafe_allow_html=True)

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
