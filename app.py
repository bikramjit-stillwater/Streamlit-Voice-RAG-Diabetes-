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

if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""

def get_text(key):
    texts = {
        "english": {
            "title": "SHARAN Conversational AI",
            "voice_input": "Voice Input",
            "sample_questions": "Sample Questions",
            "reduce_medicines": "Reduce Medicines",
            "type2_reversed": "Type 2 Reversed",
            "diabetes_reversed": "Can Type 2 Be Reversed?",
            "ask_question": "Ask your question",
            "ask_btn": "Ask",
            "answer": "Answer",
            "sources": "Sources",
            "voice_output": "Voice Output",
            "recognized": "Recognized:",
            "processing": "Processing speech...",
            "thinking": "Thinking...",
            "score": "Score:",
            "tts_failed": "Voice output failed.",
            "stt_failed": "Speech recognition failed.",
            "no_answer": "Not found clearly in the testimonials."
        },
        "hindi": {
            "title": "शरण संवादात्मक कृत्रिम बुद्धिमत्ता",
            "voice_input": "वॉइस इनपुट",
            "sample_questions": "नमूना प्रश्न",
            "reduce_medicines": "दवा कम करें",
            "type2_reversed": "टाइप 2 कम हुआ",
            "diabetes_reversed": "क्या टाइप 2 उलटा हो सकता है?",
            "ask_question": "अपना प्रश्न पूछें",
            "ask_btn": "पूछें",
            "answer": "उत्तर",
            "sources": "स्रोत",
            "voice_output": "वॉइस आउटपुट",
            "recognized": "पहचाना गया:",
            "processing": "आवाज़ प्रोसेस हो रही है...",
            "thinking": "सोचा जा रहा है...",
            "score": "स्कोर:",
            "tts_failed": "वॉइस आउटपुट विफल रहा।",
            "stt_failed": "स्पीच पहचान विफल रही।",
            "no_answer": "यह जानकारी प्रशंसापत्रों में स्पष्ट रूप से नहीं मिली।"
        }
    }
    return texts[st.session_state.language.lower()][key]

lang_map = {
    "English": {"stt": "en-IN", "tts": "en", "query_hint": "en"},
    "Hindi": {"stt": "hi-IN", "tts": "hi", "query_hint": "hi"}
}

# -----------------------------
# Language toggle
# -----------------------------
col_empty, col_lang = st.columns([3, 1])
with col_lang:
    st.session_state.language = st.selectbox(
        "🌐",
        ["English", "Hindi"],
        index=0 if st.session_state.language == "English" else 1
    )

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
# Helpers
# -----------------------------
def detect_language_from_ui():
    return st.session_state.language

def translate_query_for_retrieval(query, target_language="English"):
    if target_language == "Hindi":
        prompt = f"""
Translate the following Hindi question into simple English for semantic retrieval.
Return only the translated English sentence.
Question: {query}
"""
        try:
            response = model.generate_content(prompt)
            translated = response.text.strip()
            return translated if translated else query
        except Exception:
            return query
    return query

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

def ask_rag(query, top_k=3, answer_language="English"):
    retrieval_query = translate_query_for_retrieval(query, target_language=answer_language)
    results = retrieve(retrieval_query, top_k=top_k)

    context = "\n\n".join([
        f"""SOURCE {i+1}
TITLE: {r['title']}
URL: {r['url']}
CONTENT:
{r['text']}"""
        for i, r in enumerate(results)
    ])

    if answer_language == "Hindi":
        language_instruction = f"""
Answer only in Hindi.
Use simple, natural Hindi.
Do not answer in English.
If title or URL is in English, keep them unchanged.
If the answer is not clearly present, say exactly:
"{get_text('no_answer')}"
"""
    else:
        language_instruction = f"""
Answer only in English.
If the answer is not clearly present, say exactly:
"{get_text('no_answer')}"
"""

    prompt = f"""
You are a testimonial-based assistant.

Rules:
1. Answer only from the provided testimonial context.
2. If the answer is not clearly present, respond with the exact fallback sentence.
3. Do not give medical advice.
4. Mention relevant source title and URL in the answer.
5. Keep the answer clear and short.
6. {language_instruction}

User question:
{query}

Context:
{context}
"""

    try:
        response = model.generate_content(prompt)
        answer_text = response.text.strip() if response.text else get_text("no_answer")
    except Exception:
        answer_text = get_text("no_answer")

    return {
        "answer": answer_text,
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
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language=lang_code)
        return text

    except sr.UnknownValueError:
        return "Could not understand audio." if lang_code == "en-IN" else "ऑडियो समझ में नहीं आया।"
    except sr.RequestError as e:
        return f"Speech service error: {e}" if lang_code == "en-IN" else f"स्पीच सेवा त्रुटि: {e}"
    except Exception as e:
        return f"Speech recognition failed: {str(e)}" if lang_code == "en-IN" else f"स्पीच पहचान विफल रही: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def text_to_speech(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception:
        return None

# -----------------------------
# Theme / UI
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(15, 18, 22, 0.48), rgba(15, 18, 22, 0.48)),
                    url('https://stillwater-main.onrender.com/images/c.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 1rem;
        max-width: 1220px;
    }

    .hero-wrap {
        background: rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 34px rgba(0, 0, 0, 0.18);
        border-radius: 24px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }

    .hero-logo {
        height: 48px;
        margin: 0 auto 0.3rem auto;
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
        padding: 1rem;
        color: white;
        margin-top: 0.45rem;
    }

    .section-title {
        color: #ffffff;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        line-height: 1.2;
    }

    .stSelectbox label,
    .stTextInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 16px !important;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.88) !important;
        border: 1px solid rgba(255, 255, 255, 0.20) !important;
        color: #1b1b1b !important;
        min-height: 46px;
        padding-left: 0.9rem !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #6b7280;
    }

    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.88) !important;
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

    .mic-wrap {
        margin-top: 1.1rem;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    div[data-testid="stAudioRecorder"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    div[data-testid="stAudioRecorder"] button {
        width: 58px !important;
        height: 58px !important;
        min-height: 58px !important;
        border-radius: 999px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        background: rgba(255, 255, 255, 0.12) !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: auto !important;
    }

    div[data-testid="stAudioRecorder"] button:hover {
        background: rgba(255,255,255,0.20) !important;
        transform: scale(1.03);
    }

    div[data-testid="stAudioRecorder"] p,
    div[data-testid="stAudioRecorder"] span {
        display: none !important;
    }

    @media (max-width: 768px) {
        .hero-wrap {
            padding: 0.9rem 0.9rem 1rem 0.9rem;
            border-radius: 20px;
        }

        .hero-logo {
            height: 42px;
            margin-bottom: 0.25rem;
        }

        .hero-title {
            font-size: 1.45rem;
        }

        .panel-card {
            padding: 0.9rem;
        }

        div[data-testid="stAudioRecorder"] button {
            width: 54px !important;
            height: 54px !important;
            min-height: 54px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(f"""
<div class="hero-wrap">
    <img src="https://www.stillwater.you/images/logo.png" class="hero-logo" alt="StillWater Logo">
    <div class="hero-title">{get_text("title")}</div>
</div>
""", unsafe_allow_html=True)

preset_questions_map = {
    "English": [
        "Find testimonials where people reduced diabetes medicine after switching to plant-based diet",
        "Give me testimonial of patient who reduced type 2 diabetes",
        "Can type 2 diabetes be reversed?"
    ],
    "Hindi": [
        "ऐसे प्रशंसापत्र खोजें जहाँ लोगों ने प्लांट-बेस्ड डाइट अपनाने के बाद डायबिटीज़ की दवाइयाँ कम कीं",
        "ऐसे मरीज का प्रशंसापत्र बताइए जिसने टाइप 2 डायबिटीज़ कम की",
        "क्या टाइप 2 डायबिटीज़ उलटी हो सकती है?"
    ]
}

preset_questions = preset_questions_map[st.session_state.language]

# -----------------------------
# Main panel
# -----------------------------
st.markdown('<div class="panel-card">', unsafe_allow_html=True)

st.markdown(f'<div class="section-title">💡 {get_text("sample_questions")}</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    if st.button(get_text("reduce_medicines")):
        st.session_state.selected_query = preset_questions[0]

with c2:
    if st.button(get_text("type2_reversed")):
        st.session_state.selected_query = preset_questions[1]

with c3:
    if st.button(get_text("diabetes_reversed")):
        st.session_state.selected_query = preset_questions[2]

st.markdown("<hr>", unsafe_allow_html=True)

query = st.text_input(
    f"💬 {get_text('ask_question')}",
    value=st.session_state.selected_query
)

# -----------------------------
# Voice input
# -----------------------------
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
        active_lang = detect_language_from_ui()
        recognized_text = speech_to_text(
            audio_bytes,
            lang_code=lang_map[active_lang]["stt"]
        )
        st.session_state.selected_query = recognized_text
        query = recognized_text

    st.success(f"{get_text('recognized')} {recognized_text}")

# -----------------------------
# Ask button
# -----------------------------
if st.button(f"🚀 {get_text('ask_btn')}", type="primary"):
    if query.strip():
        st.session_state.selected_query = query.strip()

        with st.spinner(get_text("thinking")):
            result = ask_rag(
                query=query.strip(),
                top_k=3,
                answer_language=st.session_state.language
            )

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
            lang=lang_map[st.session_state.language]["tts"]
        )

        if audio_file and os.path.exists(audio_file):
            with open(audio_file, "rb") as f:
                out_audio_bytes = f.read()
                st.markdown('<div class="audio-box">', unsafe_allow_html=True)
                st.audio(out_audio_bytes, format="audio/mp3")
                st.markdown('</div>', unsafe_allow_html=True)
            os.remove(audio_file)
        else:
            st.error(get_text("tts_failed"))

st.markdown('</div>', unsafe_allow_html=True)
