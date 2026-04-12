import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Career Guide", page_icon="🚀", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, .stApp {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
#MainMenu, footer, header {visibility: hidden;}

[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 2px solid #00ff88 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stButton button {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    border-color: #00ff88 !important;
    color: #00ff88 !important;
}
div[data-testid="column"]:nth-child(1) .stButton button { border-color: #00ff88; color: #00ff88; }
div[data-testid="column"]:nth-child(2) .stButton button { border-color: #58a6ff; color: #58a6ff; }
div[data-testid="column"]:nth-child(3) .stButton button { border-color: #f78166; color: #f78166; }
div[data-testid="column"]:nth-child(4) .stButton button { border-color: #ffa657; color: #ffa657; }
button[kind="secondary"] {
    background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
    color: #0d1117 !important;
    border: none !important;
    font-weight: 700 !important;
}
[data-testid="stChatMessage"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 14px !important;
    margin: 6px 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;'>
  <div style='font-family:JetBrains Mono;font-size:2rem;font-weight:700;color:#00ff88;
              letter-spacing:2px;'>
    🚀 Career Guide
  </div>
  <div style='color:#8b949e;font-size:0.9rem;margin-top:8px;font-family:Space Grotesk;'>
    Temukan jurusan yang cocok · Prospek kerja · Panduan karir
  </div>
  <div style='margin-top:12px;display:inline-block;background:#161b22;border:1px solid #00ff88;
              padding:4px 14px;border-radius:20px;font-size:0.75rem;color:#00ff88;'>
    ● ONLINE
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Auto-build model jika belum ada ─────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'career_recommendation_final.csv')
PKL_PATH  = os.path.join(MODEL_DIR, 'all_data.pkl')

def clean_text(text):
    if not isinstance(text, str): return ''
    return re.sub(r'\s+', ' ', text.strip()).lower()

def build_model():
    """Preprocessing ringan untuk deploy — konsisten dengan preprocessing baru."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # Rename kolom panjang → singkat (sama dengan preprocessing Colab)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "What is your name?"                                                                                  : "name",
        "What is your gender?"                                                                                : "gender",
        "What was your course in UG?"                                                                         : "ug_course",
        "What is your UG specialization? Major Subject (Eg; Mathematics)"                                    : "ug_major",
        "What are your interests?"                                                                            : "interests",
        "What are your skills ? (Select multiple if necessary)"                                               : "skills",
        "What was the average CGPA or Percentage obtained in under graduation?"                               : "cgpa",
        "Did you do any certification courses additionally?"                                                  : "certification",
        "If yes, please specify your certificate course title."                                               : "cert_title",
        "Are you working?"                                                                                    : "working",
        "If yes, then what is/was your first Job title in your current field of work? If not applicable, write NA." : "job_title",
        "Have you done masters after undergraduation? If yes, mention your field of masters.(Eg; Masters in Mathematics)" : "masters",
    })

    # Fallback: jika kolom sudah pendek
    if len(df.columns) == 12 and "ug_course" not in df.columns:
        df.columns = ['name','gender','ug_course','ug_major','interests',
                      'skills','cgpa','certification','cert_title','working',
                      'job_title','masters']

    # Bersihkan kolom penting
    for col in ['interests','skills','ug_major','job_title']:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    ALL_MAJORS = sorted(df['ug_major'].dropna().unique().tolist())
    ALL_JOBS   = sorted(
        df['job_title'].dropna()[df['job_title'].dropna() != 'na'].unique().tolist()
    )

    all_data = {'career': df, 'all_majors': ALL_MAJORS, 'all_jobs': ALL_JOBS}
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(all_data, f)
    return all_data


# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_data():
    if not os.path.exists(PKL_PATH):
        st.info("⚙️ Mempersiapkan data untuk pertama kali... Tunggu sebentar ya!")
        return build_model()
    with open(PKL_PATH, 'rb') as f:
        return pickle.load(f)

try:
    data       = load_all_data()
    df         = data['career']
    ALL_MAJORS = data['all_majors']
    ALL_JOBS   = data['all_jobs']
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Deteksi nama kolom major (kompatibel dengan pickle lama & baru)
MAJOR_COL = 'ug_major' if 'ug_major' in df.columns else 'major'

HF_TOKEN = os.environ.get('HF_TOKEN', '')


# ─── System Prompt dinamis dari dataset ───────────────────────────────────────
def build_system_prompt():
    majors_str = '\n'.join(f'  - {m}' for m in ALL_MAJORS)
    jobs_str   = '\n'.join(f'  - {j}' for j in ALL_JOBS)
    return (
        "Kamu adalah Career Guide — asisten karir cerdas dan friendly untuk pelajar Indonesia.\n\n"
        "PENTING — BATASAN TOPIK:\n"
        "Kamu HANYA boleh menjawab pertanyaan yang berkaitan dengan jurusan dan karir\n"
        "yang ada dalam daftar berikut ini. Jika pertanyaan di luar daftar ini, tolak\n"
        "dengan sopan dan minta pengguna bertanya seputar topik yang tersedia.\n\n"
        "=== DAFTAR JURUSAN YANG KAMU KUASAI ===\n"
        f"{majors_str}\n\n"
        "=== DAFTAR KARIR / PEKERJAAN YANG KAMU KUASAI ===\n"
        f"{jobs_str}\n\n"
        "ATURAN MENJAWAB:\n"
        "1. Hanya jawab pertanyaan seputar jurusan dan karir di daftar atas.\n"
        "2. Jika ditanya topik lain (resep masakan, berita politik, olahraga, dsb.),\n"
        "   tolak dengan sopan: \"Maaf, aku hanya bisa bantu seputar jurusan dan karir\n"
        "   yang ada di database ya! 😊 Coba tanya tentang salah satu jurusan atau karir di atas.\"\n"
        "3. Jika pengguna menyebut jurusan/karir yang TIDAK ada di daftar, informasikan\n"
        "   bahwa data tersebut belum tersedia, lalu tawarkan alternatif yang mirip.\n"
        "4. Saat merekomendasikan, sebutkan jurusan yang relevan, prospek kerjanya,\n"
        "   skill yang dibutuhkan, dan tips konkret.\n"
        "5. Gunakan Bahasa Indonesia yang friendly, relatable, dan encouraging.\n"
        "   Boleh campurkan istilah teknis dalam bahasa Inggris jika memang lazim dipakai.\n"
        "6. Gunakan emoji secukupnya agar lebih hidup.\n"
    )

SYSTEM_PROMPT = build_system_prompt()


# ─── HuggingFace LLM ──────────────────────────────────────────────────────────
def ask_hf(messages):
    models_to_try = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2.5-72B-Instruct",
        "google/gemma-2-2b-it",
        "HuggingFaceH4/zephyr-7b-beta",
    ]
    client = InferenceClient(token=HF_TOKEN)
    for model in models_to_try:
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=700,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception:
            continue
    return "Maaf, AI sedang tidak tersedia. Coba lagi dalam beberapa menit ya!"


# ─── Rekomendasi dari dataset ─────────────────────────────────────────────────
def cari_rekomendasi(query):
    q = query.lower()
    hasil = df[
        df['interests'].str.contains(q, na=False) |
        df['skills'].str.contains(q, na=False)    |
        df[MAJOR_COL].str.contains(q, na=False)
    ]
    if len(hasil) == 0:
        return None, None
    job_counts   = hasil['job_title'].value_counts().head(5)
    major_counts = hasil[MAJOR_COL].value_counts().head(5)
    return job_counts, major_counts


# ─── Session State ────────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'quick_prompt' not in st.session_state:
    st.session_state.quick_prompt = None


# ─── Quick Prompt Buttons (hanya tampil kalau belum ada chat) ─────────────────
QUICK_PROMPTS = [
    ("💻", "Jurusan apa yang cocok untuk saya?"),
    ("📊", "Prospek kerja Data Scientist"),
    ("🎨", "Karir dari jurusan Desain UI/UX"),
    ("🌾", "Peluang kerja lulusan Agribisnis"),
]

if len(st.session_state.messages) == 0:
    cols = st.columns(2)
    for i, (icon, label) in enumerate(QUICK_PROMPTS):
        with cols[i % 2]:
            if st.button(f"{icon} {label}", key=f"qp_{i}", use_container_width=True):
                st.session_state.quick_prompt = label
                st.rerun()


# ─── Chat History ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'], unsafe_allow_html=True)


# ─── Input & Response ─────────────────────────────────────────────────────────
if st.session_state.quick_prompt:
    prompt = st.session_state.quick_prompt
    st.session_state.quick_prompt = None
else:
    prompt = st.chat_input("Tanya soal jurusan, karir, atau prospek kerja...")

if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner("Mencari info..."):

            job_recs, major_recs = cari_rekomendasi(prompt)

            context = "\n"
            if job_recs is not None:
                context += "[DATA DARI DATABASE]\n"
                context += f"Karir relevan dari data: {job_recs.to_dict()}\n"
                context += f"Jurusan relevan dari data: {major_recs.to_dict()}\n"
                context += "Gunakan data di atas sebagai referensi utama dalam menjawab.\n"

            ai_messages = [{"role": "system", "content": SYSTEM_PROMPT + context}]
            for m in st.session_state.messages:
                ai_messages.append({"role": m['role'], "content": m['content']})

            full_reply = ask_hf(ai_messages)
            st.markdown(full_reply, unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': full_reply})

    st.rerun()


# ─── Reset Button ─────────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Reset Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.quick_prompt = None
        st.rerun()
