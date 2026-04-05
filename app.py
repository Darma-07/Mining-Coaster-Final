# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install semua library yang dibutuhkan              ║
# ╚══════════════════════════════════════════════════════════════╝

!pip install streamlit sentence-transformers faiss-cpu pyngrok huggingface_hub plotly pandas numpy -q
print("✅ Semua library berhasil diinstall!")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Mount Google Drive                                 ║
# ╚══════════════════════════════════════════════════════════════╝

from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = '/content/drive/MyDrive/it-career-chatbot-final'
os.makedirs(f'{PROJECT_DIR}/data', exist_ok=True)
os.makedirs(f'{PROJECT_DIR}/models', exist_ok=True)
print(f'✅ Folder siap di Google Drive!')
print(f'📁 {PROJECT_DIR}')


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Upload data CSV                                    ║
# ╚══════════════════════════════════════════════════════════════╝

from google.colab import files
import shutil, os

PROJECT_DIR = '/content/drive/MyDrive/it-career-chatbot-final'

print('📤 Upload file: career_recommendation_final.csv')
uploaded = files.upload()

for filename in uploaded.keys():
    dest = f'{PROJECT_DIR}/data/{filename}' if filename.endswith('.csv') else f'{PROJECT_DIR}/{filename}'
    shutil.move(filename, dest)
    print(f'✅ {filename} → tersimpan!')

print('\n📁 File di folder data:')
for f in os.listdir(f'{PROJECT_DIR}/data'):
    print(f'   ✓ {f}')


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Preprocessing Data                                 ║
# ╚══════════════════════════════════════════════════════════════╝

import pandas as pd
import numpy as np
import re
import pickle
import os

PROJECT_DIR = '/content/drive/MyDrive/it-career-chatbot-final'
DATA_DIR    = f'{PROJECT_DIR}/data'
MODEL_DIR   = f'{PROJECT_DIR}/models'

def clean_text(text):
    if not isinstance(text, str): return ''
    return re.sub(r'\s+', ' ', text.strip()).lower()

# Load dataset baru
df = pd.read_csv(f'{DATA_DIR}/career_recommendation_final.csv')

# Rename kolom
df.columns = ['name','gender','degree','major','interests',
              'skills','cgpa','has_cert','cert_title','is_working',
              'job_title','masters']

# Bersihkan kolom penting
for col in ['interests','skills','major','job_title']:
    df[col] = df[col].apply(clean_text)

# Ambil daftar jurusan & job unik langsung dari dataset (dinamis)
ALL_MAJORS = sorted(df['major'].dropna().unique().tolist())
ALL_JOBS   = sorted(
    df['job_title'].dropna()[df['job_title'].dropna() != 'na'].unique().tolist()
)

# Simpan ke pickle
all_data = {
    'career'    : df,
    'all_majors': ALL_MAJORS,
    'all_jobs'  : ALL_JOBS,
}
with open(f'{MODEL_DIR}/all_data.pkl', 'wb') as f:
    pickle.dump(all_data, f)

print('✅ PREPROCESSING SELESAI!')
print(f'   📊 Total data         : {len(df):,} entri')
print(f'   🎓 Jurusan unik       : {len(ALL_MAJORS)}')
print(f'   💼 Job title unik     : {len(ALL_JOBS)}')
print()
print('Jurusan yang terdaftar:')
for m in ALL_MAJORS:
    print(f'  - {m}')


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Build FAISS Index                                  ║
# ╚══════════════════════════════════════════════════════════════╝

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

PROJECT_DIR = '/content/drive/MyDrive/it-career-chatbot-final'
MODEL_DIR   = f'{PROJECT_DIR}/models'

with open(f'{MODEL_DIR}/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

df = all_data['career']

print('⚙️  Loading embedding model...')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ Model loaded!')

print('⚙️  Building FAISS index...')
texts      = (df['interests'] + ' ' + df['skills']).tolist()
embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=64)
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, f'{MODEL_DIR}/faiss_index.index')
embed_model.save(f'{MODEL_DIR}/embed_model')

print(f'✅ FAISS index selesai!')
print(f'   🧠 Total vectors : {index.ntotal:,}')
print(f'   💾 Tersimpan di  : {MODEL_DIR}')


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Tulis app.py ke Google Drive                       ║
# ╚══════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Buat & Download requirements.txt                   ║
# ╚══════════════════════════════════════════════════════════════╝

from google.colab import files

PROJECT_DIR = '/content/drive/MyDrive/it-career-chatbot-final'

req = """streamlit
sentence-transformers
faiss-cpu
huggingface_hub
pandas
numpy
plotly"""

req_path = f'{PROJECT_DIR}/requirements.txt'

with open(req_path, 'w') as f:
    f.write(req)

print('✅ requirements.txt siap!')
files.download(req_path)
print('⬇️  Download dimulai...')


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Jalankan Streamlit + Ngrok                         ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import subprocess
import time
from pyngrok import ngrok

# Matikan dulu yang lama
subprocess.run(['pkill', '-f', 'streamlit'], capture_output=True)
time.sleep(2)

HF_TOKEN    = '...'   # ← isi token Hugging Face kamu
NGROK_TOKEN = '...'   # ← isi token Ngrok kamu

os.environ['HF_TOKEN'] = HF_TOKEN
PROJECT_DIR = '/content/drive/MyDrive/it-career-chatbot-final'

ngrok.set_auth_token(NGROK_TOKEN)

subprocess.Popen([
    'streamlit', 'run', f'{PROJECT_DIR}/app.py',
    '--server.port', '8501',
    '--server.headless', 'true'
])

time.sleep(5)

public_url = ngrok.connect(8501)
print('=' * 50)
print('✅ Career Guide BERHASIL DIJALANKAN!')
print(f'🔗 Buka link ini: {public_url.public_url}')
print('=' * 50)

import os

PROJECT_DIR = '/content/drive/MyDrive/it-career-chatbot-final'

app_code = r'''
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
    background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
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


# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_data():
    base      = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, 'models')
    with open(os.path.join(model_dir, 'all_data.pkl'), 'rb') as f:
        return pickle.load(f)

try:
    data       = load_all_data()
    df         = data['career']
    ALL_MAJORS = data['all_majors']
    ALL_JOBS   = data['all_jobs']
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

HF_TOKEN = os.environ.get('HF_TOKEN', '')


# ─── Bangun SYSTEM PROMPT dinamis dari dataset ────────────────────────────────
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
        df['major'].str.contains(q, na=False)
    ]
    if len(hasil) == 0:
        return None, None
    job_counts   = hasil['job_title'].value_counts().head(5)
    major_counts = hasil['major'].value_counts().head(5)
    return job_counts, major_counts


# ─── Quick Prompt Suggestions ─────────────────────────────────────────────────
QUICK_PROMPTS = [
    ("💻", "#00ff88", "Jurusan apa yang cocok untuk saya?"),
    ("📊", "#58a6ff", "Prospek kerja Data Scientist"),
    ("🎨", "#f78166", "Karir dari jurusan Desain UI/UX"),
    ("🌾", "#ffa657", "Peluang kerja lulusan Agribisnis"),
]

if 'messages' not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    buttons_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:1.5rem;'>"
    for icon, color, label in QUICK_PROMPTS:
        buttons_html += (
            f"<div style='background:#161b22;border:1px solid {color};padding:8px 16px;"
            f"border-radius:6px;font-size:0.82rem;color:{color};'>{icon} {label}</div>"
        )
    buttons_html += "</div>"
    st.markdown(buttons_html, unsafe_allow_html=True)


# ─── Chat History ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'], unsafe_allow_html=True)


# ─── Input & Response ─────────────────────────────────────────────────────────
if prompt := st.chat_input("Tanya soal jurusan, karir, atau prospek kerja..."):
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


# ─── Reset Button ─────────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.rerun()
'''

with open(f'{PROJECT_DIR}/app.py', 'w', encoding='utf-8') as f:
    f.write(app_code.lstrip())

print('✅ app.py berhasil dibuat!')
print(f'📁 Lokasi: {PROJECT_DIR}/app.py')
