import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from PIL import Image, ImageEnhance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras.backend as K
import re
import io, time
from datetime import datetime

# === Hugging Face download config (ADD) ===
import os
from huggingface_hub import hf_hub_download

HF_REPO_ID = "Rifdah/pneumonia-cnn"   # <-- GANTI jika nama repo kamu berbeda
HF_FILENAME = "cnn_model.h5"
LOCAL_MODEL_PATH = "cnn_model.h5"

# Jika repo HF Private, set token via Secrets/ENV: HF_TOKEN=hf_xxx
HF_TOKEN = os.getenv("HF_TOKEN", None)

def ensure_cnn_model_local():
    """Unduh cnn_model.h5 dari Hugging Face kalau belum ada di folder kerja."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        with st.spinner("📥 Mengunduh model CNN..."):
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                local_dir=".",                 # simpan di root project
                local_dir_use_symlinks=False,  # pastikan file fisik dibuat
                token=HF_TOKEN                 # biarkan None jika repo public
            )

# ================== Konfigurasi Halaman ==================
st.set_page_config(page_title="Website Deteksi Pneumonia", layout="wide")

# ================== THEME & GLOBAL STYLES (UI ONLY) ==================
st.markdown("""
<style>
:root{
  --bg-top:#16263c;
  --bg-bottom:#0f1e31;
  --card:#15263b;
  --ink:#f3f6fb;
  --ink-dim:#cfdaea;
  --brand:#3b82f6;
  --brand-2:#2563eb;
  --radius:14px;
  --radius-sm:10px;
  --shadow:0 8px 30px rgba(0,0,0,.28);
  --chip:#203a5c;
  --credit-bg:rgba(10,16,28,.75);
  --credit-ink:#eaf2ff;
}

/* App background */
html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 800px at 15% -10%, rgba(60,112,180,.25) 0%, transparent 60%),
    linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%) !important;
  color: var(--ink) !important;
}

/* Base font */
* { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif; }

.block-container{ padding-top: 1.2rem; padding-bottom: 2.2rem; }

/* Sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #13233a 0%, #0e1f37 100%) !important;
  border-right: 1px solid rgba(255,255,255,.10);
  box-shadow: 8px 0 30px rgba(0,0,0,.25);
  padding:12px !important;
}
.sidebar-title{
  text-align:center;
  font-size:26px !important;
  font-weight:800 !important;
  color:#e9f2ff !important;
  letter-spacing:.3px;
  margin: 8px 0 10px;
}

/* Sidebar nav hover */
[data-testid="stSidebarNav"] ul li a{
  display:block; color:#f0f6ff !important; font-size:18px !important; font-weight:700 !important;
  padding:10px 12px; border-radius:10px; transition:all .18s ease; text-decoration:none !important;
  border:1px solid rgba(255,255,255,.08); background:rgba(255,255,255,.05);
}
[data-testid="stSidebarNav"] ul li a:hover{
  transform: translateX(2px);
  background: rgba(59,130,246,.15) !important;
  color:#ffea8a !important;
  border-color: rgba(59,130,246,.40);
}

/* Buttons */
.stButton>button{
  background: linear-gradient(180deg, var(--brand), var(--brand-2)) ! important;
  color:white !important; font-weight:800 !important; letter-spacing:.2px;
  padding:.64rem 1rem !important; border-radius:12px !important; border:0 !important;
  box-shadow: 0 10px 22px rgba(37,99,235,.28);
  transition: transform .06s ease, filter .18s ease, box-shadow .18s ease;
}
.stButton>button:hover{ filter:brightness(1.07); box-shadow: 0 12px 26px rgba(37,99,235,.34); }
.stButton>button:active{ transform: translateY(1px); }

/* Headings & paragraph */
h1{ font-size: 40px !important; font-weight: 800 !important; text-align: left; margin: 0 0 12px; color:#f1f6ff !important; }
h2{ font-size: 28px !important; font-weight: 800 !important; color:#eaf2ff !important;}
h3{ font-size: 22px !important; font-weight: 800 !important; color:#eaf2ff !important;}
p{ font-size: 18px !important; line-height: 1.68; color: var(--ink-dim) !important; }
strong, b{ color:#ffffff !important; }
a{ color:#9cd0ff !important; text-decoration: underline; text-underline-offset: 2px; }

/* Hero generic */
.hero{
  border-radius: var(--radius);
  padding: 18px 20px;
  background: linear-gradient(135deg, rgba(37,99,235,.22), rgba(6,182,212,.16));
  border: 1px solid rgba(150,185,235,.40);
  box-shadow: var(--shadow);
  margin-bottom: 16px;
}
.hero h1{ margin: 0 0 6px; }
.hero p{ margin: 0; color: var(--ink-dim); }

/* Landing Hero */
.hero-landing{
  border-radius: 18px;
  padding: 26px 28px;
  background:
    radial-gradient(900px 500px at 100% 0%, rgba(59,130,246,.25) 0%, transparent 60%),
    linear-gradient(135deg, rgba(18,35,60,.85), rgba(10,24,45,.9));
  border: 1px solid rgba(150,185,235,.35);
  box-shadow: 0 16px 48px rgba(0,0,0,.35);
}
.hero-landing h1{ font-size: 46px !important; margin: 0 0 8px; }
.hero-landing p{ font-size: 18px !important; color:#dbe7ff; }
.chips{ display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 6px;}
.chip{ background:var(--chip); border:1px solid rgba(255,255,255,.12); padding:6px 10px; border-radius:999px; font-weight:700; font-size:13px; letter-spacing:.2px; }
.bullets{ margin:10px 0 0; }
.bullets li{ margin:4px 0; }

/* Feature cards row */
.feature-row{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px; margin-top:12px;}
.feature{ background:var(--card); border:1px solid rgba(255,255,255,.12); border-radius:14px; padding:14px 16px; box-shadow: var(--shadow); }
.feature .t{ font-weight:800; margin-bottom:4px; }

/* Cards */
.card{
  background: var(--card);
  border: 1px solid rgba(255,255,255,.12);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 16px 18px;
  margin-bottom: 14px;
}

/* Dataframe */
[data-testid="stDataFrame"] div[role="grid"]{
  border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,.12);
}
[data-testid="stDataFrame"] [role="columnheader"]{
  background: #1f3a60 !important; color:#f5f9ff !important; font-weight:700 !important;
}

/* Images */
img{ border-radius: 12px; box-shadow: 0 10px 24px rgba(0,0,0,.20); }

/* Dividers */
hr{
  border: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.20), transparent);
  margin: 12px 0 18px;
}

/* spacing awal */
.block-container > div:nth-child(1) .stColumns{ gap: 16px !important; }

/* Badge pengganti logo */
.brand-badge{
  width:96px; height:96px;
  display:flex; align-items:center; justify-content:center;
  margin: 6px auto 12px auto;
  border-radius:16px;
  background: linear-gradient(180deg, rgba(37,99,235,.12), rgba(2,132,199,.10));
  box-shadow: 0 8px 18px rgba(0,0,0,.10);
  font-size:44px;
}

/* Credit floating (pojok kanan bawah) */
.app-credit{
  position: fixed; right: 18px; bottom: 12px; z-index: 9999;
  background: var(--credit-bg);
  color: var(--credit-ink);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(255,255,255,.18);
  padding: 6px 12px; border-radius: 999px;
  font-size: 13px; font-weight: 700; letter-spacing:.2px;
  box-shadow: 0 8px 20px rgba(0,0,0,.25);
}
</style>
""", unsafe_allow_html=True)

# ===== Tema Terang (toggle) ===== #
light_theme = st.sidebar.checkbox("🌞 Tema terang", value=False)
if light_theme:
  st.markdown("""
  <style>
  :root{
    --bg-top:#f6f8fc; --bg-bottom:#eef2f8;
    --card:#ffffff;
    --ink:#0b1220;
    --ink-dim:#334155;
    --link:#1d4ed8;
    --chip:#eef2f7;
    --credit-bg:rgba(255,255,255,.96);
    --credit-ink:#0b1220;
  }

  [data-testid="stSidebar"]{
    background: linear-gradient(180deg, #f3f6fb 0%, #e9eef7 100%) !important;
    border-right: 1px solid rgba(0,0,0,.06);
    box-shadow: 8px 0 24px rgba(0,0,0,.08);
  }
  html body [data-testid="stSidebar"] h1.sidebar-title.sidebar-title{
    color:#0b1220 !important; text-shadow:none !important;
  }
  [data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li a,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span{
    color:#0b1220 !important;
  }

  h1, h2, h3{ color: var(--ink) !important; }
  p, li{ color: var(--ink-dim) !important; }
  ul li, ol li{ color: var(--ink) !important; font-weight:500; }
  html body strong, html body b{ color: var(--ink) !important; }
  html body em, html body i, html body small{ color: var(--ink-dim) !important; }
  a{ color: var(--link) !important; text-decoration: underline; text-underline-offset: 2px; }

  .stAlert p, .stAlert div{ color: var(--ink) !important; }
  [data-testid="stCaptionContainer"] *{ color:#475569 !important; }
  [data-testid="stDataFrame"] *{ color:#0b1220 !important; }

  .card{
    background: var(--card);
    border: 1px solid rgba(0,0,0,.08);
    box-shadow: 0 8px 18px rgba(15,23,42,.06);
  }
  .hero{
    background: linear-gradient(135deg, rgba(37,99,235,.08), rgba(2,132,199,.06));
    border: 1px solid rgba(0,0,0,.10);
    box-shadow: 0 10px 24px rgba(15,23,42,.08);
  }
  .hero h1{ color: var(--ink) !important; }
  .hero p{ color: var(--ink-dim) !important; }

  .hero-landing{
    background: linear-gradient(135deg, #ffffff, #f3f7ff);
    border: 1px solid rgba(0,0,0,.08);
    box-shadow: 0 16px 40px rgba(15,23,42,.08);
  }
  .hero-landing h1, .hero-landing h2, .hero-landing h3{ color: var(--ink) !important; }
  .hero-landing p{ color: var(--ink-dim) !important; }
  .chip{ background: var(--chip); border-color: rgba(0,0,0,.08); color:#0b1220 !important; }

  hr{ height:1px !important; background: rgba(0,0,0,.12) !important; border:0 !important; }
  div[role="separator"]{
    height:1px !important; background: rgba(0,0,0,.12) !important;
    border-radius:0 !important; margin:12px 0 18px !important;
  }

  [role="columnheader"]{
    background:#eef2f7 !important;
    color:#0b1220 !important;
    font-weight:700 !important;
  }

  .stButton>button{
    background: linear-gradient(180deg, #1d4ed8, #1e40af) !important;
    color:#ffffff !important; font-size:18px !important; font-weight:800 !important;
    padding:.78rem 1.15rem !important; border-radius:12px !important;
    border:1px solid #1e40af !important; text-shadow:0 1px 0 rgba(0,0,0,.22);
    box-shadow:0 12px 28px rgba(30,64,175,.22) !important;
  }
  .stButton>button *{ color:#ffffff !important; fill:#ffffff !important; }
  .stButton>button:hover{ filter:brightness(1.06); box-shadow:0 14px 32px rgba(30,64,175,.28) !important; }
  .stButton>button:focus-visible{ outline:3px solid rgba(29,78,216,.35) !important; outline-offset:2px; }

  .app-credit{ background: var(--credit-bg); color: var(--credit-ink); border:1px solid rgba(0,0,0,.08); }
  </style>
  """, unsafe_allow_html=True)

# ===== Credit floating (muncul di semua halaman) =====
st.markdown('<div class="app-credit">Didesain & dikembangkan oleh <b>Rifdah Apriliani</b></div>', unsafe_allow_html=True)

# ===== Helpers (UI only) =====
def hero(title:str, subtitle:str=""):
    st.markdown(f"""
    <div class="hero">
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def card_start(): st.markdown('<div class="card">', unsafe_allow_html=True)
def card_end():   st.markdown('</div>', unsafe_allow_html=True)

def md_tel_to_html(md_link:str) -> str:
    m = re.match(r'\[(.*?)\]\(tel:([^)]+)\)', md_link.strip())
    if m:
        label, num = m.group(1), m.group(2)
        return f'<a href="tel:{num}" style="color:#3b82f6; text-decoration:none;">{label}</a>'
    return md_link

# ---------- Grad-CAM utilities (tanpa OpenCV) ----------
def _find_last_conv_layer(model):
    last = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last = layer.name
            break
    return last

def gradcam_heatmap(img_array, model, last_conv_name=None):
    if last_conv_name is None:
        last_conv_name = _find_last_conv_layer(model)
    if last_conv_name is None:
        return None, None
    conv_layer = model.get_layer(last_conv_name)
    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    heatmap = tf.image.resize(heatmap[..., np.newaxis], (img_array.shape[1], img_array.shape[2])).numpy().squeeze()
    return heatmap, last_conv_name

def overlay_heatmap(pil_img, heatmap, alpha=0.35, cmap_name='jet'):
    heatmap = np.uint8(255 * heatmap)
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(heatmap)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored).resize(pil_img.size)
    return Image.blend(pil_img.convert("RGB"), colored_img.convert("RGB"), alpha=alpha)

# ---- Util: Ekstraktor fitur dari CNN untuk PCA-LDA ----
def build_feature_extractor(model):
    # coba nama layer umum
    for lname in ["global_average_pooling2d", "avg_pool", "flatten"]:
        try:
            layer = model.get_layer(lname)
            return Model(model.input, layer.output)
        except Exception:
            pass
    # fallback: cari layer sebelum output yang berdimensi > 1
    for layer in reversed(model.layers[:-1]):
        try:
            shp = layer.output_shape
            last_dim = shp[-1] if isinstance(shp, tuple) else None
            if isinstance(last_dim, int) and last_dim > 1:
                return Model(model.input, layer.output)
        except Exception:
            try:
                return Model(model.input, layer.output)
            except Exception:
                continue
    # fallback terakhir: pakai layer -2
    return Model(model.input, model.layers[-2].output)

# ================== Halaman Selamat Datang ==================
if "started" not in st.session_state:
    st.session_state["started"] = False

if not st.session_state["started"]:
    st.markdown("""
    <div class="hero-landing">
      <div class="chips">
        <span class="chip">PCA</span>
        <span class="chip">LDA</span>
        <span class="chip">CNN</span>
        <span class="chip">X-Ray Analysis</span>
      </div>
      <h1>👋 Selamat Datang di Sistem Informasi<br/>Diagnosis Pneumonia</h1>
      <p>Aplikasi ini menggunakan PCA dan LDA untuk menganalisis citra rontgen paru-paru dalam mendeteksi pneumonia.</p>
      <ul class="bullets">
        <li>✅ Antarmuka sederhana & informatif</li>
        <li>✅ Prediksi cepat dengan confidences</li>
        <li>✅ Eksplanasi opsional (Grad-CAM)</li>
      </ul>
      <div class="feature-row">
        <div class="feature">
          <div class="t">🧠 Akurasi & Efisiensi</div>
          <div class="d">Reduksi dimensi (PCA) + Klasifikasi (LDA/CNN).</div>
        </div>
        <div class="feature">
          <div class="t">🗂️ Kelola Data Pasien</div>
          <div class="d">Pencarian, filter, ekspor CSV, dan kontrol hapus.</div>
        </div>
        <div class="feature">
          <div class="t">📄 Laporan</div>
          <div class="d">Unduh laporan hasil (PDF/TXT) dan Grad-CAM.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    if st.button("🚀 Mulai"):
        st.session_state["started"] = True
        st.rerun()
    st.stop()

# ================== Memuat Model PCA, LDA, dan CNN (cached + safe) ==================
@st.cache_resource
def load_all_models():
    with open("pca_model.pkl", "rb") as pca_file:
        pca_obj = pickle.load(pca_file)
    with open("lda_model.pkl", "rb") as lda_file:
        lda_obj = pickle.load(lda_file)

    # pastikan file CNN ada (download dari Hugging Face jika belum)
    ensure_cnn_model_local()

    # compile=False aman untuk Keras 3 / TF 2.20
    cnn_obj = load_model(LOCAL_MODEL_PATH, compile=False)
    return pca_obj, lda_obj, cnn_obj

try:
    pca, lda, cnn_model = load_all_models()
    feature_extractor = build_feature_extractor(cnn_model)
except Exception as e:
    st.error(f"❌ Gagal memuat model/artefak: {e}")
    st.stop()

# ================== Style Tambahan (sidebar) ==================
st.markdown("""
    <style>
        [data-testid="stSidebar"] { padding: 10px !important; border-right: 3px solid #004080; }
        .sidebar-title { text-align: center; font-size: 26px !important; font-weight: 800 !important; color: white !important; margin-bottom: 10px; letter-spacing: 1px; }
        .stButton>button { font-size: 14px !important; }
    </style>
""", unsafe_allow_html=True)

# ================== Sidebar dengan Navigasi ==================
SHOW_LOGO = False
if SHOW_LOGO:
    st.sidebar.image("images/untad.png", width=120)
else:
    st.sidebar.markdown("""<div class="brand-badge">🩺</div>""", unsafe_allow_html=True)

st.sidebar.markdown("<h1 class='sidebar-title'>🔬 Dashboard PCA-LDA</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigasi",
    ["🏠 Home", "🔍 Diagnosa", "📊 Data Pasien", "💡 Tentang Pneumonia", "💊 Pengobatan",
     "👨‍⚕️ Konsultasi & Pelayanan Kesehatan", "🧪 Tentang Model"],
    index=0
)

# ================== Halaman Home ==================
if page == "🏠 Home":
    hero("📊 Sistem Informasi Diagnosis Pneumonia",
         "PCA & LDA untuk reduksi dimensi dan klasifikasi berdasarkan citra rontgen paru-paru.")

    card_start()
    st.subheader("🔬 Apa itu PCA (Principal Component Analysis)?")
    st.markdown("""
    **Principal Component Analysis (PCA)** adalah metode statistik yang digunakan untuk **mengurangi dimensi data** tanpa banyak kehilangan informasi.
    Dalam diagnosis pneumonia:
    - **PCA membantu mengekstraksi fitur utama** dari citra rontgen paru-paru.
    - **Menghilangkan noise atau informasi yang tidak relevan**, sehingga model lebih akurat dan cepat.
    - **Hasil reduksi PCA** digunakan sebagai input untuk model klasifikasi, seperti LDA atau CNN.
    """)
    card_end()

    card_start()
    st.subheader("📊 Apa itu LDA (Linear Discriminant Analysis)?")
    st.markdown("""
    **Linear Discriminant Analysis (LDA)** adalah metode klasifikasi yang mencari **kombinasi terbaik dari fitur-fitur** agar dapat membedakan antara kelas dengan lebih efektif.
    Dalam kasus ini:
    - LDA membantu dalam **klasifikasi pasien Normal vs Pneumonia**.
    - LDA bekerja setelah PCA, **menggunakan fitur utama dari PCA** untuk meningkatkan akurasi prediksi.
    - LDA memastikan bahwa fitur yang digunakan memberikan informasi yang relevan bagi klasifikasi penyakit.
    """)
    card_end()

    card_start()
    st.subheader("📋 Langkah-langkah PCA-LDA dalam Diagnosis Pneumonia:")
    st.markdown("""
    1️⃣ **Preprocessing Data**
       - Menghilangkan noise dan menyesuaikan skala citra rontgen.

    2️⃣ **Ekstraksi Fitur dengan PCA**
       - Mengambil fitur penting dari citra rontgen dan mengurangi dimensi data.

    3️⃣ **Reduksi Dimensi dengan PCA**
       - Memilih komponen utama yang paling berkontribusi dalam klasifikasi.

    4️⃣ **Klasifikasi Data dengan LDA**
       - Menganalisis fitur utama dari PCA dan menentukan apakah pasien mengalami **pneumonia atau normal**.

    5️⃣ **Evaluasi Model**
       - Mengukur akurasi model menggunakan confusion matrix, ROC-AUC, dan teknik cross-validation.
    """)
    st.markdown("---")
    st.subheader("📈 Keunggulan PCA-LDA dalam Diagnosis Pneumonia")
    st.markdown("""
    - **Meningkatkan Akurasi**: PCA-LDA menghilangkan informasi tidak relevan dan fokus pada pola penting.
    - **Mengurangi Dimensi Data**: Membantu model bekerja lebih cepat dan efisien.
    - **Mendeteksi Pneumonia dengan Lebih Baik**: Menggunakan kombinasi PCA dan LDA membantu membedakan pasien normal dan yang terkena pneumonia.
    """)
    card_end()

    if st.button("🔙 Kembali ke Halaman Awal"):
        st.session_state["started"] = False
        st.rerun()

# ================== Halaman Diagnosa (Prediksi Gambar) ==================
elif page == "🔍 Diagnosa":
    hero("🔍 Diagnosa Pneumonia", "Unggah citra rontgen, isi data pasien, lalu jalankan prediksi.")

    card_start()

    batch_mode = st.checkbox("🧪 Mode batch (uji banyak gambar sekaligus)", value=False)

    if batch_mode:
        files = st.file_uploader("Unggah beberapa citra rontgen", type=["jpg","jpeg","png"], accept_multiple_files=True)
        thr = st.slider("⚙️ Ambang deteksi (threshold)", 0.10, 0.90, 0.50, 0.05)
        save_ok = st.checkbox("💾 Simpan ke Data Pasien", value=True)
        show_gc = st.checkbox("🔥 Perlihatkan Grad-CAM untuk setiap gambar", value=False)
        alpha = st.slider("Transparansi overlay Grad-CAM", 0.05, 0.90, 0.35, 0.05)
        cmap_name = st.selectbox("Colormap Grad-CAM", ["jet","viridis","plasma","magma","inferno"], index=0)
        if files:
            for idx, f in enumerate(files, 1):
                st.markdown(f"**Berkas {idx}: {f.name}**")
                image = Image.open(f)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                img_resized = image.resize((224,224))
                img_array = np.array(img_resized)/255.0
                img_array = np.expand_dims(img_array, axis=0)

                start = time.perf_counter()
                with st.spinner("🔄 Menghitung prediksi..."):
                    pred = cnn_model.predict(img_array, verbose=0)
                dur = time.perf_counter() - start

                prob = float(pred[0][0]) * 100.0
                result = "⚠️ Pneumonia" if pred[0][0] > thr else "✅ Normal"

                # LDA berbasis fitur
                feat = feature_extractor.predict(img_array, verbose=0)
                feat_flat = feat.reshape(1, -1) if len(feat.shape) > 2 else feat
                pca_feat = pca.transform(feat_flat)
                lda_pred = lda.predict_proba(pca_feat)
                prob_lda = float(lda_pred[0][1]) * 100

                st.image(image, caption="🖼️ Citra Rontgen", use_column_width=True)
                st.write(f"📊 **CNN Model:** {result}")
                st.write(f"📈 **Probabilitas CNN:** {prob:.2f}%")
                st.write(f"📈 **Probabilitas LDA:** {prob_lda:.2f}%")
                st.caption(f"⏱️ {dur:.2f} detik • threshold {thr:.2f}")

                if show_gc:
                    heat, last_name = gradcam_heatmap(img_array, cnn_model)
                    if heat is None:
                        st.info("Tidak menemukan layer konvolusi terakhir untuk Grad-CAM.")
                    else:
                        overlay = overlay_heatmap(img_resized, heat, alpha=alpha, cmap_name=cmap_name)
                        st.image(overlay, caption=f"Grad-CAM (layer: {last_name})", use_column_width=True)

                if save_ok:
                    if "data_pasien" not in st.session_state:
                        st.session_state["data_pasien"] = []
                    st.session_state["data_pasien"].append({
                        "No": len(st.session_state["data_pasien"]) + 1,
                        "Nama": f.name,
                        "Usia": 0,
                        "Gejala": "-",
                        "Hasil Prediksi": result,
                        "Confidence CNN (%)": f"{prob:.2f}",
                        "Confidence LDA (%)": f"{prob_lda:.2f}"
                    })

            st.success("✅ Batch selesai.")
        card_end()
    else:
        uploaded_file = st.file_uploader("Unggah Citra Rontgen Paru-paru", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="🖼️ Citra Rontgen", use_column_width=True)

            w, h = image.size
            if w < 224 or h < 224:
                st.warning("⚠️ Resolusi gambar rendah (<224px). Hasil bisa kurang akurat.")
            if st.checkbox("✨ Tingkatkan kontras otomatis (preview)"):
                preview = ImageEnhance.Contrast(image).enhance(1.2)
                st.image(preview, caption="Preview kontras ditingkatkan", use_column_width=True)

            nama = st.text_input("Nama Pasien")
            usia = st.number_input("Usia", min_value=0, max_value=120, step=1)
            gejala = st.text_area("Gejala yang dialami")

            thr = st.slider("⚙️ Ambang deteksi (threshold)", 0.10, 0.90, 0.50, 0.05)
            save_ok = st.checkbox("💾 Simpan ke Data Pasien", value=True)

            col_pred, col_reset = st.columns([1,1])
            with col_pred:
                pred_btn = st.button("🔍 Prediksi")
            with col_reset:
                if st.button("🧹 Reset Form"):
                    st.rerun()

            if pred_btn:
                if nama and usia > 0 and gejala:
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    image_resized = image.resize((224, 224))
                    image_array = np.array(image_resized) / 255.0
                    image_array = np.expand_dims(image_array, axis=0)

                    start = time.perf_counter()
                    with st.spinner("🔄 Menghitung prediksi..."):
                        cnn_prediction = cnn_model.predict(image_array, verbose=0)
                    dur = time.perf_counter() - start
                    prob_pneumonia = float(cnn_prediction[0][0]) * 100

                    if cnn_prediction[0][0] > thr:
                        cnn_result = "⚠️ Pneumonia"
                        interpretation = "Citra menunjukkan indikasi pneumonia. Segera konsultasikan dengan dokter!"
                    else:
                        cnn_result = "✅ Normal"
                        interpretation = "Citra menunjukkan kondisi paru-paru normal. Tetap jaga kesehatan!"

                    # Ekstraksi fitur yang tepat untuk PCA-LDA
                    feat = feature_extractor.predict(image_array, verbose=0)
                    feat_flat = feat.reshape(1, -1) if len(feat.shape) > 2 else feat
                    pca_features = pca.transform(feat_flat)
                    lda_prediction = lda.predict_proba(pca_features)
                    prob_lda = float(lda_prediction[0][1]) * 100

                    if save_ok:
                        if "data_pasien" not in st.session_state:
                            st.session_state["data_pasien"] = []
                        st.session_state["data_pasien"].append({
                            "No": len(st.session_state["data_pasien"]) + 1,
                            "Nama": nama,
                            "Usia": usia,
                            "Gejala": gejala,
                            "Hasil Prediksi": cnn_result,
                            "Confidence CNN (%)": f"{prob_pneumonia:.2f}",
                            "Confidence LDA (%)": f"{prob_lda:.2f}"
                        })
                        st.success("✅ Data pasien dan hasil prediksi berhasil disimpan!")

                    st.subheader("🩺 Hasil Prediksi:")
                    st.write(f"📊 **CNN Model:** {cnn_result}")
                    st.write(f"📊 **PCA-LDA Model:** {'✅ Normal' if lda_prediction[0][0] > thr else '⚠️ Pneumonia'}")
                    st.write(f"📈 **Probabilitas CNN:** {prob_pneumonia:.2f}%")
                    st.write(f"📈 **Probabilitas LDA:** {prob_lda:.2f}%")
                    st.caption(f"⏱️ {dur:.2f} detik • threshold {thr:.2f}")
                    st.warning(interpretation)

                    if st.checkbox("🔥 Tampilkan Grad-CAM (eksplanasi)"):
                        alpha = st.slider("Transparansi overlay", 0.05, 0.90, 0.35, 0.05)
                        cmap_name = st.selectbox("Colormap", ["jet","viridis","plasma","magma","inferno"])
                        heatmap, last_name = gradcam_heatmap(image_array, cnn_model)
                        if heatmap is None:
                            st.info("Tidak menemukan layer konvolusi terakhir untuk Grad-CAM.")
                        else:
                            overlay_img = overlay_heatmap(image_resized, heatmap, alpha=alpha, cmap_name=cmap_name)
                            st.image(overlay_img, caption=f"Grad-CAM (layer: {last_name})", use_column_width=True)
                            buf = io.BytesIO()
                            overlay_img.save(buf, format="PNG")
                            st.download_button("⬇️ Unduh Grad-CAM (PNG)", data=buf.getvalue(),
                                               file_name=f"gradcam_{nama or 'pasien'}.png", mime="image/png")

                    if st.checkbox("📄 Siapkan laporan (PDF)"):
                        try:
                            from fpdf import FPDF
                            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
                            pdf.cell(0,10,"Laporan Pemeriksaan Pneumonia", ln=1)
                            pdf.cell(0,8,f"Tanggal: {datetime.now():%Y-%m-%d %H:%M}", ln=1)
                            pdf.cell(0,8,f"Nama: {nama} | Usia: {usia}", ln=1)
                            pdf.multi_cell(0,8,f"Gejala: {gejala}")
                            pdf.ln(2)
                            pdf.cell(0,8,f"Hasil CNN: {cnn_result}", ln=1)
                            pdf.cell(0,8,f"Prob. CNN: {prob_pneumonia:.2f}% | Prob. LDA: {prob_lda:.2f}%", ln=1)
                            pdf.cell(0,8,f"Threshold: {thr:.2f}", ln=1)
                            path = "laporan_pemeriksaan.pdf"; pdf.output(path)
                            with open(path,"rb") as f:
                                st.download_button("⬇️ Unduh Laporan (PDF)", f, file_name="laporan_pneumonia.pdf", mime="application/pdf")
                        except Exception:
                            txt = f"""Laporan Pemeriksaan Pneumonia
Tanggal: {datetime.now():%Y-%m-%d %H:%M}
Nama: {nama} | Usia: {usia}
Gejala: {gejala}

Hasil CNN: {cnn_result}
Prob. CNN: {prob_pneumonia:.2f}% | Prob. LDA: {prob_lda:.2f}%
Threshold: {thr:.2f}
"""
                            st.download_button("⬇️ Unduh Laporan (TXT)", txt, file_name="laporan_pneumonia.txt")

                    st.caption("⚠️ Hasil ini adalah alat bantu dan bukan diagnosis final. Tetap konsultasikan dengan dokter, terutama jika gejala menetap/berat.")
    card_end()

    if st.button("🔙 Kembali ke Dashboard"):
        st.session_state["page"] = "🏠 Home"; st.rerun()
    if st.button("🔙 Kembali ke Halaman Awal"):
        st.session_state["started"] = False; st.rerun()

# ================== Halaman Data Pasien ==================
elif page == "📊 Data Pasien":
    hero("📊 Data Pasien", "Cari, filter, dan kelola data hasil pemeriksaan.")

    if "data_pasien" not in st.session_state:
        st.session_state["data_pasien"] = []

    card_start()
    st.markdown("### 🔍 Pencarian & Filter Data Pasien")

    search_query = st.text_input("🔍 Cari Nama Pasien", "")
    filter_usia = st.slider("🔢 Filter Usia", min_value=0, max_value=120, value=(0, 120))
    filter_hasil = st.selectbox("📋 Filter Hasil Prediksi", ["Semua", "✅ Normal", "⚠️ Pneumonia"])

    st.markdown("---")
    st.markdown("### 📄 Data Pasien Terdaftar")

    if len(st.session_state["data_pasien"]) > 0:
        df_pasien = pd.DataFrame(st.session_state["data_pasien"])
        if search_query:
            df_pasien = df_pasien[df_pasien["Nama"].str.contains(search_query, case=False, na=False)]
        df_pasien = df_pasien[(df_pasien["Usia"] >= filter_usia[0]) & (df_pasien["Usia"] <= filter_usia[1])]
        if filter_hasil != "Semua":
            df_pasien = df_pasien[df_pasien["Hasil Prediksi"] == filter_hasil]
        if "No" in df_pasien.columns:
            df_pasien = df_pasien.sort_values("No", ascending=False)

        # Editor dengan kolom "Hapus?"
        edit_df = df_pasien.copy()
        if "Hapus?" not in edit_df.columns:
            edit_df["Hapus?"] = False
        edited = st.data_editor(edit_df, width=1000, height=400, disabled=[], key="editor")

        # Unduh CSV
        csv_bytes = df_pasien.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh CSV (tampilan saat ini)", csv_bytes, "data_pasien.csv", "text/csv")

        # Unduh Excel
        from io import BytesIO
        excel_buf = BytesIO()
        with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
            df_pasien.to_excel(writer, index=False, sheet_name="Data Pasien")
        st.download_button("⬇️ Unduh Excel", excel_buf.getvalue(),
                           "data_pasien.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Hapus yang dicentang (fitur tambahan, tanpa menghapus fitur lama)
        to_drop_idx = edited.index[edited["Hapus?"] == True].tolist()
        if to_drop_idx and st.button(f"🗑️ Hapus {len(to_drop_idx)} baris terpilih"):
            # mapping kembali ke session_state berdasarkan "No"
            asal = pd.DataFrame(st.session_state["data_pasien"])
            no_drop = edited.iloc[to_drop_idx]["No"].values
            asal = asal[~asal["No"].isin(no_drop)]
            st.session_state["data_pasien"] = asal.to_dict(orient="records")
            st.success("✅ Data terpilih dihapus.")
            st.rerun()

        # === Mekanisme hapus lama (dipertahankan) ===
        hapus_index = st.number_input("Masukkan nomor pasien yang ingin dihapus", min_value=1, max_value=len(df_pasien), step=1, key="hapus_index")
        if st.button("🗑️ Hapus Data Pasien"):
            if 1 <= hapus_index <= len(st.session_state["data_pasien"]):
                st.session_state["data_pasien"].pop(hapus_index - 1)
                st.success(f"✅ Data pasien ke-{hapus_index} telah dihapus!")
                st.rerun()

        confirm_del = st.checkbox("Saya yakin ingin menghapus nomor ini", key="confirm_del")
        if st.button("✅ Konfirmasi Hapus"):
            if confirm_del and 1 <= hapus_index <= len(st.session_state["data_pasien"]):
                st.session_state["data_pasien"].pop(hapus_index - 1)
                st.success(f"✅ Data pasien ke-{hapus_index} telah dihapus!")
                st.rerun()
            elif not confirm_del:
                st.warning("Centang konfirmasi dulu sebelum menghapus.")

        if st.button("❌ Hapus Semua Data"):
            st.session_state["data_pasien"] = []
            st.success("✅ Semua data pasien telah dihapus!")
            st.rerun()
    else:
        st.info("📌 Belum ada data pasien yang terdaftar.")
    card_end()

    if st.button("🔙 Kembali ke Dashboard"):
        st.session_state["page"] = "🏠 Home"; st.rerun()
    if st.button("🔙 Kembali ke Halaman Awal"):
        st.session_state["started"] = False; st.rerun()

# ================== Halaman Tentang Pneumonia ==================
elif page == "💡 Tentang Pneumonia":
    hero("💡 Informasi tentang Pneumonia", "Ringkasan penyebab, gejala, faktor risiko, dan referensi.")

    card_start()
    st.markdown("""
    Pneumonia adalah infeksi yang menyebabkan peradangan pada kantung udara di satu atau kedua paru-paru.
    Kantung udara ini dapat terisi cairan atau nanah, sehingga menyebabkan batuk berdahak, demam, menggigil, dan kesulitan bernapas.
    """)
    st.subheader("🔬 Penyebab Pneumonia")
    st.markdown("""
    Pneumonia dapat disebabkan oleh berbagai mikroorganisme, antara lain:
    - **Bakteri**: Streptococcus pneumoniae adalah penyebab utama Pneumonia bakteri.
    - **Virus**: Termasuk virus flu (influenza) dan COVID-19.
    - **Jamur**: Lebih sering menyerang orang dengan sistem kekebalan tubuh lemah.
    - **Parasit**: Infeksi dari parasit tertentu yang menyerang paru-paru.
    """)
    st.subheader("📌 Ciri-ciri & Gejala Pneumonia")
    st.markdown("""
    - 🔹 Batuk berdahak atau kering
    - 🔹 Demam tinggi dan menggigil
    - 🔹 Nyeri dada saat bernapas atau batuk
    - 🔹 Sesak napas atau napas cepat
    - 🔹 Lemas dan mudah lelah
    - 🔹 Sakit kepala dan nyeri otot
    """)
    st.subheader("⚠️ Faktor Risiko Pneumonia")
    st.markdown("""
    - **Usia**: Bayi di bawah 2 tahun dan lansia di atas 65 tahun lebih rentan.
    - **Sistem imun lemah**: Penderita HIV/AIDS, kanker, atau diabetes.
    - **Perokok aktif & pasif**: Paparan asap rokok merusak sistem pernapasan.
    - **Penyakit kronis**: Seperti asma, penyakit paru obstruktif kronis (PPOK), dan gagal jantung.
    """)
    st.subheader("📌 Jenis Pneumonia Berdasarkan Penyebabnya")
    pneumonia_types = {
        "1️⃣ Pneumonia Bakteri": "Disebabkan oleh bakteri seperti Streptococcus pneumoniae.",
        "2️⃣ Pneumonia Virus": "Disebabkan oleh virus seperti influenza atau COVID-19.",
        "3️⃣ Pneumonia Jamur": "Biasanya menyerang penderita dengan sistem imun rendah.",
        "4️⃣ Pneumonia Aspirasi": "Terjadi akibat masuknya makanan, minuman, atau cairan ke paru-paru."
    }
    for jenis, deskripsi in pneumonia_types.items():
        st.markdown(f"**{jenis}**: {deskripsi}")
    st.markdown("---")
    st.subheader("🎥 Video Edukasi Pneumonia")
    st.video("https://youtu.be/EdLDuXW8jy4?si=_BOtAthwzR9O2Tvh")
    st.markdown("---")
    st.subheader("📷 Contoh Citra Rontgen Pneumonia vs Normal")
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/pneumonia.jpeg", width=300, caption="🫁 Rontgen Pasien Pneumonia")
    with col2:
        st.image("images/normal.jpg", width=300, caption="🫀 Rontgen Paru-Paru Normal")

    # Sumber tambahan
    st.markdown("---")
    st.subheader("📚 Sumber Informasi")
    st.markdown("""
- Kemenkes RI – **Panduan Nasional Pencegahan dan Pengendalian Pneumonia 2023–2030**  
  📄 https://p2p.kemkes.go.id/wp-content/uploads/2023/12/NAPPD_2023-2030-compressed.pdf
- World Health Organization (WHO) – Pneumonia Overview  
  🌐 https://www.who.int/health-topics/pneumonia
- CDC – Clinical Overview of Pneumonia  
  🌐 https://www.cdc.gov/pneumonia/
""")
    st.caption("Catatan: Informasi di atas bersifat edukatif dan tidak menggantikan diagnosis dokter.")
    card_end()

    if st.button("🔙 Kembali ke Dashboard"):
        st.session_state["page"] = "🏠 Home"; st.rerun()
    if st.button("🔙 Kembali ke Halaman Awal"):
        st.session_state["started"] = False; st.rerun()

# ================== Halaman Pengobatan ==================
elif page == "💊 Pengobatan":
    hero("💊 Pengobatan dan Pencegahan Pneumonia", "Ringkas, jelas, dan mudah dipraktikkan.")

    card_start()
    st.subheader("🩺 Saran Pengobatan:")
    st.markdown("""
    - Menggunakan antibiotik sesuai resep dokter
    - Minum banyak cairan dan istirahat yang cukup
    - Gunakan oksigen jika mengalami kesulitan bernapas
    - Rawat inap di rumah sakit untuk kasus yang parah
    """)
    st.subheader("🛡️ Pencegahan Pneumonia:")
    st.markdown("""
    - Vaksinasi pneumonia dan influenza
    - Menjaga kebersihan tangan
    - Menghindari asap rokok
    - Menjaga daya tahan tubuh dengan pola hidup sehat
    """)
    st.markdown("---")
    st.subheader("📚 Sumber Informasi")
    st.markdown("""
- Kemenkes RI – **Panduan Nasional Pencegahan dan Pengendalian Pneumonia 2023–2030**  
  📄 https://p2p.kemkes.go.id/wp-content/uploads/2023/12/NAPPD_2023-2030-compressed.pdf
- WHO & CDC guideline ringkas pencegahan pneumonia.  
  🌐 https://www.who.int/health-topics/pneumonia • https://www.cdc.gov/pneumonia/
""")
    card_end()

    if st.button("🔙 Kembali ke Dashboard"):
        st.session_state["page"] = "🏠 Home"; st.rerun()
    if st.button("🔙 Kembali ke Halaman Awal"):
        st.session_state["started"] = False; st.rerun()

# ================== Halaman Konsultasi & Pelayanan Kesehatan ==================
elif page == "👨‍⚕️ Konsultasi & Pelayanan Kesehatan":
    hero("👨‍⚕️ Konsultasi & Pelayanan Kesehatan",
         "Jika Anda memiliki pertanyaan atau membutuhkan bantuan medis terkait Pneumonia, silakan datang ke fasilitas terdekat di Kota Palu!")

    card_start()
    st.subheader("🏥 Rumah Sakit Rujukan Terdekat")

    # ==== Grid Cards untuk daftar RS (rapi & responsif) ====
    st.markdown("""
    <style>
    .rs-grid{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px; }
    @media (max-width:1100px){ .rs-grid{ grid-template-columns:1fr; } }
    .rs-card{
      background: var(--card);
      border: 1px solid rgba(0,0,0,.08);
      border-radius: 12px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
    }
    .rs-card h4{ margin:0 0 6px; font-weight:800; color: var(--ink); }
    .rs-meta{ margin:6px 0 0; line-height:1.55; color: var(--ink-dim); }
    .rs-actions a{ margin-right:12px; font-weight:700; text-decoration: underline; text-underline-offset: 2px; }
    </style>
    """, unsafe_allow_html=True)

    rumah_sakit = [
        {"nama": "RSUD Anutapura Palu - Palu",
         "alamat": "Jl. Kangkung No.1, Donggala Kodi, Kec. Ulujadi, Kota Palu, Sulawesi Tengah",
         "tel": ["(0451) 460570"],
         "website": "https://rsapkotapalu.com/",
         "maps": "https://maps.app.goo.gl/U55pJgjEswcik45n7"},
        {"nama": "RSUD Undata Palu - Palu",
         "alamat": "Jl. RE Martadinata, Tondo, Kec. Mantikulore, Kota Palu, Sulawesi Tengah",
         "tel": ["(0451) 4131446", "082195155175"],
         "website": "https://rsudundata.sultengprov.go.id/",
         "maps": "https://maps.app.goo.gl/wKq2waeqjJ5B6Sya7"},
        {"nama": "RSU Samaritan Palu - Palu",
         "alamat": "Jl. Towua No.77",
         "tel": ["(0451) 4010925"],
         "website": "https://www.samaritan.id/",
         "maps": "https://maps.app.goo.gl/uUsVK2xtvRtr9W5F7"},
        {"nama": "RSU Budi Agung - Palu",
         "alamat": "Jl. Maluku No.44, Lolu Selatan, Kec. Palu Timur, Kota Palu, Sulawesi Tengah",
         "tel": ["(0451) 421360"],
         "website": "https://www.rsbapalu.com/",
         "maps": "https://maps.app.goo.gl/o6DiRnjzuMf1tAvu5"},
        {"nama": "RSUD Madani - Palu",
         "alamat": "Jl. Thalua Kochi No.11, Mamboro, Kec. Palu Utara, Kota Palu, Sulawesi Tengah",
         "tel": ["(+62) 451-4916058"],
         "website": "https://rsmadani.sultengprov.go.id/",
         "maps": "https://maps.app.goo.gl/1jYgP5S5tP6tCYZ69"},
        {"nama": "RSU Sis Al-Jufri - Palu",
         "alamat": "Jl. Sis Aljufri No.72, Siranindi, Kec. Palu Barat, Kota Palu, Sulawesi Tengah",
         "tel": ["(0451) 456925"],
         "website": None,
         "maps": "https://maps.app.goo.gl/mSwHdJkR2veceUCJ9"},
        {"nama": "RSU Wirabuana - Palu",
         "alamat": "Jl. Sisingamangaraja No.4, Palu, Sulawesi Tengah",
         "tel": ["(0451) 4215757"],
         "website": None,
         "maps": "https://maps.app.goo.gl/8mo7g1Ei8z3yCcuq8"},
        {"nama": "RSU Bhayangkara - Palu",
         "alamat": "Jl. Dr. Soeharso Lrg.III No.2, Besusu Barat, Kec. Palu Timur, Kota Palu, Sulawesi Tengah",
         "tel": ["(0451) 429714"],
         "website": None,
         "maps": "https://maps.app.goo.gl/vfTqWPMu2tJNb7KQ7"},
        {"nama": "RSU Woodward (BK) - Palu",
         "alamat": "Jl. Woodward No.1, Lolu Selatan, Kec. Palu Timur, Kota Palu, Sulawesi Tengah",
         "tel": ["(0451) 4027430"],
         "website": None,
         "maps": "https://maps.app.goo.gl/nfeXrcAgUrSprJET7"},
    ]

    st.markdown('<div class="rs-grid">', unsafe_allow_html=True)
    for rs in rumah_sakit:
        tel_links = " / ".join(
            [f'<a href="tel:{re.sub(r"[^0-9+]", "", t)}">{t}</a>' for t in rs["tel"]]
        )
        website_html = f'<a href="{rs["website"]}" target="_blank">Website</a>' if rs.get("website") else ""
        maps_html = f'<a href="{rs["maps"]}" target="_blank">Buka di Maps</a>' if rs.get("maps") else ""
        sep = " · " if website_html and maps_html else ""
        st.markdown(
            f'''
            <div class="rs-card">
              <h4>🏥 {rs["nama"]}</h4>
              <div class="rs-meta">📍 {rs["alamat"]}</div>
              <div class="rs-meta">📞 {tel_links}</div>
              <div class="rs-actions">{website_html}{sep}{maps_html}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)
    # ==== akhir grid RS ====

    st.markdown("---")
    st.subheader("👨‍⚕️ Dokter Spesialis Paru-Paru")

    dokter_list = [
        {"nama": "Dr. H. Abdullah Ammarie, SpPD, FINASIM",
        "alamat": "Jl. Suharso, No.14, Besusu Barat, Kec. Palu Timur, Kota Palu, Sulawesi Tengah",
        "kontak": "[(0451) 421270](tel:0451421270)"},
        {"nama": "Dr. Wirjadi Ali",
        "alamat": "Jl. Kimaja, No.74, Donggala Kodi, Kec. Ulujadi, Kota Palu, Sulawesi Tengah",
        "kontak": "[(0451) 425132](tel:0451425132)"},
        {"nama": "Dr. Anton Sinarli",
        "alamat": "Jl. Tg. Santigi, No.11, Lolu Selatan, Kec. Palu Selatan, Kota Palu, Sulawesi Tengah",
        "kontak": "[0812-4222-5225](tel:081242225225)"},
    ]

    for dokter in dokter_list:
        st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
            <p style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">🩺 {dokter['nama']}</p>
            <p style="font-size: 16px; margin: 5px 0;">📍 <b>Alamat:</b> {dokter['alamat']}</p>
            <p style="font-size: 16px; margin: 5px 0;">📞 <b>Kontak:</b> {dokter['kontak']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔙 Kembali ke Dashboard"):
        st.session_state["page"] = "🏠 Home"; st.rerun()
    if st.button("🔙 Kembali ke Halaman Awal"):
        st.session_state["started"] = False; st.rerun()

# ================== Halaman Tentang Model ==================
elif page == "🧪 Tentang Model":
    hero("🧪 Tentang Model", "Detail ringkas model, data latih, metrik, dan keterbatasan.")
    card_start()
    st.markdown("""
**Arsitektur:** CNN (input 224×224), PCA untuk reduksi fitur, LDA untuk klasifikasi tambahan.  
**Ekstraksi Fitur:** Mengambil vektor fitur dari layer sebelum output (Flatten/GAP), lalu diproyeksikan oleh PCA dan dinilai oleh LDA.  
**Data latih/validasi:** (lengkapi sesuai dataset Anda: jumlah gambar, sumber, proporsi train/val).  
**Metrik (contoh):** Akurasi 0.93 • Precision 0.92 • Recall 0.94 • ROC-AUC 0.96.  
**Keterbatasan:** Bukan alat medis; kualitas gambar, artefak, komorbid, dan domain-shift dapat menurunkan akurasi.  
**Keamanan data:** Gambar yang diunggah dipakai untuk prediksi dalam sesi ini saja. Hasil bersifat edukatif dan bukan diagnosis final.
""")
    card_end()

# ====== Privasi (footer singkat) ======
st.caption("🔒 Privasi: Data yang diunggah hanya dipakai untuk proses prediksi dalam sesi ini. "
           "Hasil bersifat edukatif dan **bukan** pengganti diagnosis dokter.")
