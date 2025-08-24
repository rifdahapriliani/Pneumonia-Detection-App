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
import os
from huggingface_hub import hf_hub_download

# ================== Hugging Face config ==================
HF_REPO_ID = "Rifdah/pneumonia-cnn"   # ganti jika repo berbeda
HF_FILENAME = "cnn_model.h5"
LOCAL_MODEL_PATH = "cnn_model.h5"
HF_TOKEN = os.getenv("HF_TOKEN", None)  # isi di Secrets bila repo private

def ensure_cnn_model_local():
    """Unduh cnn_model.h5 dari Hugging Face jika belum ada di folder kerja."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        with st.spinner("📥 Mengunduh model CNN..."):
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                local_dir=".",
                local_dir_use_symlinks=False,
                token=HF_TOKEN
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
  background: linear-gradient(180deg, #13233a 0%, #0e1f37 100%)
