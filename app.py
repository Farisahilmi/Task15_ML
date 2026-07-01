# pyrefly: ignore [missing-import]
"""
Aplikasi Streamlit: Prediksi Tingkat Burnout Mahasiswa
Judul Penelitian:
    Implementasi Algoritma Random Forest untuk Prediksi Tingkat Burnout Mahasiswa
    Berdasarkan Faktor Akademik dan Gaya Hidup

Struktur Kode:
    1. Konfigurasi & CSS
    2. Konstanta & Label
    3. Load Model & Encoder
    4. Load Dataset
    5. Sidebar Navigasi
    6. Page 1: Beranda
    7. Page 2: Prediksi Burnout  ← Input Form + Prediction + Result
    8. Page 3: Analisis Dataset  (EDA)
    9. Page 4: Performa Model
    10. Page 5: Tentang
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Burnout Mahasiswa | Random Forest",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. CUSTOM CSS — DARK PREMIUM THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.1) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        margin-bottom: 32px;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.4;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.65);
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99,102,241,0.3);
        border-color: rgba(99,102,241,0.4);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.6);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Badges */
    .badge-low {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white; padding: 8px 24px;
        border-radius: 50px; font-size: 1rem;
        font-weight: 600; display: inline-block;
    }
    .badge-medium {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white; padding: 8px 24px;
        border-radius: 50px; font-size: 1rem;
        font-weight: 600; display: inline-block;
    }
    .badge-high {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white; padding: 8px 24px;
        border-radius: 50px; font-size: 1rem;
        font-weight: 600; display: inline-block;
    }

    /* Result & Input cards */
    .result-card {
        background: rgba(255,255,255,0.04);
        border-radius: 20px; padding: 32px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .input-group {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 20px;
    }

    /* Section Titles */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 16px;
        border-left: 4px solid #a78bfa;
        padding-left: 12px;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(99,102,241,0.4) !important;
    }

    hr { border-color: rgba(255,255,255,0.1) !important; }

    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
    }

    /* Recommendation items */
    .rec-item {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: rgba(255,255,255,0.85);
    }
    .rec-item-warn {
        background: rgba(239,68,68,0.08);
        border-left: 3px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: rgba(255,255,255,0.85);
    }

    /* Probability bars */
    .prob-row {
        display: flex;
        align-items: center;
        margin: 10px 0;
        gap: 12px;
    }
    .prob-label {
        width: 90px;
        font-weight: 600;
        color: rgba(255,255,255,0.85);
        font-size: 0.9rem;
    }
    .prob-bar-bg {
        flex: 1;
        background: rgba(255,255,255,0.08);
        border-radius: 50px;
        height: 20px;
        overflow: hidden;
    }
    .prob-value {
        width: 70px;
        text-align: right;
        font-size: 0.88rem;
        color: rgba(255,255,255,0.7);
        font-weight: 500;
    }
    .validation-error {
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 4px 0;
        color: #fca5a5;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. KONSTANTA & LABEL
# ============================================================
MODEL_PATH   = "model.pkl"
DATASET_PATH = "dataset/student_mental_health_burnout_100k_fixed.xlsx"

# Urutan kolom fitur — HARUS SAMA persis dengan saat training
FEATURE_COLUMNS = [
    'age',
    'gender',
    'academic_year',
    'study_hours_per_day',
    'exam_pressure',
    'academic_performance',
    'stress_level',
    'anxiety_score',
    'depression_score',
    'sleep_hours',
    'physical_activity',
    'social_support',
    'screen_time',
    'internet_usage',
    'financial_stress',
    'family_expectation',
]

# Kolom yang di-encode saat training
CATEGORICAL_COLS = ['gender', 'social_support']

# Label burnout
BURNOUT_LABELS  = ['Low', 'Medium', 'High']
BURNOUT_COLORS  = {
    'Low'   : '#10b981',
    'Medium': '#f59e0b',
    'High'  : '#ef4444',
}

# Label tampilan fitur
FEATURE_LABELS = {
    'age'                 : 'Usia (tahun)',
    'gender'              : 'Jenis Kelamin',
    'academic_year'       : 'Tahun Akademik',
    'study_hours_per_day' : 'Jam Belajar/Hari',
    'exam_pressure'       : 'Tekanan Ujian (1–10)',
    'academic_performance': 'Performa Akademik (IPK)',
    'stress_level'        : 'Tingkat Stres (1–10)',
    'anxiety_score'       : 'Skor Kecemasan (1–20)',
    'depression_score'    : 'Skor Depresi (1–20)',
    'sleep_hours'         : 'Jam Tidur/Hari',
    'physical_activity'   : 'Aktivitas Fisik (jam/minggu)',
    'social_support'      : 'Dukungan Sosial',
    'screen_time'         : 'Screen Time (jam/hari)',
    'internet_usage'      : 'Penggunaan Internet (jam/hari)',
    'financial_stress'    : 'Stres Finansial (1–10)',
    'family_expectation'  : 'Ekspektasi Keluarga (1–10)',
}

# Rentang validasi input
INPUT_RANGES = {
    'age'                 : (18, 24),
    'academic_year'       : (1, 4),
    'exam_pressure'       : (1, 10),
    'stress_level'        : (1, 10),
    'anxiety_score'       : (1, 20),
    'depression_score'    : (1, 20),
    'sleep_hours'         : (3, 9),
    'physical_activity'   : (0, 6),
    'financial_stress'    : (1, 10),
    'family_expectation'  : (1, 10),
    'study_hours_per_day' : (2, 12),
    'academic_performance': (2.00, 4.00),
    'screen_time'         : (1, 8),
    'internet_usage'      : (1, 10),
}

# ============================================================
# 4. LOAD MODEL & ENCODER
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load model Random Forest, scaler, encoder, dan metadata dari model.pkl.
    Kembalikan None jika file tidak tersedia atau tidak valid.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        required_keys = ['model', 'scaler', 'encoders', 'features', 'metrics']
        if all(k in data for k in required_keys):
            return data
    except Exception:
        pass
    return None


# ============================================================
# 5. LOAD DATASET (untuk EDA & statistik)
# ============================================================
@st.cache_data(show_spinner=False)
def load_dataset():
    """
    Load dataset mentah dari Excel/CSV.
    Kembalikan DataFrame, atau None jika tidak tersedia.
    """
    if not os.path.exists(DATASET_PATH):
        return None
    try:
        if DATASET_PATH.endswith('.xlsx') or DATASET_PATH.endswith('.xls'):
            df = pd.read_excel(DATASET_PATH)
        else:
            df = pd.read_csv(DATASET_PATH)
        return df
    except Exception:
        return None


# ============================================================
# 6. FUNGSI VALIDASI INPUT
# ============================================================
def validate_inputs(inputs: dict) -> list[str]:
    """
    Validasi nilai input berdasarkan INPUT_RANGES.
    Kembalikan list pesan error (kosong = semua valid).
    """
    errors = []
    for field, (lo, hi) in INPUT_RANGES.items():
        val = inputs.get(field)
        if val is None:
            continue
        if not (lo <= val <= hi):
            label = FEATURE_LABELS.get(field, field)
            errors.append(f"**{label}** harus antara {lo}–{hi} (nilai sekarang: {val})")
    return errors


# ============================================================
# 7. FUNGSI PREDIKSI
# ============================================================
def predict_burnout(model_data: dict, raw_inputs: dict):
    """
    Lakukan prediksi tingkat burnout dari raw input user.

    Parameters:
        model_data : dict hasil load_model()
        raw_inputs : dict {nama_fitur: nilai}

    Returns:
        pred_label  : str — 'Low' / 'Medium' / 'High'
        prob_dict   : dict — {'Low': float, 'Medium': float, 'High': float}
    """
    rf_model = model_data['model']
    scaler   = model_data['scaler']
    encoders = model_data['encoders']

    # Buat DataFrame dengan urutan kolom yang sama seperti training
    df_input = pd.DataFrame([raw_inputs], columns=FEATURE_COLUMNS)

    # Encode kolom kategorik menggunakan encoder yang sudah disimpan (JANGAN fit ulang)
    for col in CATEGORICAL_COLS:
        if col in encoders and col in df_input.columns:
            le = encoders[col]
            val = df_input[col].iloc[0]
            # Pastikan nilai ada di classes_ encoder
            if val not in le.classes_:
                raise ValueError(
                    f"Nilai '{val}' pada kolom '{col}' tidak dikenali oleh encoder. "
                    f"Nilai yang valid: {list(le.classes_)}"
                )
            df_input[col] = le.transform(df_input[col])

    # Scale fitur — kembalikan sebagai DataFrame agar model.predict() mendapat nama kolom
    input_scaled = pd.DataFrame(
        scaler.transform(df_input),
        columns=FEATURE_COLUMNS
    )

    # Prediksi
    pred_encoded  = rf_model.predict(input_scaled)[0]
    probabilities = rf_model.predict_proba(input_scaled)[0]

    # Decode label prediksi
    le_target = encoders.get('burnout_level')
    if le_target is not None:
        pred_label   = le_target.inverse_transform([pred_encoded])[0]
        class_labels = le_target.inverse_transform(rf_model.classes_)
    else:
        pred_label   = str(pred_encoded)
        class_labels = [str(c) for c in rf_model.classes_]

    prob_dict = dict(zip(class_labels, probabilities))
    return pred_label, prob_dict


# ============================================================
# 8. INISIALISASI: LOAD MODEL & DATASET
# ============================================================
with st.spinner("🔄 Memuat model..."):
    model_data = load_model()

df_raw = load_dataset()

# ============================================================
# 9. SIDEBAR — NAVIGASI
# ============================================================
with st.sidebar:
    st.markdown("## 🎓 Burnout Mahasiswa")
    st.markdown("*Powered by Random Forest*")
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "🔍 Prediksi Burnout", "📊 Analisis Dataset",
         "📈 Performa Model", "ℹ️ Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Status model
    if model_data is not None:
        metrics = model_data.get('metrics', {})
        acc = metrics.get('accuracy', 0)
        st.markdown(
            f"<div style='color:rgba(255,255,255,0.5);font-size:0.8rem;'>"
            f"✅ Model Aktif<br>"
            f"🎯 Akurasi: <b style='color:#a78bfa'>{acc*100:.2f}%</b><br>"
            f"🌳 Random Forest Classifier<br>"
            f"📂 Dataset: student_mental_health_burnout</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='color:#fca5a5;font-size:0.8rem;'>❌ Model belum tersedia.<br>"
            "Jalankan: <code>python train_model.py</code></div>",
            unsafe_allow_html=True
        )

# ============================================================
# PAGE 1 — BERANDA
# ============================================================
if page == "🏠 Beranda":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🎓 Prediksi Tingkat Burnout Mahasiswa</div>
        <div class="hero-subtitle">
            Implementasi Algoritma Random Forest untuk Prediksi Tingkat Burnout Mahasiswa<br>
            Berdasarkan Faktor Akademik dan Gaya Hidup
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Statistik Dataset
    if df_raw is not None and 'burnout_level' in df_raw.columns:
        col1, col2, col3, col4 = st.columns(4)
        total  = len(df_raw)
        low_n  = (df_raw['burnout_level'].str.lower() == 'low').sum()
        med_n  = (df_raw['burnout_level'].str.lower() == 'medium').sum()
        high_n = (df_raw['burnout_level'].str.lower() == 'high').sum()

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total:,}</div>
                <div class="metric-label">Total Data Mahasiswa</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background:linear-gradient(135deg,#10b981,#059669);
                    -webkit-background-clip:text;">{low_n:,}</div>
                <div class="metric-label">🟢 Burnout Rendah</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background:linear-gradient(135deg,#f59e0b,#d97706);
                    -webkit-background-clip:text;">{med_n:,}</div>
                <div class="metric-label">🟡 Burnout Sedang</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background:linear-gradient(135deg,#ef4444,#dc2626);
                    -webkit-background-clip:text;">{high_n:,}</div>
                <div class="metric-label">🔴 Burnout Tinggi</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown('<div class="section-title">📖 Tentang Proyek</div>', unsafe_allow_html=True)
        st.markdown("""
        Proyek ini mengimplementasikan algoritma **Random Forest Classifier** untuk memprediksi
        tingkat burnout mahasiswa menjadi tiga kategori:

        - 🟢 **Low Burnout** — Kondisi baik, tidak ada tanda kelelahan signifikan
        - 🟡 **Medium Burnout** — Tanda-tanda kelelahan mulai muncul, perlu perhatian
        - 🔴 **High Burnout** — Tingkat kelelahan tinggi, perlu intervensi segera

        Model dilatih menggunakan **16 fitur** mencakup faktor akademik (tekanan ujian,
        performa IPK, jam belajar) dan faktor gaya hidup (tidur, olahraga, screen time,
        dukungan sosial, stres finansial).
        """)

    with col_right:
        st.markdown('<div class="section-title">🗂️ Kelompok Fitur</div>', unsafe_allow_html=True)
        features_info = [
            ("👤", "Demografis",   "Usia, Jenis Kelamin, Tahun Akademik"),
            ("📚", "Akademik",     "Jam Belajar, Tekanan Ujian, IPK"),
            ("🧠", "Kesehatan Mental", "Stres, Kecemasan, Depresi"),
            ("💤", "Gaya Hidup",   "Tidur, Aktivitas Fisik"),
            ("📱", "Digital",      "Screen Time, Internet"),
            ("💛", "Sosial",       "Dukungan Sosial"),
            ("💰", "Finansial",    "Stres Finansial, Ekspektasi Keluarga"),
        ]
        for icon, cat, desc in features_info:
            st.markdown(f"""
            <div class="rec-item">
                {icon} <b>{cat}</b>: <small>{desc}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Alur Sistem</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Load Data",     "Dataset mahasiswa 100K baris"),
        ("2", "Preprocessing", "Encoding, cleaning, scaling"),
        ("3", "Training",      "Random Forest 200 estimators"),
        ("4", "Evaluasi",      "Accuracy, F1, CV, Confusion Matrix"),
        ("5", "Prediksi",      "Input data → prediksi burnout level"),
    ]
    cols = st.columns(5)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="padding:16px">
                <div style="font-size:1.8rem;font-weight:700;color:#a78bfa;">{num}</div>
                <div style="font-weight:600;color:#fff;margin:4px 0;">{title}</div>
                <div style="font-size:0.75rem;color:rgba(255,255,255,0.5);">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 2 — PREDIKSI BURNOUT
# ============================================================
elif page == "🔍 Prediksi Burnout":
    st.markdown(
        '<div class="hero-title" style="text-align:center;color:#fff;font-size:1.6rem;'
        'margin-bottom:8px;">🔍 Prediksi Tingkat Burnout Mahasiswa</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align:center;color:rgba(255,255,255,0.5);margin-bottom:32px;">'
        'Isi semua data berikut untuk mendapatkan prediksi tingkat burnout</div>',
        unsafe_allow_html=True
    )

    if model_data is None:
        st.error(
            "❌ Model tidak tersedia. Jalankan terlebih dahulu: `python train_model.py`"
        )
    else:
        # ── INPUT FORM ──────────────────────────────────────────────
        with st.form("burnout_form"):
            col1, col2, col3 = st.columns(3)

            # ─ Kolom 1: Data Demografis & Akademik ─
            with col1:
                st.markdown(
                    '<div class="section-title">👤 Demografis & Akademik</div>',
                    unsafe_allow_html=True
                )
                age = st.number_input(
                    "Usia (tahun) [18–24]",
                    min_value=18, max_value=24, value=20, step=1
                )
                gender = st.selectbox(
                    "Jenis Kelamin",
                    ["Male", "Female"]
                )
                academic_year = st.number_input(
                    "Tahun Akademik [1–4]",
                    min_value=1, max_value=4, value=2, step=1
                )
                study_hours = st.number_input(
                    "Jam Belajar/Hari [2–12]",
                    min_value=2.0, max_value=12.0, value=5.0, step=0.5
                )
                academic_performance = st.number_input(
                    "Performa Akademik / IPK [2.00–4.00]",
                    min_value=2.00, max_value=4.00, value=3.00, step=0.01,
                    format="%.2f"
                )
                exam_pressure = st.slider(
                    "Tekanan Ujian [1–10]", 1, 10, 5
                )

            # ─ Kolom 2: Kesehatan Mental & Gaya Hidup ─
            with col2:
                st.markdown(
                    '<div class="section-title">🧠 Kesehatan Mental & Gaya Hidup</div>',
                    unsafe_allow_html=True
                )
                stress_level = st.slider(
                    "Tingkat Stres [1–10]", 1, 10, 5
                )
                anxiety_score = st.slider(
                    "Skor Kecemasan [1–20]", 1, 20, 10
                )
                depression_score = st.slider(
                    "Skor Depresi [1–20]", 1, 20, 10
                )
                sleep_hours = st.number_input(
                    "Jam Tidur/Hari [3–9]",
                    min_value=3.0, max_value=9.0, value=7.0, step=0.5
                )
                physical_activity = st.number_input(
                    "Aktivitas Fisik (jam/minggu) [0–6]",
                    min_value=0.0, max_value=6.0, value=3.0, step=0.5
                )

            # ─ Kolom 3: Sosial, Digital & Finansial ─
            with col3:
                st.markdown(
                    '<div class="section-title">💛 Sosial, Digital & Finansial</div>',
                    unsafe_allow_html=True
                )
                social_support = st.selectbox(
                    "Dukungan Sosial",
                    ["Low", "Medium", "High"]
                )
                screen_time = st.number_input(
                    "Screen Time (jam/hari) [1–8]",
                    min_value=1.0, max_value=8.0, value=4.0, step=0.5
                )
                internet_usage = st.number_input(
                    "Penggunaan Internet (jam/hari) [1–10]",
                    min_value=1.0, max_value=10.0, value=5.0, step=0.5
                )
                financial_stress = st.slider(
                    "Stres Finansial [1–10]", 1, 10, 5
                )
                family_expectation = st.slider(
                    "Ekspektasi Keluarga [1–10]", 1, 10, 5
                )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "🔍 Prediksi Tingkat Burnout",
                use_container_width=True,  # noqa: deprecated after 2025-12-31
                type="primary"
            )

        # ── PREDICTION & RESULT ─────────────────────────────────────
        if submitted:
            # Kumpulkan input
            raw_inputs = {
                'age'                 : int(age),
                'gender'              : gender,
                'academic_year'       : int(academic_year),
                'study_hours_per_day' : float(study_hours),
                'exam_pressure'       : int(exam_pressure),
                'academic_performance': float(academic_performance),
                'stress_level'        : int(stress_level),
                'anxiety_score'       : int(anxiety_score),
                'depression_score'    : int(depression_score),
                'sleep_hours'         : float(sleep_hours),
                'physical_activity'   : float(physical_activity),
                'social_support'      : social_support,
                'screen_time'         : float(screen_time),
                'internet_usage'      : float(internet_usage),
                'financial_stress'    : int(financial_stress),
                'family_expectation'  : int(family_expectation),
            }

            # ── Validasi Input ──
            errors = validate_inputs(raw_inputs)
            if errors:
                st.markdown("---")
                st.error("⚠️ Terdapat nilai input yang tidak valid. Harap periksa kembali:")
                for err in errors:
                    st.markdown(
                        f'<div class="validation-error">❌ {err}</div>',
                        unsafe_allow_html=True
                    )
            else:
                try:
                    # ── Prediksi ──
                    pred_label, prob_dict = predict_burnout(model_data, raw_inputs)

                    st.markdown("---")
                    st.markdown("## 📊 Hasil Prediksi")

                    res_col1, res_col2 = st.columns([1, 2])

                    # ─ Kolom Kiri: Badge Hasil ─
                    with res_col1:
                        badge_class = f"badge-{pred_label.lower()}"
                        emoji_map   = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                        emoji       = emoji_map.get(pred_label, "⚪")
                        conf        = prob_dict.get(pred_label, 0) * 100

                        label_map_display = {
                            "Low"   : "Low Burnout",
                            "Medium": "Medium Burnout",
                            "High"  : "High Burnout",
                        }
                        display_label = label_map_display.get(pred_label, pred_label)

                        st.markdown(f"""
                        <div class="result-card" style="text-align:center">
                            <div style="font-size:4rem;">{emoji}</div>
                            <div style="font-size:0.95rem;color:rgba(255,255,255,0.6);margin:8px 0;">
                                Tingkat Burnout Mahasiswa
                            </div>
                            <div><span class="{badge_class}">{display_label.upper()}</span></div>
                            <div style="margin-top:16px;font-size:0.9rem;color:rgba(255,255,255,0.5);">
                                Keyakinan Model: <b style="color:#a78bfa">{conf:.2f}%</b>
                            </div>
                        </div>""", unsafe_allow_html=True)

                    # ─ Kolom Kanan: Probabilitas + Chart ─
                    with res_col2:
                        st.markdown("#### Probabilitas Prediksi")

                        # Text probabilitas
                        color_map = {
                            "Low"   : "#10b981",
                            "Medium": "#f59e0b",
                            "High"  : "#ef4444",
                        }
                        for lbl in BURNOUT_LABELS:
                            prob_pct  = prob_dict.get(lbl, 0) * 100
                            bar_color = color_map.get(lbl, "#a78bfa")
                            st.markdown(f"""
                            <div class="prob-row">
                                <div class="prob-label">{lbl}</div>
                                <div class="prob-bar-bg">
                                    <div style="width:{prob_pct:.1f}%;background:{bar_color};
                                        height:100%;border-radius:50px;
                                        transition:width 0.8s ease;"></div>
                                </div>
                                <div class="prob-value">{prob_pct:.2f}%</div>
                            </div>""", unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Bar chart probabilitas
                        probs  = [prob_dict.get(l, 0) * 100 for l in BURNOUT_LABELS]
                        colors = [BURNOUT_COLORS[l] for l in BURNOUT_LABELS]
                        fig = go.Figure(go.Bar(
                            x=BURNOUT_LABELS,
                            y=probs,
                            marker_color=colors,
                            text=[f"{p:.2f}%" for p in probs],
                            textposition='outside',
                            textfont=dict(color='white', size=13),
                        ))
                        fig.update_layout(
                            title=dict(
                                text="Distribusi Probabilitas Prediksi",
                                font=dict(color='white', size=14)
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(255,255,255,0.03)',
                            font=dict(color='white'),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(
                                range=[0, 115],
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.1)',
                                title="Probabilitas (%)"
                            ),
                            margin=dict(t=50, b=20, l=20, r=20),
                            height=300,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # ── Rekomendasi Personal ──
                    st.markdown("---")
                    st.markdown("### 💡 Rekomendasi Personal")

                    if pred_label == 'Low':
                        st.success(
                            "**Kondisi baik!** Mahasiswa menunjukkan tingkat burnout yang rendah. "
                            "Pertahankan pola hidup sehat yang sudah ada."
                        )
                        recs = [
                            "✅ Pertahankan jam tidur yang cukup (7–9 jam untuk mahasiswa)",
                            "✅ Jaga keseimbangan antara akademik dan waktu istirahat",
                            "✅ Pertahankan aktivitas fisik secara rutin",
                            "✅ Jaga komunikasi yang baik dengan keluarga dan teman",
                            "📖 Buat jadwal belajar yang terstruktur agar tetap produktif",
                        ]
                        for r in recs:
                            st.markdown(f'<div class="rec-item">{r}</div>', unsafe_allow_html=True)

                    elif pred_label == 'Medium':
                        st.warning(
                            "**Perhatian!** Mahasiswa menunjukkan tanda-tanda burnout sedang. "
                            "Perlu penyesuaian gaya belajar dan gaya hidup."
                        )
                        recs = []
                        if sleep_hours < 7:
                            recs.append("⚠️ Jam tidur di bawah ideal — targetkan minimal 7 jam/malam")
                        if stress_level >= 7:
                            recs.append("⚠️ Tingkat stres tinggi — coba teknik relaksasi atau meditasi")
                        if anxiety_score >= 14:
                            recs.append("⚠️ Skor kecemasan cukup tinggi — pertimbangkan konseling")
                        if screen_time > 5:
                            recs.append("⚠️ Screen time tinggi — batasi layar non-akademik sebelum tidur")
                        if social_support == 'Low':
                            recs.append("⚠️ Dukungan sosial rendah — aktif bergabung dalam komunitas kampus")
                        recs += [
                            "💬 Bicarakan beban akademik dengan dosen wali atau konselor kampus",
                            "📅 Prioritaskan tugas dengan to-do list yang realistis",
                            "🏃 Tambah aktivitas fisik minimal 30 menit/hari",
                        ]
                        for r in recs:
                            st.markdown(f'<div class="rec-item-warn">{r}</div>', unsafe_allow_html=True)

                    else:  # High
                        st.error(
                            "**⚠️ Perhatian Serius!** Mahasiswa berisiko tinggi mengalami burnout. "
                            "Intervensi dan dukungan segera sangat dianjurkan."
                        )
                        recs = [
                            "🚨 Segera konsultasikan kondisi ke konselor atau psikolog kampus",
                            "🚨 Kurangi beban akademik — diskusikan dengan dosen wali",
                            "💤 Prioritaskan kualitas tidur sebagai kebutuhan utama (min. 7 jam)",
                            "📵 Batasi screen time dan internet usage untuk kegiatan non-akademik",
                            "❤️ Ceritakan perasaan kepada orang tua, teman, atau orang kepercayaan",
                            "🏃 Olahraga ringan setiap hari terbukti mengurangi stres dan depresi",
                            "📚 Pertimbangkan cuti akademik sementara jika kondisi tidak membaik",
                        ]
                        for r in recs:
                            st.markdown(f'<div class="rec-item-warn">{r}</div>', unsafe_allow_html=True)

                    # ── Analisis Faktor Risiko dari Input ──
                    st.markdown("---")
                    st.markdown("### 🎯 Analisis Faktor Risiko dari Input Anda")
                    risk_factors = []

                    if sleep_hours < 6:
                        risk_factors.append(("💤 Jam Tidur Sangat Kurang", f"{sleep_hours} jam/hari (< 6 jam)", "high"))
                    elif sleep_hours < 7:
                        risk_factors.append(("💤 Jam Tidur Kurang", f"{sleep_hours} jam/hari", "medium"))

                    if stress_level >= 8:
                        risk_factors.append(("😰 Stres Sangat Tinggi", f"Skor {stress_level}/10", "high"))
                    elif stress_level >= 6:
                        risk_factors.append(("😰 Stres Cukup Tinggi", f"Skor {stress_level}/10", "medium"))

                    if anxiety_score >= 15:
                        risk_factors.append(("😟 Kecemasan Tinggi", f"Skor {anxiety_score}/20", "high"))
                    elif anxiety_score >= 10:
                        risk_factors.append(("😟 Kecemasan Sedang", f"Skor {anxiety_score}/20", "medium"))

                    if depression_score >= 15:
                        risk_factors.append(("😔 Depresi Tinggi", f"Skor {depression_score}/20", "high"))
                    elif depression_score >= 10:
                        risk_factors.append(("😔 Depresi Sedang", f"Skor {depression_score}/20", "medium"))

                    if exam_pressure >= 8:
                        risk_factors.append(("📝 Tekanan Ujian Sangat Tinggi", f"Skor {exam_pressure}/10", "high"))
                    elif exam_pressure >= 6:
                        risk_factors.append(("📝 Tekanan Ujian Cukup Tinggi", f"Skor {exam_pressure}/10", "medium"))

                    if financial_stress >= 8:
                        risk_factors.append(("💰 Stres Finansial Tinggi", f"Skor {financial_stress}/10", "high"))
                    elif financial_stress >= 6:
                        risk_factors.append(("💰 Stres Finansial Sedang", f"Skor {financial_stress}/10", "medium"))

                    if social_support == 'Low':
                        risk_factors.append(("💛 Dukungan Sosial Rendah", "Low", "high"))

                    if screen_time > 6:
                        risk_factors.append(("📱 Screen Time Berlebihan", f"{screen_time} jam/hari", "high"))
                    elif screen_time > 4:
                        risk_factors.append(("📱 Screen Time Cukup Tinggi", f"{screen_time} jam/hari", "medium"))

                    if physical_activity < 1:
                        risk_factors.append(("🏃 Aktivitas Fisik Sangat Rendah", f"{physical_activity} jam/minggu", "high"))
                    elif physical_activity < 2:
                        risk_factors.append(("🏃 Aktivitas Fisik Rendah", f"{physical_activity} jam/minggu", "medium"))

                    if risk_factors:
                        for label, value, level in risk_factors:
                            css_class = "rec-item-warn" if level == "high" else "rec-item"
                            icon = "🔴" if level == "high" else "🟡"
                            st.markdown(
                                f'<div class="{css_class}">'
                                f'{icon} <b>{label}</b>: {value}</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            '<div class="rec-item">✅ Tidak ada faktor risiko signifikan '
                            'yang terdeteksi dari input Anda.</div>',
                            unsafe_allow_html=True
                        )

                except ValueError as ve:
                    st.error(f"❌ Error encoding input: {ve}")
                except Exception as e:
                    st.error(f"❌ Terjadi error saat prediksi: {e}")

# ============================================================
# PAGE 3 — ANALISIS DATASET (EDA)
# ============================================================
elif page == "📊 Analisis Dataset":
    st.markdown(
        '<div class="hero-title" style="color:#fff;font-size:1.6rem;margin-bottom:24px;">'
        '📊 Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    if df_raw is None:
        st.error("❌ Dataset tidak ditemukan di: " + DATASET_PATH)
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Distribusi Burnout", "🔗 Korelasi", "🧩 Fitur vs Burnout", "📋 Statistik Deskriptif"]
        )

        # Pastikan kolom burnout_level ada
        has_target = 'burnout_level' in df_raw.columns

        # ── Tab 1: Distribusi ──
        with tab1:
            if not has_target:
                st.warning("Kolom 'burnout_level' tidak ditemukan di dataset.")
            else:
                col1, col2 = st.columns(2)
                counts  = df_raw['burnout_level'].value_counts()
                ordered = ['Low', 'Medium', 'High']
                vals    = [counts.get(l, 0) for l in ordered]
                colors  = [BURNOUT_COLORS.get(l, '#a78bfa') for l in ordered]

                with col1:
                    fig = go.Figure(go.Bar(
                        x=ordered, y=vals,
                        marker_color=colors,
                        text=vals, textposition='outside',
                        textfont=dict(color='white', size=13),
                    ))
                    fig.update_layout(
                        title="Distribusi Kelas Burnout Level",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.03)',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                        margin=dict(t=50, b=20), height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig2 = go.Figure(go.Pie(
                        labels=ordered, values=vals,
                        marker_colors=colors,
                        hole=0.45,
                        textinfo='label+percent',
                        textfont=dict(color='white', size=13),
                    ))
                    fig2.update_layout(
                        title="Proporsi Burnout Level",
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(t=50, b=20), height=350,
                        showlegend=False,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Distribusi per gender
                if 'gender' in df_raw.columns:
                    gender_burnout = df_raw.groupby(
                        ['gender', 'burnout_level']
                    ).size().reset_index(name='count')
                    fig4 = px.bar(
                        gender_burnout, x='gender', y='count',
                        color='burnout_level',
                        color_discrete_map=BURNOUT_COLORS,
                        barmode='group',
                        title="Distribusi Burnout per Jenis Kelamin",
                        labels={'gender': 'Jenis Kelamin', 'count': 'Jumlah Mahasiswa'},
                    )
                    fig4.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.03)',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                        margin=dict(t=50, b=20), height=350,
                    )
                    st.plotly_chart(fig4, use_container_width=True)

                # Distribusi per academic_year
                if 'academic_year' in df_raw.columns:
                    year_burnout = df_raw.groupby(
                        ['academic_year', 'burnout_level']
                    ).size().reset_index(name='count')
                    fig5 = px.bar(
                        year_burnout, x='academic_year', y='count',
                        color='burnout_level',
                        color_discrete_map=BURNOUT_COLORS,
                        barmode='group',
                        title="Distribusi Burnout per Tahun Akademik",
                        labels={'academic_year': 'Tahun Akademik', 'count': 'Jumlah Mahasiswa'},
                        category_orders={'burnout_level': ['Low', 'Medium', 'High']},
                    )
                    fig5.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.03)',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False, tickmode='array',
                                   tickvals=[1, 2, 3, 4],
                                   ticktext=['Tahun 1', 'Tahun 2', 'Tahun 3', 'Tahun 4']),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                        margin=dict(t=50, b=20), height=380,
                    )
                    st.plotly_chart(fig5, use_container_width=True)

        # ── Tab 2: Korelasi ──
        with tab2:
            from sklearn.preprocessing import LabelEncoder as LE
            df_enc = df_raw.copy()
            for col in df_enc.select_dtypes(include=['object', 'str']).columns:
                le = LE()
                df_enc[col] = le.fit_transform(df_enc[col].astype(str))

            numeric_cols = df_enc.select_dtypes(include='number').columns.tolist()
            corr = df_enc[numeric_cols].corr()

            fig_hm = px.imshow(
                corr, text_auto='.2f', aspect='auto',
                color_continuous_scale='RdYlGn',
                title="Heatmap Korelasi Fitur",
            )
            fig_hm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(t=60, b=20), height=600,
                coloraxis_colorbar=dict(
                    tickfont=dict(color='white'),
                    title=dict(font=dict(color='white'))
                ),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            # Korelasi terhadap burnout_level
            if 'burnout_level' in corr.columns:
                st.markdown("#### Korelasi dengan Burnout Level")
                corr_target = (
                    corr['burnout_level']
                    .drop('burnout_level', errors='ignore')
                    .sort_values(key=abs, ascending=False)
                )
                fig_bar = go.Figure(go.Bar(
                    x=corr_target.values,
                    y=[FEATURE_LABELS.get(k, k) for k in corr_target.index],
                    orientation='h',
                    marker_color=['#ef4444' if v > 0 else '#10b981' for v in corr_target.values],
                    text=[f"{v:.3f}" for v in corr_target.values],
                    textposition='outside',
                    textfont=dict(color='white'),
                ))
                fig_bar.update_layout(
                    title="Korelasi Setiap Fitur dengan Burnout Level",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.03)',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=False),
                    margin=dict(t=50, b=20, l=200), height=480,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # ── Tab 3: Fitur vs Burnout ──
        with tab3:
            numeric_features_eda = [
                f for f in FEATURE_COLUMNS
                if f not in CATEGORICAL_COLS
                and f in df_raw.select_dtypes(include='number').columns
            ]
            selected_feature = st.selectbox(
                "Pilih fitur untuk dianalisis:",
                numeric_features_eda,
                format_func=lambda x: FEATURE_LABELS.get(x, x)
            )

            if has_target and selected_feature in df_raw.columns:
                col1, col2 = st.columns(2)
                feat_label = FEATURE_LABELS.get(selected_feature, selected_feature)

                with col1:
                    fig_box = px.box(
                        df_raw, x='burnout_level', y=selected_feature,
                        color='burnout_level',
                        color_discrete_map=BURNOUT_COLORS,
                        category_orders={'burnout_level': ['Low', 'Medium', 'High']},
                        title=f"Box Plot: {feat_label} vs Burnout",
                    )
                    fig_box.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.03)',
                        font=dict(color='white'), showlegend=False,
                        margin=dict(t=50, b=20), height=400,
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

                with col2:
                    fig_vio = px.violin(
                        df_raw, x='burnout_level', y=selected_feature,
                        color='burnout_level',
                        color_discrete_map=BURNOUT_COLORS,
                        category_orders={'burnout_level': ['Low', 'Medium', 'High']},
                        box=True,
                        title=f"Violin Plot: {feat_label} vs Burnout",
                    )
                    fig_vio.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.03)',
                        font=dict(color='white'), showlegend=False,
                        margin=dict(t=50, b=20), height=400,
                    )
                    st.plotly_chart(fig_vio, use_container_width=True)

                fig_hist = px.histogram(
                    df_raw, x=selected_feature, color='burnout_level',
                    color_discrete_map=BURNOUT_COLORS,
                    barmode='overlay', opacity=0.75,
                    category_orders={'burnout_level': ['Low', 'Medium', 'High']},
                    title=f"Distribusi {feat_label} per Level Burnout",
                    nbins=30,
                )
                fig_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.03)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    margin=dict(t=50, b=20), height=380,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        # ── Tab 4: Statistik Deskriptif ──
        with tab4:
            st.markdown("#### Statistik Deskriptif Dataset")
            numeric_only = df_raw.select_dtypes(include='number')
            st.dataframe(numeric_only.describe().round(3), use_container_width=True)

            st.markdown("#### Missing Values")
            mv = df_raw.isnull().sum()
            mv_df = pd.DataFrame({
                'Kolom'             : mv.index,
                'Missing Values'    : mv.values,
                'Persentase (%)'    : (mv.values / len(df_raw) * 100).round(2)
            })
            st.dataframe(mv_df, use_container_width=True, hide_index=True)

            st.markdown(f"#### Ukuran Dataset")
            st.markdown(f"- **Total baris**: {len(df_raw):,}")
            st.markdown(f"- **Total kolom**: {len(df_raw.columns)}")
            st.markdown(f"- **Kolom**: {', '.join(df_raw.columns.tolist())}")

# ============================================================
# PAGE 4 — PERFORMA MODEL
# ============================================================
elif page == "📈 Performa Model":
    st.markdown(
        '<div class="hero-title" style="color:#fff;font-size:1.6rem;margin-bottom:24px;">'
        '📈 Performa Model Random Forest</div>',
        unsafe_allow_html=True
    )

    if model_data is None:
        st.error("❌ Model tidak tersedia. Jalankan `python train_model.py` terlebih dahulu.")
    else:
        metrics = model_data.get('metrics', {})
        cm      = model_data.get('confusion_matrix')
        fi      = model_data.get('feature_importances', {})

        # ── Metric Cards ──
        st.markdown("### 🎯 Metrik Evaluasi")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        metric_items = [
            (mc1, "Accuracy",  metrics.get('accuracy', 0),  "#a78bfa"),
            (mc2, "Precision", metrics.get('precision', 0), "#60a5fa"),
            (mc3, "Recall",    metrics.get('recall', 0),    "#34d399"),
            (mc4, "F1-Score",  metrics.get('f1', 0),        "#fb923c"),
            (mc5, "CV Mean",   metrics.get('cv_mean', 0),   "#f472b6"),
        ]
        for col, label, val, color in metric_items:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value"
                        style="background:linear-gradient(135deg,{color},{color}aa);
                        -webkit-background-clip:text;">
                        {val*100:.2f}%
                    </div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        cv_std = metrics.get('cv_std', 0)
        st.markdown(f"""
        <div style="text-align:center;margin:16px 0;color:rgba(255,255,255,0.5);font-size:0.85rem;">
            5-Fold Cross Validation: {metrics.get('cv_mean', 0)*100:.2f}% ± {cv_std*100:.2f}%
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        # ── Confusion Matrix ──
        with col1:
            st.markdown("### 🔢 Confusion Matrix")
            if cm is not None:
                class_labels = model_data.get(
                    'confusion_labels',
                    model_data.get('class_labels', ['High', 'Low', 'Medium'])
                )
                fig_cm = px.imshow(
                    cm,
                    x=class_labels, y=class_labels,
                    text_auto=True, aspect='auto',
                    color_continuous_scale='Purples',
                    labels=dict(x="Prediksi", y="Aktual"),
                    title="Confusion Matrix (Aktual vs Prediksi)",
                )
                fig_cm.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=13),
                    margin=dict(t=60, b=20), height=380,
                    coloraxis_colorbar=dict(tickfont=dict(color='white')),
                    xaxis=dict(
                        title_font=dict(color='white'),
                        tickfont=dict(color='white')
                    ),
                    yaxis=dict(
                        title_font=dict(color='white'),
                        tickfont=dict(color='white')
                    ),
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix tidak tersedia. Jalankan ulang `train_model.py`.")

        # ── Feature Importance ──
        with col2:
            st.markdown("### 🏆 Feature Importance")
            if fi:
                fi_series = pd.Series(fi).sort_values(ascending=True)
                display_labels = [FEATURE_LABELS.get(k, k) for k in fi_series.index]

                fig_fi = go.Figure(go.Bar(
                    x=fi_series.values,
                    y=display_labels,
                    orientation='h',
                    marker=dict(
                        color=fi_series.values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(tickfont=dict(color='white')),
                    ),
                    text=[f"{v:.4f}" for v in fi_series.values],
                    textposition='outside',
                    textfont=dict(color='white', size=10),
                ))
                fig_fi.update_layout(
                    title="Tingkat Kepentingan Setiap Fitur",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.03)',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=False, tickfont=dict(size=10)),
                    margin=dict(t=50, b=20, l=10, r=80), height=480,
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance tidak tersedia.")

        # ── Penjelasan Model ──
        st.markdown("---")
        st.markdown("### ℹ️ Tentang Model Random Forest")
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("""
            **Parameter Model:**
            - `n_estimators = 200` — 200 pohon keputusan
            - `min_samples_split = 5` — Min sampel untuk split node
            - `min_samples_leaf = 2` — Min sampel di leaf node
            - `random_state = 42` — Reprodusibilitas hasil
            - `n_jobs = -1` — Gunakan semua CPU core
            """)
        with p2:
            st.markdown("""
            **Mengapa Random Forest?**
            - ✅ Robust terhadap outlier
            - ✅ Mampu menangani non-linearitas data
            - ✅ Memberikan feature importance secara native
            - ✅ Tidak memerlukan asumsi distribusi data
            - ✅ Ensemble → prediksi lebih stabil dari Decision Tree tunggal
            """)

# ============================================================
# PAGE 5 — TENTANG
# ============================================================
elif page == "ℹ️ Tentang":
    st.markdown(
        '<div class="hero-title" style="color:#fff;font-size:1.6rem;margin-bottom:24px;">'
        'ℹ️ Tentang Proyek</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ## Implementasi Algoritma Random Forest untuk Prediksi Tingkat Burnout Mahasiswa Berdasarkan Faktor Akademik dan Gaya Hidup

        ---

        ### 📌 Deskripsi
        Proyek *Machine Learning* berbasis **Supervised Learning (Classification)** untuk memprediksi
        tingkat burnout **mahasiswa** menjadi tiga kategori (**Low / Medium / High**) menggunakan
        algoritma **Random Forest Classifier**.

        *Burnout* mahasiswa adalah kondisi kelelahan emosional, fisik, dan mental akibat tuntutan
        akademik jangka panjang. Deteksi dini melalui pendekatan Machine Learning memungkinkan
        intervensi yang lebih proaktif dan tepat sasaran.

        ---

        ### 🗂️ Dataset
        - **File**: `student_mental_health_burnout_100k_fixed.xlsx`
        - **Fitur**: 16 fitur (demografis + akademik + kesehatan mental + gaya hidup)
        - **Target**: `burnout_level` — Low / Medium / High

        #### Daftar Fitur:
        | # | Fitur | Deskripsi | Rentang |
        |---|---|---|---|
        | 1 | age | Usia mahasiswa | 18–24 |
        | 2 | gender | Jenis kelamin | Male/Female |
        | 3 | academic_year | Tahun akademik | 1–4 |
        | 4 | study_hours_per_day | Jam belajar per hari | 2–12 |
        | 5 | exam_pressure | Tekanan ujian | 1–10 |
        | 6 | academic_performance | IPK / performa akademik | 2.00–4.00 |
        | 7 | stress_level | Tingkat stres | 1–10 |
        | 8 | anxiety_score | Skor kecemasan | 1–20 |
        | 9 | depression_score | Skor depresi | 1–20 |
        | 10 | sleep_hours | Jam tidur per hari | 3–9 |
        | 11 | physical_activity | Aktivitas fisik (jam/minggu) | 0–6 |
        | 12 | social_support | Dukungan sosial | Low/Medium/High |
        | 13 | screen_time | Screen time (jam/hari) | 1–8 |
        | 14 | internet_usage | Penggunaan internet (jam/hari) | 1–10 |
        | 15 | financial_stress | Stres finansial | 1–10 |
        | 16 | family_expectation | Ekspektasi keluarga | 1–10 |

        ---

        ### 🛠️ Tech Stack
        | Komponen | Teknologi |
        |---|---|
        | Bahasa | Python 3.8+ |
        | ML Framework | Scikit-learn |
        | Web App | Streamlit |
        | Visualisasi | Plotly |
        | Data | Pandas, NumPy |
        """)

    with col2:
        st.markdown("""
        ### 📁 Struktur Proyek
        ```
        Student-Burnout-Prediction/
        │
        ├── dataset/
        │   └── student_mental_health_burnout_100k_fixed.xlsx
        ├── images/
        ├── laporan/
        ├── notebook.ipynb
        ├── preprocessing.py
        ├── train_model.py
        ├── app.py          ← Streamlit App
        ├── model.pkl       ← Model terlatih
        └── requirements.txt
        ```

        ---

        ### 🚀 Cara Menjalankan
        **1. Install dependencies:**
        ```bash
        pip install -r requirements.txt
        ```
        **2. Train model (jika belum ada model.pkl):**
        ```bash
        python train_model.py
        ```
        **3. Jalankan aplikasi:**
        ```bash
        streamlit run app.py
        ```

        ---

        ### 📋 Preprocessing Pipeline
        1. **Load** dataset Excel
        2. **Pilih** 16 kolom fitur + target
        3. **Drop** missing values & duplikat
        4. **Encode** gender & social_support (LabelEncoder)
        5. **Encode** burnout_level (LabelEncoder)
        6. **Split** 80:20 train-test (stratified)
        7. **Scale** fitur (StandardScaler)

        ---

        ### 📚 Referensi
        - Breiman, L. (2001). Random Forests. *Machine Learning*.
        - Salmela-Aro et al. (2022). Student burnout. *CHB*.
        - Scikit-learn: RandomForestClassifier
        """)
