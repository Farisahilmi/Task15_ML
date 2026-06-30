# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Burnout Siswa | Random Forest",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — DARK PREMIUM THEME
# ============================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Cards */
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
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
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
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.3;
        margin-bottom: 12px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.65);
    }

    /* Burnout Level Badges */
    .badge-low {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 8px 24px;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-medium {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 8px 24px;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 8px 24px;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Result Card */
    .result-card {
        background: rgba(255,255,255,0.04);
        border-radius: 20px;
        padding: 32px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Section Titles */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 16px;
        border-left: 4px solid #a78bfa;
        padding-left: 12px;
    }

    /* Input Group Card */
    .input-group {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px;
    }

    /* Nav Button */
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

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* Streamlit metric overrides */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
    }

    /* Table */
    .dataframe { font-size: 0.85rem; }

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
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONSTANTA
# ============================================================
MODEL_PATH   = "model.pkl"
DATASET_PATH = "dataset/student_burnout.csv"

BURNOUT_COLORS = {
    "Low"    : "#10b981",
    "Medium" : "#f59e0b",
    "High"   : "#ef4444",
}

FEATURE_LABELS = {
    "grade"                : "Kelas (Grade)",
    "gender"               : "Jenis Kelamin",
    "sleep_hours"          : "Jam Tidur/Hari",
    "sleep_quality"        : "Kualitas Tidur (1–5)",
    "homework_hours"       : "Jam PR/Hari",
    "tests_per_week"       : "Jumlah Tes/Minggu",
    "extracurricular_hours": "Jam Ekskul/Hari",
    "num_activities"       : "Jumlah Aktivitas",
    "screen_time_hours"    : "Screen Time/Hari (jam)",
    "commute_minutes"      : "Waktu Perjalanan (menit)",
    "family_support"       : "Dukungan Keluarga (1–5)",
    "friend_support"       : "Dukungan Teman (1–5)",
    "teacher_support"      : "Dukungan Guru (1–5)",
    "self_rated_stress"    : "Tingkat Stres Mandiri (1–5)",
}

# ============================================================
# LOAD / TRAIN MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """Load model dari model.pkl, atau latih ulang jika tidak ada / tidak kompatibel."""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            # Validasi bahwa model punya semua key yang dibutuhkan
            required_keys = ['model', 'scaler', 'encoders', 'features', 'metrics']
            if all(k in data for k in required_keys):
                return data
        except Exception:
            pass

    # ── Latih ulang jika model.pkl tidak ada / tidak kompatibel ──
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    if not os.path.exists(DATASET_PATH):
        return None

    df = pd.read_csv(DATASET_PATH)

    # Buat label burnout_level
    def map_score(s):
        if s <= 2: return 'Low'
        elif s == 3: return 'Medium'
        else: return 'High'

    df['burnout_level'] = df['burnout_score'].apply(map_score)
    cols_to_drop = ['student_id', 'burnout_score', 'high_burnout']
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
    df = df.dropna().drop_duplicates()

    # Encode
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('burnout_level', axis=1)
    y = df['burnout_level']
    features = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Gunakan integer labels dari LabelEncoder (bukan string)
    le_target  = label_encoders.get('burnout_level')
    if le_target is not None:
        int_labels = list(range(len(le_target.classes_)))
        str_labels = list(le_target.classes_)
    else:
        int_labels = sorted(y_test.unique())
        str_labels = [str(l) for l in int_labels]
    cm = confusion_matrix(y_test, y_pred, labels=int_labels)

    cv_scores = cross_val_score(rf, X_train_s, y_train, cv=5, scoring='accuracy')
    fi = dict(zip(features, rf.feature_importances_))

    model_data = {
        'model'   : rf,
        'scaler'  : scaler,
        'encoders': label_encoders,
        'features': features,
        'metrics' : {
            'accuracy' : acc,
            'precision': prec,
            'recall'   : rec,
            'f1'       : f1,
            'cv_mean'  : cv_scores.mean(),
            'cv_std'   : cv_scores.std(),
        },
        'confusion_matrix'   : cm,
        'confusion_labels'   : str_labels,
        'feature_importances': fi,
        'class_labels'       : str_labels,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    return model_data

@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    def map_score(s):
        if s <= 2: return 'Low'
        elif s == 3: return 'Medium'
        else: return 'High'
    df['burnout_level'] = df['burnout_score'].apply(map_score)
    return df

# ============================================================
# LOAD MODEL & DATASET
# ============================================================
with st.spinner("🔄 Memuat model..."):
    model_data = load_or_train_model()

df_raw = load_dataset()

# ============================================================
# SIDEBAR — NAVIGASI
# ============================================================
with st.sidebar:
    st.markdown("## 🔥 Burnout Predictor")
    st.markdown("*Powered by Random Forest*")
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "🔍 Prediksi Burnout", "📊 Analisis Dataset", "📈 Performa Model", "ℹ️ Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:rgba(255,255,255,0.4)'>Dataset: student_burnout.csv<br>"
        "Model: Random Forest Classifier<br>"
        "Algoritma: Ensemble Learning</small>",
        unsafe_allow_html=True
    )

# ============================================================
# PAGE 1 — BERANDA
# ============================================================
if page == "🏠 Beranda":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🔥 Prediksi Tingkat Burnout Siswa</div>
        <div class="hero-subtitle">
            Implementasi Algoritma Random Forest untuk Prediksi Tingkat Burnout<br>
            Berdasarkan Faktor Akademik dan Gaya Hidup
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Statistik Dataset
    if df_raw is not None:
        col1, col2, col3, col4 = st.columns(4)
        total = len(df_raw)
        low_n    = (df_raw['burnout_level'] == 'Low').sum()
        med_n    = (df_raw['burnout_level'] == 'Medium').sum()
        high_n   = (df_raw['burnout_level'] == 'High').sum()

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total:,}</div>
                <div class="metric-label">Total Data Siswa</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background:linear-gradient(135deg,#10b981,#059669);-webkit-background-clip:text;">
                    {low_n}</div>
                <div class="metric-label">🟢 Burnout Rendah</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background:linear-gradient(135deg,#f59e0b,#d97706);-webkit-background-clip:text;">
                    {med_n}</div>
                <div class="metric-label">🟡 Burnout Sedang</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background:linear-gradient(135deg,#ef4444,#dc2626);-webkit-background-clip:text;">
                    {high_n}</div>
                <div class="metric-label">🔴 Burnout Tinggi</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Info Proyek
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown('<div class="section-title">📖 Tentang Proyek</div>', unsafe_allow_html=True)
        st.markdown("""
        Proyek ini mengimplementasikan algoritma **Random Forest Classifier** untuk memprediksi
        tingkat burnout siswa menjadi tiga kategori:

        - 🟢 **Low** — Burnout rendah, kondisi baik
        - 🟡 **Medium** — Burnout sedang, perlu perhatian
        - 🔴 **High** — Burnout tinggi, perlu intervensi segera

        Model dilatih menggunakan **14 fitur** yang mencakup faktor akademik (jam belajar,
        jumlah tes, kelas) dan faktor gaya hidup (jam tidur, screen time, dukungan sosial).
        """)

    with col_right:
        st.markdown('<div class="section-title">🗂️ Fitur Dataset</div>', unsafe_allow_html=True)
        features_info = [
            ("📚", "Akademik", "Grade, Homework Hours, Tests/Week"),
            ("💤", "Tidur", "Sleep Hours, Sleep Quality"),
            ("📱", "Digital", "Screen Time"),
            ("🏃", "Aktivitas", "Extracurricular, Num Activities"),
            ("🚌", "Komuter", "Commute Minutes"),
            ("💛", "Sosial", "Family, Friend, Teacher Support"),
            ("😰", "Stres", "Self-rated Stress"),
        ]
        for icon, cat, desc in features_info:
            st.markdown(f"""
            <div class="rec-item">
                {icon} <b>{cat}</b>: <small>{desc}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Alur Sistem</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Load Data", "Memuat dataset student_burnout.csv"),
        ("2", "Preprocessing", "Encoding, cleaning, scaling"),
        ("3", "Training", "Random Forest dengan 200 estimators"),
        ("4", "Evaluasi", "Accuracy, F1, Confusion Matrix, CV"),
        ("5", "Prediksi", "Input data baru → prediksi level burnout"),
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
    st.markdown('<div class="hero-title" style="text-align:center;color:#fff;font-size:1.6rem;margin-bottom:8px;">🔍 Prediksi Tingkat Burnout Siswa</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:rgba(255,255,255,0.5);margin-bottom:32px;">Isi semua data berikut untuk mendapatkan prediksi tingkat burnout</div>', unsafe_allow_html=True)

    if model_data is None:
        st.error("❌ Model tidak tersedia. Pastikan dataset ada di folder `dataset/`.")
    else:
        model    = model_data['model']
        scaler   = model_data['scaler']
        encoders = model_data['encoders']
        features = model_data['features']

        with st.form("burnout_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="section-title">📚 Data Akademik</div>', unsafe_allow_html=True)
                grade           = st.selectbox("Kelas (Grade)", [9, 10, 11, 12], index=1)
                homework_hours  = st.number_input("Jam Mengerjakan PR/Hari", min_value=0.0, max_value=12.0, value=2.5, step=0.5)
                tests_per_week  = st.number_input("Jumlah Tes/Minggu", min_value=0, max_value=10, value=2)
                ext_hours       = st.number_input("Jam Kegiatan Ekskul/Hari", min_value=0.0, max_value=8.0, value=1.0, step=0.5)
                num_activities  = st.number_input("Jumlah Aktivitas Ekskul", min_value=0, max_value=10, value=2)
                commute_minutes = st.number_input("Waktu Perjalanan ke Sekolah (menit)", min_value=0.0, max_value=180.0, value=30.0, step=5.0)

            with col2:
                st.markdown('<div class="section-title">💤 Gaya Hidup</div>', unsafe_allow_html=True)
                gender          = st.selectbox("Jenis Kelamin", ["Male", "Female", "Nonbinary"])
                sleep_hours     = st.number_input("Jam Tidur/Hari", min_value=2.0, max_value=12.0, value=7.0, step=0.5)
                sleep_quality   = st.slider("Kualitas Tidur (1=Sangat Buruk, 5=Sangat Baik)", 1, 5, 3)
                screen_time     = st.number_input("Screen Time/Hari (jam)", min_value=0.0, max_value=16.0, value=4.0, step=0.5)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="rec-item" style="font-size:0.8rem;">
                    💡 Rekomendasi WHO: Tidur 8–10 jam/malam untuk remaja
                </div>""", unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="section-title">💛 Dukungan & Stres</div>', unsafe_allow_html=True)
                family_support  = st.slider("Dukungan Keluarga (1=Rendah, 5=Tinggi)", 1, 5, 3)
                friend_support  = st.slider("Dukungan Teman (1=Rendah, 5=Tinggi)", 1, 5, 3)
                teacher_support = st.slider("Dukungan Guru (1=Rendah, 5=Tinggi)", 1, 5, 3)
                self_stress     = st.slider("Tingkat Stres Mandiri (1=Rendah, 5=Sangat Tinggi)", 1, 5, 3)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="rec-item" style="font-size:0.8rem;">
                    ⚠️ Stres tinggi (4–5) sangat berkorelasi dengan burnout tinggi
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "🔍 Prediksi Tingkat Burnout",
                use_container_width=True,
                type="primary"
            )

        # ── HASIL PREDIKSI ──
        if submitted:
            input_dict = {
                'grade'                : grade,
                'gender'               : gender,
                'sleep_hours'          : sleep_hours,
                'sleep_quality'        : sleep_quality,
                'homework_hours'       : homework_hours,
                'tests_per_week'       : tests_per_week,
                'extracurricular_hours': ext_hours,
                'num_activities'       : num_activities,
                'screen_time_hours'    : screen_time,
                'commute_minutes'      : commute_minutes,
                'family_support'       : family_support,
                'friend_support'       : friend_support,
                'teacher_support'      : float(teacher_support),
                'self_rated_stress'    : self_stress,
            }

            df_input = pd.DataFrame([input_dict])

            # Encode kolom kategorik
            for col in df_input.select_dtypes(include=['object']).columns:
                if col in encoders:
                    df_input[col] = encoders[col].transform(df_input[col])

            try:
                df_input = df_input[features]
                input_scaled = scaler.transform(df_input)

                prediction   = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                
                # Decode model classes back to string labels
                if 'burnout_level' in encoders:
                    class_labels = encoders['burnout_level'].inverse_transform(model.classes_)
                else:
                    class_labels = model.classes_

                # Decode label prediction jika perlu
                if 'burnout_level' in encoders:
                    pred_label = encoders['burnout_level'].inverse_transform([prediction])[0]
                else:
                    pred_label = prediction

                prob_dict = dict(zip(class_labels, probabilities))
                ordered_labels = ['Low', 'Medium', 'High']

                st.markdown("---")
                st.markdown("## 📊 Hasil Prediksi")

                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    badge_class = f"badge-{pred_label.lower()}"
                    emoji_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                    emoji = emoji_map.get(pred_label, "⚪")

                    st.markdown(f"""
                    <div class="result-card" style="text-align:center">
                        <div style="font-size:4rem;">{emoji}</div>
                        <div style="font-size:1rem;color:rgba(255,255,255,0.6);margin:8px 0;">Tingkat Burnout</div>
                        <div><span class="{badge_class}">{pred_label.upper()}</span></div>
                        <div style="margin-top:16px;font-size:0.9rem;color:rgba(255,255,255,0.5);">
                            Keyakinan Model: <b style="color:#a78bfa">{prob_dict.get(pred_label, 0)*100:.1f}%</b>
                        </div>
                    </div>""", unsafe_allow_html=True)

                with res_col2:
                    # Gauge Chart Probabilitas
                    fig = go.Figure()
                    colors = [BURNOUT_COLORS[l] for l in ordered_labels]
                    probs  = [prob_dict.get(l, 0) * 100 for l in ordered_labels]

                    fig.add_trace(go.Bar(
                        x=ordered_labels,
                        y=probs,
                        marker_color=colors,
                        text=[f"{p:.1f}%" for p in probs],
                        textposition='outside',
                        textfont=dict(color='white', size=13),
                    ))
                    fig.update_layout(
                        title=dict(text="Distribusi Probabilitas Prediksi", font=dict(color='white', size=14)),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.03)',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(range=[0, 110], showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                                   title="Probabilitas (%)"),
                        margin=dict(t=50, b=20, l=20, r=20),
                        height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Rekomendasi
                st.markdown("---")
                st.markdown("### 💡 Rekomendasi Personal")

                if pred_label == 'Low':
                    st.success(
                        "**Kondisi baik!** Siswa menunjukkan tanda-tanda burnout yang rendah. "
                        "Pertahankan kebiasaan positif yang sudah ada."
                    )
                    recs = [
                        "✅ Pertahankan jam tidur yang cukup (8–10 jam untuk remaja)",
                        "✅ Lanjutkan aktivitas ekskul yang seimbang",
                        "✅ Jaga komunikasi baik dengan keluarga dan teman",
                        "📖 Buat jadwal belajar yang terstruktur agar tetap produktif",
                    ]
                    for r in recs:
                        st.markdown(f'<div class="rec-item">{r}</div>', unsafe_allow_html=True)

                elif pred_label == 'Medium':
                    st.warning(
                        "**Perhatian!** Siswa menunjukkan tanda-tanda burnout sedang. "
                        "Perlu penyesuaian gaya belajar dan gaya hidup."
                    )
                    recs = []
                    if sleep_hours < 7:
                        recs.append("⚠️ Jam tidur di bawah ideal — targetkan 8–10 jam/malam")
                    if screen_time > 5:
                        recs.append("⚠️ Screen time cukup tinggi — kurangi waktu layar sebelum tidur")
                    if self_stress >= 4:
                        recs.append("⚠️ Tingkat stres tinggi — coba teknik relaksasi atau meditasi singkat")
                    if family_support <= 2 or friend_support <= 2:
                        recs.append("⚠️ Dukungan sosial rendah — cari waktu untuk berinteraksi dengan orang-orang terdekat")
                    recs += [
                        "💬 Bicarakan beban dengan guru BK atau konselor sekolah",
                        "📅 Prioritaskan tugas dan buat daftar to-do yang realistis",
                    ]
                    for r in recs:
                        st.markdown(f'<div class="rec-item-warn">{r}</div>', unsafe_allow_html=True)

                else:  # High
                    st.error(
                        "**⚠️ Perhatian Serius!** Siswa berisiko tinggi mengalami burnout. "
                        "Intervensi segera sangat dianjurkan."
                    )
                    recs = [
                        "🚨 Segera konsultasikan kondisi ke guru BK / konselor sekolah",
                        "🚨 Kurangi beban ekskul jika terlalu banyak aktivitas",
                        "💤 Prioritaskan kualitas tidur sebagai kebutuhan utama",
                        "📵 Batasi screen time maksimal 2–3 jam di luar kegiatan belajar",
                        "❤️ Ceritakan perasaan kepada orang tua atau teman kepercayaan",
                        "📚 Bicarakan kepada guru tentang tekanan tugas yang terlalu berat",
                    ]
                    for r in recs:
                        st.markdown(f'<div class="rec-item-warn">{r}</div>', unsafe_allow_html=True)

                # Faktor risiko dari input
                st.markdown("---")
                st.markdown("### 🎯 Analisis Faktor Risiko dari Input Anda")
                risk_factors = []
                if sleep_hours < 6:
                    risk_factors.append(("💤 Jam Tidur Kurang", f"{sleep_hours} jam (< 6 jam)", "high"))
                elif sleep_hours < 7.5:
                    risk_factors.append(("💤 Jam Tidur Sedikit Kurang", f"{sleep_hours} jam", "medium"))
                if self_stress >= 4:
                    risk_factors.append(("😰 Stres Tinggi", f"Skor {self_stress}/5", "high"))
                elif self_stress == 3:
                    risk_factors.append(("😰 Stres Sedang", f"Skor {self_stress}/5", "medium"))
                if screen_time > 6:
                    risk_factors.append(("📱 Screen Time Berlebihan", f"{screen_time} jam/hari", "high"))
                elif screen_time > 4:
                    risk_factors.append(("📱 Screen Time Cukup Tinggi", f"{screen_time} jam/hari", "medium"))
                if homework_hours > 5:
                    risk_factors.append(("📝 Beban PR Tinggi", f"{homework_hours} jam/hari", "high"))
                if family_support <= 2:
                    risk_factors.append(("👨‍👩‍👧 Dukungan Keluarga Rendah", f"Skor {family_support}/5", "high"))
                if friend_support <= 2:
                    risk_factors.append(("👫 Dukungan Teman Rendah", f"Skor {friend_support}/5", "medium"))

                if risk_factors:
                    for label, value, level in risk_factors:
                        css = "rec-item-warn" if level == "high" else "rec-item"
                        st.markdown(f'<div class="{css}"><b>{label}</b>: {value}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="rec-item">✅ Tidak ada faktor risiko signifikan yang terdeteksi dari input Anda.</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Terjadi error saat prediksi: {e}")

# ============================================================
# PAGE 3 — ANALISIS DATASET (EDA)
# ============================================================
elif page == "📊 Analisis Dataset":
    st.markdown('<div class="hero-title" style="color:#fff;font-size:1.6rem;margin-bottom:24px;">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if df_raw is None:
        st.error("❌ Dataset tidak ditemukan.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Distribusi Burnout", "🔗 Korelasi", "🧩 Fitur vs Burnout", "📋 Statistik Deskriptif"]
        )

        # ── Tab 1: Distribusi ──
        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                # Distribusi burnout_level
                counts = df_raw['burnout_level'].value_counts()
                ordered = ['Low', 'Medium', 'High']
                vals = [counts.get(l, 0) for l in ordered]
                colors = [BURNOUT_COLORS[l] for l in ordered]

                fig = go.Figure(go.Bar(
                    x=ordered, y=vals, marker_color=colors,
                    text=vals, textposition='outside',
                    textfont=dict(color='white', size=13),
                ))
                fig.update_layout(
                    title="Distribusi Kelas Burnout Level",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
                    font=dict(color='white'), xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    margin=dict(t=50, b=20), height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Pie chart
                fig2 = go.Figure(go.Pie(
                    labels=ordered, values=vals,
                    marker_colors=colors,
                    hole=0.45,
                    textinfo='label+percent',
                    textfont=dict(color='white', size=13),
                ))
                fig2.update_layout(
                    title="Proporsi Burnout Level",
                    paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                    margin=dict(t=50, b=20), height=350,
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Distribusi burnout_score
            score_counts = df_raw['burnout_score'].value_counts().sort_index()
            fig3 = go.Figure(go.Bar(
                x=[str(i) for i in score_counts.index],
                y=score_counts.values,
                marker_color='#6366f1',
                text=score_counts.values, textposition='outside',
                textfont=dict(color='white'),
            ))
            fig3.update_layout(
                title="Distribusi Burnout Score (1–5)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
                font=dict(color='white'), xaxis=dict(showgrid=False, title="Burnout Score"),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Jumlah Siswa"),
                margin=dict(t=50, b=20), height=350,
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Distribusi per gender
            gender_burnout = df_raw.groupby(['gender', 'burnout_level']).size().reset_index(name='count')
            fig4 = px.bar(
                gender_burnout, x='gender', y='count', color='burnout_level',
                color_discrete_map=BURNOUT_COLORS, barmode='group',
                title="Distribusi Burnout per Jenis Kelamin",
            )
            fig4.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
                font=dict(color='white'), xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                margin=dict(t=50, b=20), height=350,
            )
            st.plotly_chart(fig4, use_container_width=True)

        # ── Tab 2: Korelasi ──
        with tab2:
            from sklearn.preprocessing import LabelEncoder
            df_enc = df_raw.drop(columns=['burnout_level']).copy()
            for col in df_enc.select_dtypes(include='object').columns:
                le = LabelEncoder()
                df_enc[col] = le.fit_transform(df_enc[col].astype(str))

            numeric_cols = df_enc.select_dtypes(include='number').columns.tolist()
            corr = df_enc[numeric_cols].corr()

            fig_hm = px.imshow(
                corr, text_auto='.2f', aspect='auto',
                color_continuous_scale='RdYlGn',
                title="Heatmap Korelasi Fitur",
            )
            fig_hm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                margin=dict(t=60, b=20), height=600,
                coloraxis_colorbar=dict(tickfont=dict(color='white'), title=dict(font=dict(color='white'))),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            # Korelasi tertinggi dengan burnout_score
            st.markdown("#### Korelasi dengan Burnout Score")
            corr_target = corr['burnout_score'].drop('burnout_score').sort_values(key=abs, ascending=False)
            fig_bar = go.Figure(go.Bar(
                x=corr_target.values,
                y=corr_target.index,
                orientation='h',
                marker_color=['#ef4444' if v > 0 else '#10b981' for v in corr_target.values],
                text=[f"{v:.3f}" for v in corr_target.values],
                textposition='outside',
                textfont=dict(color='white'),
            ))
            fig_bar.update_layout(
                title="Korelasi Setiap Fitur dengan Burnout Score",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
                font=dict(color='white'), xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=False), margin=dict(t=50, b=20, l=170), height=450,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Tab 3: Fitur vs Burnout ──
        with tab3:
            numeric_features = [
                'sleep_hours', 'homework_hours', 'screen_time_hours',
                'extracurricular_hours', 'commute_minutes', 'tests_per_week'
            ]
            selected_feature = st.selectbox(
                "Pilih fitur untuk dianalisis:",
                numeric_features,
                format_func=lambda x: FEATURE_LABELS.get(x, x)
            )

            col1, col2 = st.columns(2)
            with col1:
                fig_box = px.box(
                    df_raw, x='burnout_level', y=selected_feature,
                    color='burnout_level', color_discrete_map=BURNOUT_COLORS,
                    category_orders={'burnout_level': ['Low', 'Medium', 'High']},
                    title=f"Box Plot: {FEATURE_LABELS.get(selected_feature, selected_feature)} vs Burnout",
                )
                fig_box.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
                    font=dict(color='white'), showlegend=False,
                    margin=dict(t=50, b=20), height=400,
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with col2:
                fig_vio = px.violin(
                    df_raw, x='burnout_level', y=selected_feature,
                    color='burnout_level', color_discrete_map=BURNOUT_COLORS,
                    category_orders={'burnout_level': ['Low', 'Medium', 'High']},
                    box=True,
                    title=f"Violin Plot: {FEATURE_LABELS.get(selected_feature, selected_feature)} vs Burnout",
                )
                fig_vio.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
                    font=dict(color='white'), showlegend=False,
                    margin=dict(t=50, b=20), height=400,
                )
                st.plotly_chart(fig_vio, use_container_width=True)

            # Histogram
            fig_hist = px.histogram(
                df_raw, x=selected_feature, color='burnout_level',
                color_discrete_map=BURNOUT_COLORS, barmode='overlay',
                opacity=0.75,
                category_orders={'burnout_level': ['Low', 'Medium', 'High']},
                title=f"Distribusi {FEATURE_LABELS.get(selected_feature, selected_feature)} per Level Burnout",
                nbins=25,
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
                font=dict(color='white'), xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                margin=dict(t=50, b=20), height=380,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Distribusi per kelas
            st.markdown("#### Distribusi Burnout per Kelas (Grade)")
            grade_burnout = df_raw.groupby(['grade', 'burnout_level']).size().reset_index(name='count')
            fig_grade = px.bar(
                grade_burnout, x='grade', y='count', color='burnout_level',
                color_discrete_map=BURNOUT_COLORS, barmode='group',
                category_orders={'burnout_level': ['Low', 'Medium', 'High']},
                title="Burnout Level per Kelas (Grade 9–12)",
                labels={'grade': 'Kelas', 'count': 'Jumlah Siswa'},
            )
            fig_grade.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
                font=dict(color='white'), xaxis=dict(showgrid=False, tickmode='array',
                tickvals=[9,10,11,12], ticktext=['Grade 9','Grade 10','Grade 11','Grade 12']),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                margin=dict(t=50, b=20), height=380,
            )
            st.plotly_chart(fig_grade, use_container_width=True)

        # ── Tab 4: Statistik ──
        with tab4:
            st.markdown("#### Statistik Deskriptif Dataset")
            st.dataframe(
                df_raw.drop(columns=['burnout_level']).describe().round(3),
                use_container_width=True
            )
            st.markdown("#### Missing Values")
            mv = df_raw.isnull().sum()
            mv_df = pd.DataFrame({'Kolom': mv.index, 'Missing Values': mv.values, 'Persentase (%)': (mv.values / len(df_raw) * 100).round(2)})
            st.dataframe(mv_df, use_container_width=True, hide_index=True)

# ============================================================
# PAGE 4 — PERFORMA MODEL
# ============================================================
elif page == "📈 Performa Model":
    st.markdown('<div class="hero-title" style="color:#fff;font-size:1.6rem;margin-bottom:24px;">📈 Performa Model Random Forest</div>', unsafe_allow_html=True)

    if model_data is None:
        st.error("❌ Model tidak tersedia.")
    else:
        metrics = model_data.get('metrics', {})
        cm      = model_data.get('confusion_matrix')
        fi      = model_data.get('feature_importances', {})

        # ── Metric Cards ──
        st.markdown("### 🎯 Metrik Evaluasi")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        metric_items = [
            (mc1, "Accuracy", metrics.get('accuracy', 0), "#a78bfa"),
            (mc2, "Precision", metrics.get('precision', 0), "#60a5fa"),
            (mc3, "Recall", metrics.get('recall', 0), "#34d399"),
            (mc4, "F1-Score", metrics.get('f1', 0), "#fb923c"),
            (mc5, "CV Mean", metrics.get('cv_mean', 0), "#f472b6"),
        ]
        for col, label, val, color in metric_items:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="background:linear-gradient(135deg,{color},{color}aa);-webkit-background-clip:text;">
                        {val*100:.2f}%
                    </div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        cv_std = metrics.get('cv_std', 0)
        st.markdown(f"""
        <div style="text-align:center;margin:16px 0;color:rgba(255,255,255,0.5);font-size:0.85rem;">
            5-Fold Cross Validation: {metrics.get('cv_mean',0)*100:.2f}% ± {cv_std*100:.2f}%
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        # ── Confusion Matrix ──
        with col1:
            st.markdown("### 🔢 Confusion Matrix")
            if cm is not None:
                class_labels = model_data.get('confusion_labels', model_data.get('class_labels', ['High', 'Low', 'Medium']))
                fig_cm = px.imshow(
                    cm, x=class_labels, y=class_labels,
                    text_auto=True, aspect='auto',
                    color_continuous_scale='Purples',
                    labels=dict(x="Prediksi", y="Aktual"),
                    title="Confusion Matrix (Aktual vs Prediksi)",
                )
                fig_cm.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white', size=13),
                    margin=dict(t=60, b=20), height=380,
                    coloraxis_colorbar=dict(tickfont=dict(color='white')),
                    xaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
                    yaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix tidak tersedia. Jalankan ulang train_model.py")

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
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
                    font=dict(color='white'), xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=False, tickfont=dict(size=10)),
                    margin=dict(t=50, b=20, l=10, r=80), height=420,
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance tidak tersedia.")

        # ── Penjelasan ──
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
            - ✅ Mampu menangani non-linearitas
            - ✅ Memberikan feature importance
            - ✅ Tidak memerlukan asumsi distribusi data
            - ✅ Ensemble → prediksi lebih stabil dari Decision Tree tunggal
            """)

# ============================================================
# PAGE 5 — TENTANG
# ============================================================
elif page == "ℹ️ Tentang":
    st.markdown('<div class="hero-title" style="color:#fff;font-size:1.6rem;margin-bottom:24px;">ℹ️ Tentang Proyek</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ## Implementasi Algoritma Random Forest untuk Prediksi Tingkat Burnout pada Siswa Berdasarkan Faktor Akademik dan Gaya Hidup

        ---

        ### 📌 Deskripsi
        Proyek *Machine Learning* berbasis **Supervised Learning (Classification)** untuk memprediksi
        tingkat burnout siswa menjadi tiga kategori (**Low / Medium / High**) menggunakan algoritma
        **Random Forest Classifier**.

        *Burnout* siswa adalah kondisi kelelahan emosional, fisik, dan mental akibat tuntutan akademik
        jangka panjang. Deteksi dini melalui pendekatan Machine Learning memungkinkan intervensi
        yang lebih proaktif dan tepat sasaran.

        ---

        ### 🗂️ Dataset
        - **File**: `student_burnout.csv`
        - **Ukuran**: 2.000 data siswa
        - **Fitur**: 14 fitur (akademik + gaya hidup + sosial)
        - **Target**: Burnout Level (Low / Medium / High) — diturunkan dari `burnout_score` (1–5)
          - 🟢 **Low**: Score 1–2
          - 🟡 **Medium**: Score 3
          - 🔴 **High**: Score 4–5

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
        │   └── student_burnout.csv
        ├── images/         # Plot EDA
        ├── laporan/        # Laporan proyek
        ├── notebook.ipynb  # EDA & Visualisasi
        ├── preprocessing.py
        ├── train_model.py
        ├── app.py          # Streamlit App
        ├── model.pkl       # Model terlatih
        └── requirements.txt
        ```

        ---

        ### 🚀 Cara Menjalankan
        **1. Install dependencies:**
        ```bash
        pip install -r requirements.txt
        ```
        **2. (Opsional) Train ulang model:**
        ```bash
        python train_model.py
        ```
        **3. Jalankan aplikasi:**
        ```bash
        streamlit run app.py
        ```

        ---

        ### 📚 Referensi
        - Breiman, L. (2001). Random Forests. *Machine Learning*.
        - Salmela-Aro et al. (2022). Student burnout during academic year. *CHB*.
        - Scikit-learn documentation: RandomForestClassifier
        """)
