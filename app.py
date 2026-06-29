# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import pickle
import os
import sys

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Student Burnout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# TRAIN MODEL ON-DEMAND (jika model.pkl belum ada)
# ============================================================
MODEL_PATH = "model.pkl"
DATASET_PATH = "dataset/student_mental_health_burnout.csv"

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """
    Load model dari file jika ada.
    Jika tidak ada, latih model dari dataset dan simpan.
    """
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data['model'], data['scaler'], data['encoders'], data['features']
    
    # Import dependencies untuk training
    import numpy as np  # pyrefly: ignore [missing-import]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if not os.path.exists(DATASET_PATH):
        return None, None, None, None
    
    # Load & preprocess data
    df = pd.read_csv(DATASET_PATH)
    
    # Drop kolom tidak diperlukan
    if 'student_id' in df.columns:
        df = df.drop('student_id', axis=1)
    
    # Hapus missing values dan duplikat
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Encode semua kolom kategorik
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Split features dan target
    X = df.drop('burnout_level', axis=1)
    y = df['burnout_level']
    features = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale fitur
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    # Latih model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)
    
    # Simpan model
    model_data = {
        'model': model,
        'scaler': scaler,
        'encoders': label_encoders,
        'features': features
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    
    return model, scaler, label_encoders, features


# ============================================================
# LOAD MODEL (dengan progress indicator)
# ============================================================
with st.spinner("🔄 Memuat model... (Proses pertama kali bisa memakan waktu 1-2 menit)"):
    model, scaler, encoders, feature_cols = load_or_train_model()

# ============================================================
# HEADER
# ============================================================
st.title("🎓 Student Burnout Prediction System")
st.markdown("""
Aplikasi ini menggunakan model *Machine Learning* (**Random Forest**) untuk memprediksi 
tingkat burnout mahasiswa berdasarkan faktor akademik dan gaya hidup.
Isi formulir di bawah dan klik **Prediksi** untuk melihat hasilnya.
""")
st.divider()

# ============================================================
# FORM INPUT & PREDIKSI
# ============================================================
if model is None:
    st.error(
        "❌ **Dataset tidak ditemukan.** Pastikan file "
        "`dataset/student_mental_health_burnout.csv` ada di dalam repositori GitHub Anda."
    )
else:
    # Mapping nama lengkap jurusan → singkatan yang dipakai di dataset
    COURSE_MAP = {
        "Bachelor of Technology (BTech)": "BTech",
        "Bachelor of Computer Applications (BCA)": "BCA",
        "Bachelor of Science (BSc)": "BSc",
        "Master of Business Administration (MBA)": "MBA",
        "Master of Computer Applications (MCA)": "MCA",
        "Bachelor of Business Administration (BBA)": "BBA",
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📚 Info Akademik & Personal")
        age = st.number_input("Usia (Age)", min_value=15, max_value=40, value=20)
        gender = st.selectbox("Jenis Kelamin (Gender)", ["Male", "Female", "Other"])
        course_display = st.selectbox("Jurusan (Course)", list(COURSE_MAP.keys()))
        course = COURSE_MAP[course_display]  # konversi ke singkatan untuk model
        year = st.selectbox("Tahun Kuliah (Year)", ["1st", "2nd", "3rd", "4th"])
        cgpa = st.number_input("IPK (CGPA)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        attendance = st.number_input("Kehadiran (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

    with col2:
        st.subheader("🌙 Gaya Hidup")
        daily_study_hours = st.number_input("Jam Belajar/Hari", min_value=0.0, max_value=24.0, value=4.0)
        daily_sleep_hours = st.number_input("Jam Tidur/Hari", min_value=0.0, max_value=24.0, value=7.0)
        screen_time_hours = st.number_input("Screen Time (Jam/Hari)", min_value=0.0, max_value=24.0, value=5.0)
        physical_activity = st.number_input("Aktivitas Fisik (Jam/Hari)", min_value=0.0, max_value=10.0, value=1.0)
        sleep_quality = st.selectbox("Kualitas Tidur", ["Poor", "Average", "Good"])
        internet_quality = st.selectbox("Kualitas Internet", ["Poor", "Average", "Good"])

    with col3:
        st.subheader("🧠 Kondisi Psikologis")
        stress_level = st.selectbox("Tingkat Stres", ["Low", "Medium", "High"])
        anxiety_score = st.slider("Skor Kecemasan (0-10)", 0, 10, 5)
        depression_score = st.slider("Skor Depresi (0-10)", 0, 10, 3)
        academic_pressure = st.slider("Tekanan Akademik (0-10)", 0, 10, 6)
        financial_stress = st.slider("Tekanan Finansial (0-10)", 0, 10, 4)
        social_support = st.slider("Dukungan Sosial (0-10)", 0, 10, 7)

    st.divider()

    predict_button = st.button("🔍 Prediksi Tingkat Burnout", type="primary", use_container_width=True)

    if predict_button:
        input_data = {
            'age': [age],
            'gender': [gender],
            'course': [course],
            'year': [year],
            'daily_study_hours': [daily_study_hours],
            'daily_sleep_hours': [daily_sleep_hours],
            'screen_time_hours': [screen_time_hours],
            'stress_level': [stress_level],
            'anxiety_score': [anxiety_score],
            'depression_score': [depression_score],
            'academic_pressure_score': [academic_pressure],
            'financial_stress_score': [financial_stress],
            'social_support_score': [social_support],
            'physical_activity_hours': [physical_activity],
            'sleep_quality': [sleep_quality],
            'attendance_percentage': [attendance],
            'cgpa': [cgpa],
            'internet_quality': [internet_quality]
        }

        df_input = pd.DataFrame(input_data)

        # Apply encoding pada fitur kategorik
        for col in df_input.select_dtypes(include=['object']).columns:
            if col in encoders:
                df_input[col] = encoders[col].transform(df_input[col])

        try:
            # Urutkan kolom sesuai training
            df_input = df_input[feature_cols]
            input_scaled = scaler.transform(df_input)

            # Prediksi
            prediction = model.predict(input_scaled)[0]

            # Decode label
            if 'burnout_level' in encoders:
                pred_label = encoders['burnout_level'].inverse_transform([prediction])[0]
            else:
                # Fallback: LabelEncoder mengurutkan secara alfabetis → High=0, Low=1, Medium=2
                label_map = {0: "High", 1: "Low", 2: "Medium"}
                pred_label = label_map.get(prediction, str(prediction))

            # Tampilkan hasil
            st.markdown("## 📊 Hasil Prediksi")

            if pred_label == 'Low':
                st.success(
                    "### 🟢 Tingkat Burnout: **RENDAH (Low)**\n\n"
                    "**Selamat!** Kondisi Anda saat ini cukup baik. Pertahankan keseimbangan antara "
                    "belajar, istirahat, dan aktivitas sosial untuk menjaga kesehatan mental Anda."
                )
            elif pred_label == 'Medium':
                st.warning(
                    "### 🟡 Tingkat Burnout: **SEDANG (Medium)**\n\n"
                    "Anda menunjukkan beberapa tanda burnout. Pertimbangkan untuk:\n"
                    "- Memperbaiki kualitas tidur\n"
                    "- Mengurangi screen time\n"
                    "- Berbagi beban dengan teman atau konselor akademik"
                )
            else:
                st.error(
                    "### 🔴 Tingkat Burnout: **TINGGI (High)**\n\n"
                    "Perhatian! Anda berisiko tinggi mengalami burnout. Segera lakukan:\n"
                    "- Konsultasikan kondisi Anda ke konselor kampus\n"
                    "- Prioritaskan tidur yang berkualitas (7-8 jam)\n"
                    "- Kurangi tekanan akademik yang berlebihan\n"
                    "- Cari dukungan sosial dari keluarga dan teman"
                )

        except Exception as e:
            st.error(f"❌ Terjadi error saat prediksi: {e}")
