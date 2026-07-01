import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATASET_PATH = 'dataset/student_mental_health_burnout_100k_fixed.xlsx'

# Kolom fitur yang digunakan saat training (urutan harus sama persis)
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

# Kolom target
TARGET_COLUMN = 'burnout_level'

# Kolom yang di-encode menggunakan LabelEncoder
CATEGORICAL_COLS = ['gender', 'social_support']


def load_data(file_path=DATASET_PATH):
    """Load dataset dari file Excel atau CSV."""
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """
    Lakukan tahapan preprocessing:
    1. Pilih hanya kolom fitur + target
    2. Handle missing values (drop baris NaN)
    3. Drop duplikat
    4. Encode kolom kategorik (gender, social_support, burnout_level)
    5. Kembalikan df_processed dan dict encoders

    Returns:
        df_processed (pd.DataFrame): DataFrame yang sudah diproses
        label_encoders (dict): Dictionary berisi LabelEncoder untuk setiap kolom kategorik
    """
    # Pilih kolom yang relevan
    all_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    available_cols = [c for c in all_cols if c in df.columns]
    df = df[available_cols].copy()

    # Handle missing values
    df = df.dropna()

    # Drop duplikat
    df = df.drop_duplicates()

    # Encode kolom kategorik (termasuk target)
    label_encoders = {}
    encode_cols = CATEGORICAL_COLS + [TARGET_COLUMN]

    for col in encode_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df, label_encoders


def split_and_scale(df, target_col=TARGET_COLUMN):
    """Split data ke train/test set dan lakukan StandardScaler pada fitur."""
    X = df[FEATURE_COLUMNS]
    y = df[target_col]

    # Train-test split 80:20 dengan stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisasi fitur numerik
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS)
    X_test_df  = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS)

    return X_train_df, X_test_df, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    df_processed, encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_processed)

    print("Preprocessing selesai.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}")
    print(f"Encoders: {list(encoders.keys())}")
